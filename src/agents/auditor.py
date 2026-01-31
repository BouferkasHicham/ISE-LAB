"""
Auditor Agent - Code Analysis and Refactoring Plan Generation.

The Auditor reads code, runs static analysis, and produces a comprehensive
refactoring plan for the Fixer agent to execute.
"""

import os
import json
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agents.base_agent import BaseAgent
from src.tools.file_tools import FileTools
from src.tools.analysis_tools import AnalysisTools, PylintReport
from src.utils.logger import ActionType
from src.config.settings import get_settings, DEFAULT_PYLINT_THRESHOLD


# Pylint threshold - loaded from centralized settings
# Each file must individually meet this score (not just the average)
try:
    _settings = get_settings()
    PYLINT_PERFECT_THRESHOLD = _settings.analysis.pylint_threshold
except Exception:
    PYLINT_PERFECT_THRESHOLD = DEFAULT_PYLINT_THRESHOLD  # Fallback to centralized default


@dataclass
class CodeIssue:
    """Represents a single code issue identified by the Auditor."""
    file_path: str
    line_number: int
    issue_type: str  # bug, style, documentation, performance, security
    severity: str  # critical, high, medium, low
    description: str
    suggested_fix: str
    pylint_symbol: Optional[str] = None


@dataclass
class RefactoringPlan:
    """Comprehensive refactoring plan for a codebase."""
    files_analyzed: List[str]
    total_issues: int
    issues: List[CodeIssue]
    priority_order: List[str]  # Files in order of fix priority
    initial_scores: Dict[str, float]  # File -> Pylint score
    estimated_complexity: str  # low, medium, high
    summary: str
    fix_type: str = "BUGS"  # NONE, STYLE_ONLY, or BUGS
    files_needing_style: List[str] = field(default_factory=list)  # Files with Pylint < 9.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary."""
        return {
            "files_analyzed": self.files_analyzed,
            "total_issues": self.total_issues,
            "issues": [
                {
                    "file_path": i.file_path,
                    "line_number": i.line_number,
                    "issue_type": i.issue_type,
                    "severity": i.severity,
                    "description": i.description,
                    "suggested_fix": i.suggested_fix,
                    "pylint_symbol": i.pylint_symbol
                }
                for i in self.issues
            ],
            "priority_order": self.priority_order,
            "initial_scores": self.initial_scores,
            "estimated_complexity": self.estimated_complexity,
            "summary": self.summary,
            "fix_type": self.fix_type,
            "files_needing_style": self.files_needing_style
        }
    
    def get_issues_for_file(self, file_path: str) -> List[CodeIssue]:
        """Get all issues for a specific file."""
        return [i for i in self.issues if i.file_path == file_path]
    
    def get_critical_issues(self) -> List[CodeIssue]:
        """Get only critical issues."""
        return [i for i in self.issues if i.severity == "critical"]


class AuditorAgent(BaseAgent):
    """
    The Auditor Agent analyzes Python code and creates refactoring plans.
    
    Responsibilities:
    - Read and understand code structure
    - Run Pylint for static analysis
    - Identify bugs, style issues, missing documentation
    - Create prioritized refactoring plans
    - Provide context for the Fixer agent
    """
    
    def __init__(
        self,
        sandbox_root: str,
        llm=None,
        model_name: str = "gemini-2.5-pro",
        verbose: bool = True
    ):
        """
        Initialize the Auditor Agent.
        
        Args:
            sandbox_root: Root directory containing code to analyze.
            llm: Optional pre-configured LLM instance.
            model_name: Gemini model to use.
            verbose: Enable verbose output.
        """
        super().__init__(
            name="Auditor_Agent",
            llm=llm,
            model_name=model_name,
            verbose=verbose
        )
        
        self.sandbox_root = sandbox_root
        self.file_tools = FileTools(sandbox_root)
        self.analysis_tools = AnalysisTools(sandbox_root)
        
        # Parallel processing settings
        self.MAX_PARALLEL_FILES = 3  # Max files to analyze in parallel
        self._log_lock = threading.Lock()
    
    def _log_safe(self, message: str, level: str = "INFO"):
        """Thread-safe logging method."""
        with self._log_lock:
            self._log(message, level)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Auditor agent."""
        return """You are an expert Python code auditor.

Analyze the code for bugs and issues. You are smart enough to identify real problems - 
crashes, logic errors, type mismatches, unhandled edge cases, etc.

CRITICAL PRINCIPLE:
You cannot know business-specific values like rates, thresholds, formulas, or domain rules.
Don't assume you know what specific numbers should be. If the code calculates something 
and you don't know what the "correct" value is, leave it alone.

Focus on: Does the code do what it APPEARS to be trying to do?
Not: Does the code do what I THINK it should do?

Examples:
- A discount function that ADDS to price instead of subtracting → flag it (logic error)
- A discount function that uses 0.15 instead of 0.10 → don't flag (you don't know the rate)
- Division without checking for zero → flag it (potential crash)
- A retry count of 5 instead of 3 → don't flag (you don't know the requirement)

Also address Pylint issues (style, documentation, type hints) from static analysis.

If the code looks correct and complete, say so. It's OK to find nothing wrong.
Report only issues you are genuinely confident about.

Format your responses as structured JSON when requested."""
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the full audit process.
        
        Returns:
            Dictionary containing the refactoring plan and analysis results.
        """
        self._update_state(status="working", task="Full code audit")
        self._log("Starting comprehensive code audit...")
        
        try:
            # Step 1: Discover Python files
            all_python_files = self.file_tools.list_python_files()
            
            # Filter out test files
            python_files = [
                f for f in all_python_files 
                if not os.path.basename(f).startswith('test_') 
                and not os.path.basename(f).endswith('_test.py')
            ]
            
            if not python_files:
                self._log("No source Python files found (excluding tests)", "WARN")
                return {
                    "success": False,
                    "error": "No Python files found",
                    "plan": None
                }
            
            self._log(f"Found {len(python_files)} Python files to analyze")
            
            # Step 2: Run Pylint static analysis
            pylint_reports = self._run_pylint_analysis(python_files)
            
            # Identify files needing style fixes (per-file threshold)
            files_needing_style = [
                f for f, r in pylint_reports.items()
                if r.score < PYLINT_PERFECT_THRESHOLD
            ]
            
            # Calculate average Pylint score (for logging)
            avg_pylint = sum(r.score for r in pylint_reports.values()) / len(pylint_reports) if pylint_reports else 0.0
            self._log(f"Average Pylint score: {avg_pylint:.2f}/10")
            
            # Step 3: Create refactoring plan using LLM analysis
            # Note: fix_type will be determined by Judge after tests run
            # Auditor always assumes code may need fixes (static analysis only)
            refactoring_plan = self._create_refactoring_plan(
                python_files, pylint_reports,
                files_needing_style=files_needing_style
            )
            
            self._update_state(status="completed", result=refactoring_plan.to_dict())
            self._log(f"Audit completed! Found {refactoring_plan.total_issues} issues", "SUCCESS")
            
            return {
                "success": True,
                "plan": refactoring_plan,
                "files_analyzed": python_files,
                "pylint_reports": {f: r.to_dict() for f, r in pylint_reports.items()}
            }
            
        except Exception as e:
            self._update_state(status="failed", error=str(e))
            self._log(f"Audit failed: {e}", "ERROR")
            return {
                "success": False,
                "error": str(e),
                "plan": None
            }
    
    def _run_pylint_analysis(self, files: List[str]) -> Dict[str, PylintReport]:
        """
        Run Pylint analysis on all files.
        
        Args:
            files: List of file paths to analyze.
            
        Returns:
            Dictionary mapping file paths to PylintReports.
        """
        self._log("Running Pylint static analysis...")
        reports = {}
        
        if len(files) <= 1:
            # Single file - no parallelization needed
            for filepath in files:
                full_path = os.path.join(self.sandbox_root, filepath)
                report = self.analysis_tools.analyze_file(full_path)
                reports[filepath] = report
                
                if self.verbose:
                    status = "✅" if report.score >= 7.0 else "⚠️" if report.score >= 5.0 else "❌"
                    print(f"  {status} {filepath}: {report.score:.2f}/10 ({len(report.messages)} issues)")
        else:
            # PARALLEL PROCESSING for multiple files
            def analyze_single_file(filepath: str) -> Tuple[str, PylintReport]:
                full_path = os.path.join(self.sandbox_root, filepath)
                report = self.analysis_tools.analyze_file(full_path)
                return filepath, report
            
            with ThreadPoolExecutor(max_workers=self.MAX_PARALLEL_FILES) as executor:
                future_to_file = {
                    executor.submit(analyze_single_file, filepath): filepath
                    for filepath in files
                }
                
                for future in as_completed(future_to_file):
                    try:
                        filepath, report = future.result()
                        reports[filepath] = report
                        
                        if self.verbose:
                            status = "✅" if report.score >= 7.0 else "⚠️" if report.score >= 5.0 else "❌"
                            print(f"  {status} {filepath}: {report.score:.2f}/10 ({len(report.messages)} issues)")
                    except Exception as e:
                        filepath = future_to_file[future]
                        self._log_safe(f"  ❌ Error analyzing {filepath}: {e}")
                        # Create a failed report
                        reports[filepath] = PylintReport(
                            score=0.0,
                            messages=[],
                            success=False
                        )
        
        return reports
    
    def _create_refactoring_plan(
        self,
        files: List[str],
        pylint_reports: Dict[str, PylintReport],
        files_needing_style: List[str] = None
    ) -> RefactoringPlan:
        """
        Create a comprehensive refactoring plan using LLM analysis.
        
        Args:
            files: List of Python files.
            pylint_reports: Pylint analysis results.
            files_needing_style: List of files with Pylint < threshold that need style fixes.
            
        Returns:
            RefactoringPlan with prioritized issues and fixes.
        """
        self._log("Creating refactoring plan based on static analysis...")
        
        if files_needing_style is None:
            files_needing_style = []
        
        all_issues = []
        initial_scores = {}
        issues_lock = threading.Lock()
        
        def analyze_file_task(filepath: str) -> None:
            """Analyze a single file (thread-safe task)."""
            # Read file content
            try:
                content = self.file_tools.read_file(filepath)
            except Exception as e:
                self._log_safe(f"Could not read {filepath}: {e}", "WARN")
                return
            
            # Get Pylint report
            report = pylint_reports.get(filepath)
            with issues_lock:
                initial_scores[filepath] = report.score if report else 0.0
            
            # Analyze with LLM (static analysis only, no test context)
            file_issues, api_failed = self._analyze_file_with_llm(
                filepath, content, report
            )
            
            if api_failed:
                self._log_safe("⚠️ LLM API quota exhausted", "WARN")
            else:
                with issues_lock:
                    all_issues.extend(file_issues)
        
        # Process files in parallel
        self._log_safe(f"  🚀 Analyzing {len(files)} files in parallel (max {self.MAX_PARALLEL_FILES} workers)...")
        with ThreadPoolExecutor(max_workers=self.MAX_PARALLEL_FILES) as executor:
            futures = {executor.submit(analyze_file_task, fp): fp for fp in files}
            for future in as_completed(futures):
                filepath = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self._log_safe(f"Error analyzing {filepath}: {e}", "ERROR")
        
        # Sort issues by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_issues.sort(key=lambda x: severity_order.get(x.severity, 4))
        
        # Determine priority order for files
        priority_order = self._determine_priority_order(files, pylint_reports, all_issues)
        
        # Generate summary
        summary = self._generate_summary(files, all_issues, initial_scores)
        
        # Estimate complexity
        complexity = "low" if len(all_issues) < 10 else "medium" if len(all_issues) < 30 else "high"
        
        return RefactoringPlan(
            files_analyzed=files,
            total_issues=len(all_issues),
            issues=all_issues,
            priority_order=priority_order,
            initial_scores=initial_scores,
            estimated_complexity=complexity,
            summary=summary,
            files_needing_style=files_needing_style
        )
    
    def _analyze_file_with_llm(
        self,
        filepath: str,
        content: str,
        pylint_report: Optional[PylintReport]
    ) -> Tuple[List[CodeIssue], bool]:
        """
        Analyze a single file using the LLM for static code review.
        
        Args:
            filepath: Path to the file.
            content: File contents.
            pylint_report: Pylint analysis results.
            
        Returns:
            Tuple of (List of CodeIssue objects, bool indicating if API failed).
        """
        # Prepare Pylint context
        pylint_context = ""
        if pylint_report and pylint_report.messages:
            pylint_issues = []
            for msg in pylint_report.messages[:20]:  # Limit to avoid token overflow
                pylint_issues.append(f"- Line {msg.line}: [{msg.symbol}] {msg.message}")
            pylint_context = "Pylint Issues:\n" + "\n".join(pylint_issues)
        
        # Static analysis prompt - analyze code structure and quality
        prompt = f"""Analyze this Python file and identify issues through static code review.

File: {filepath}

```python
{content}
```

{pylint_context}

Your task is to perform STATIC ANALYSIS only (no test execution).

Analyze the code and identify:
1. CODE STRUCTURE: Document what each function/class is supposed to do based on names and docstrings
2. POTENTIAL BUGS: Logic errors, edge cases, off-by-one errors visible through code review
3. PYLINT ISSUES: Address the warnings and errors from static analysis above
4. STYLE ISSUES: Missing docstrings, PEP 8 violations, missing type hints
5. MISSING FUNCTIONALITY: Empty functions, TODO comments, incomplete implementations

Respond with a JSON array of issues. Each issue must have:
- "line_number": integer (the line where the issue is)
- "issue_type": one of ["bug", "style", "documentation", "performance", "security"]
- "severity": one of ["critical", "high", "medium", "low"]
- "description": string (clear description of the problem)
- "suggested_fix": string (specific fix to apply)
- "pylint_symbol": string or null (if related to a pylint message)

SEVERITY GUIDELINES:
- critical: Obvious bugs (division by zero, infinite loops, syntax-level issues)
- high: Likely bugs (unhandled edge cases, suspicious logic)
- medium: Missing docstrings, incomplete error handling
- low: Style issues, missing type hints, minor improvements

Return ONLY a JSON array, no other text."""

        response = self._call_llm_json(
            prompt=prompt,
            action=ActionType.ANALYSIS,
            additional_details={"file_analyzed": filepath}
        )
        
        issues = []
        api_failed = False
        
        # Check if API call failed (quota exhausted)
        if isinstance(response, dict) and response.get("success") is False:
            api_failed = True
            return issues, api_failed
        
        # Parse response
        if isinstance(response, list):
            issue_list = response
        elif isinstance(response, dict) and "issues" in response:
            issue_list = response["issues"]
        elif isinstance(response, dict) and not response.get("error"):
            issue_list = [response]
        else:
            issue_list = []
        
        for item in issue_list:
            try:
                issue = CodeIssue(
                    file_path=filepath,
                    line_number=item.get("line_number", 1),
                    issue_type=item.get("issue_type", "style"),
                    severity=item.get("severity", "medium"),
                    description=item.get("description", "Unknown issue"),
                    suggested_fix=item.get("suggested_fix", "Review this line"),
                    pylint_symbol=item.get("pylint_symbol")
                )
                issues.append(issue)
            except Exception:
                continue
        
        return issues, api_failed
    
    def _determine_priority_order(
        self,
        files: List[str],
        reports: Dict[str, PylintReport],
        issues: List[CodeIssue]
    ) -> List[str]:
        """
        Determine the order in which files should be fixed.
        
        Args:
            files: List of all files.
            reports: Pylint reports.
            issues: All identified issues.
            
        Returns:
            List of file paths in priority order.
        """
        # Score each file based on:
        # 1. Number of critical/high issues
        # 2. Pylint score (lower = higher priority)
        # 3. Total number of issues
        
        file_scores = {}
        
        for filepath in files:
            score = 0
            file_issues = [i for i in issues if i.file_path == filepath]
            
            # Critical issues add most weight
            score += sum(10 for i in file_issues if i.severity == "critical")
            score += sum(5 for i in file_issues if i.severity == "high")
            score += sum(2 for i in file_issues if i.severity == "medium")
            score += sum(1 for i in file_issues if i.severity == "low")
            
            # Lower pylint score = higher priority
            report = reports.get(filepath)
            if report:
                score += (10 - report.score) * 3
            
            file_scores[filepath] = score
        
        # Sort by score (highest priority first)
        return sorted(files, key=lambda f: file_scores.get(f, 0), reverse=True)
    
    def _generate_summary(
        self,
        files: List[str],
        issues: List[CodeIssue],
        scores: Dict[str, float]
    ) -> str:
        """Generate a human-readable summary of the audit."""
        avg_score = sum(scores.values()) / len(scores) if scores else 0
        
        issue_counts = {
            "critical": sum(1 for i in issues if i.severity == "critical"),
            "high": sum(1 for i in issues if i.severity == "high"),
            "medium": sum(1 for i in issues if i.severity == "medium"),
            "low": sum(1 for i in issues if i.severity == "low")
        }
        
        type_counts = {
            "bug": sum(1 for i in issues if i.issue_type == "bug"),
            "style": sum(1 for i in issues if i.issue_type == "style"),
            "documentation": sum(1 for i in issues if i.issue_type == "documentation"),
            "performance": sum(1 for i in issues if i.issue_type == "performance"),
            "security": sum(1 for i in issues if i.issue_type == "security")
        }
        
        return f"""
AUDIT SUMMARY
=============
Files Analyzed: {len(files)}
Average Pylint Score: {avg_score:.2f}/10
Total Issues Found: {len(issues)}

By Severity:
  🔴 Critical: {issue_counts['critical']}
  🟠 High: {issue_counts['high']}
  🟡 Medium: {issue_counts['medium']}
  🟢 Low: {issue_counts['low']}

By Type:
  🐛 Bugs: {type_counts['bug']}
  📝 Style: {type_counts['style']}
  📚 Documentation: {type_counts['documentation']}
  ⚡ Performance: {type_counts['performance']}
  🔒 Security: {type_counts['security']}
"""
    
    def analyze_single_file(self, filepath: str) -> Dict[str, Any]:
        """
        Analyze a single file (useful for re-analysis after fixes).
        
        Args:
            filepath: Path to the file to analyze.
            
        Returns:
            Analysis results for the single file.
        """
        self._log(f"Analyzing single file: {filepath}")
        
        try:
            content = self.file_tools.read_file(filepath)
            full_path = os.path.join(self.sandbox_root, filepath)
            report = self.analysis_tools.analyze_file(full_path)
            issues, api_failed = self._analyze_file_with_llm(filepath, content, report)
            
            if api_failed:
                return {
                    "success": False,
                    "error": "LLM API quota exhausted",
                    "filepath": filepath
                }
            
            return {
                "success": True,
                "filepath": filepath,
                "pylint_score": report.score,
                "issues": [
                    {
                        "line": i.line_number,
                        "type": i.issue_type,
                        "severity": i.severity,
                        "description": i.description,
                        "fix": i.suggested_fix
                    }
                    for i in issues
                ]
            }
        except Exception as e:
            return {
                "success": False,
                "filepath": filepath,
                "error": str(e)
            }
