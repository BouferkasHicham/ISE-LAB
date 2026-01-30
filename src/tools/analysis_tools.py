"""
Analysis Tools - Code quality analysis using Pylint.

This module provides integration with Pylint for static code analysis,
enabling agents to assess code quality, identify issues, and track
improvements after refactoring.
"""

import subprocess
import json
import re
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class PylintMessage:
    """Represents a single Pylint message/issue."""
    type: str  # C=Convention, R=Refactor, W=Warning, E=Error, F=Fatal
    symbol: str  # e.g., "missing-docstring", "line-too-long"
    message: str  # Human-readable message
    line: int
    column: int
    module: str
    path: str
    
    @property
    def severity(self) -> str:
        """Get human-readable severity level."""
        severity_map = {
            'C': 'Convention',
            'R': 'Refactor',
            'W': 'Warning',
            'E': 'Error',
            'F': 'Fatal'
        }
        return severity_map.get(self.type, 'Unknown')
    
    @property
    def is_error(self) -> bool:
        """Check if this is an error or fatal issue."""
        return self.type in ('E', 'F')


@dataclass
class PylintReport:
    """Comprehensive Pylint analysis report."""
    score: float
    previous_score: Optional[float]
    messages: List[PylintMessage]
    statistics: Dict[str, int]
    raw_output: str
    file_path: str
    success: bool
    error_message: Optional[str] = None
    
    @property
    def errors(self) -> List[PylintMessage]:
        """Get only error-level messages."""
        return [m for m in self.messages if m.is_error]
    
    @property
    def warnings(self) -> List[PylintMessage]:
        """Get only warning-level messages."""
        return [m for m in self.messages if m.type == 'W']
    
    @property
    def conventions(self) -> List[PylintMessage]:
        """Get convention-related messages."""
        return [m for m in self.messages if m.type == 'C']
    
    @property
    def refactors(self) -> List[PylintMessage]:
        """Get refactoring suggestions."""
        return [m for m in self.messages if m.type == 'R']
    
    def get_summary(self) -> str:
        """Get a formatted summary of the report."""
        summary = f"Pylint Score: {self.score:.2f}/10\n"
        summary += f"Total Issues: {len(self.messages)}\n"
        summary += f"  - Errors: {len(self.errors)}\n"
        summary += f"  - Warnings: {len(self.warnings)}\n"
        summary += f"  - Conventions: {len(self.conventions)}\n"
        summary += f"  - Refactoring: {len(self.refactors)}\n"
        
        if self.previous_score is not None:
            delta = self.score - self.previous_score
            direction = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
            summary += f"\nScore Change: {direction} {delta:+.2f}\n"
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for logging."""
        return {
            "score": self.score,
            "previous_score": self.previous_score,
            "total_issues": len(self.messages),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "conventions": len(self.conventions),
            "refactors": len(self.refactors),
            "issues": [
                {
                    "type": m.type,
                    "symbol": m.symbol,
                    "message": m.message,
                    "line": m.line,
                    "column": m.column
                }
                for m in self.messages
            ]
        }


class AnalysisTools:
    """
    Code analysis tools using Pylint.
    
    Provides functionality to analyze Python code quality,
    identify issues, and track improvements across refactoring iterations.
    """
    
    def __init__(self, sandbox_root: str):
        """
        Initialize the analysis tools.
        
        Args:
            sandbox_root: Root directory for code analysis.
        """
        self.sandbox_root = os.path.abspath(sandbox_root)
        self._previous_scores: Dict[str, float] = {}
        # Store the Python executable path for running pylint as a module
        import sys
        self._python_executable = sys.executable
    
    def analyze_file(self, filepath: str) -> PylintReport:
        """
        Analyze a single Python file using Pylint.
        
        Args:
            filepath: Path to the Python file to analyze.
            
        Returns:
            PylintReport with analysis results.
        """
        abs_path = os.path.abspath(filepath)
        
        if not os.path.exists(abs_path):
            return PylintReport(
                score=0.0,
                previous_score=self._previous_scores.get(abs_path),
                messages=[],
                statistics={},
                raw_output="",
                file_path=filepath,
                success=False,
                error_message=f"File not found: {filepath}"
            )
        
        try:
            # Run Pylint as a Python module for maximum compatibility
            # This works regardless of how pylint was installed (pip, pipx, system, etc.)
            # Use json2 format which includes the score in the statistics field
            cmd = [
                self._python_executable,
                "-m",
                "pylint",
                abs_path,
                "--output-format=json2",
                "--score=y",
                "--disable=import-error",  # Avoid false positives from missing imports
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.sandbox_root  # Run from sandbox root for proper imports
            )
            
            # Check if pylint module is not installed
            combined_output = result.stdout + result.stderr
            if "No module named pylint" in combined_output:
                return PylintReport(
                    score=0.0,
                    previous_score=self._previous_scores.get(abs_path),
                    messages=[],
                    statistics={},
                    raw_output=combined_output,
                    file_path=filepath,
                    success=False,
                    error_message="Pylint is not installed. Run: pip install pylint"
                )
            
            # Parse JSON2 output which contains messages and statistics with score
            messages = []
            score = None
            if result.stdout.strip():
                try:
                    json_output = json.loads(result.stdout)
                    # json2 format has 'messages' and 'statistics' fields
                    messages_list = json_output.get('messages', [])
                    for item in messages_list:
                        messages.append(PylintMessage(
                            type=item.get('type', 'U')[0].upper(),
                            symbol=item.get('symbol', 'unknown'),
                            message=item.get('message', ''),
                            line=item.get('line', 0),
                            column=item.get('column', 0),
                            module=item.get('module', ''),
                            path=item.get('path', filepath)
                        ))
                    # Extract score from statistics
                    statistics = json_output.get('statistics', {})
                    score = statistics.get('score')
                except json.JSONDecodeError:
                    pass
            
            # Fallback: try to extract score from text output if json2 parsing failed
            if score is None:
                score = self._extract_score_from_output(result.stderr)
            if score is None:
                score = self._extract_score_from_output(result.stdout)
            
            # Calculate score from exit code if still not found
            # Pylint exit code encodes: bit 0=fatal, 1=error, 2=warning, 3=refactor, 4=convention, 5=usage
            # For a clean file, exit code is 0 and score is 10.0
            if score is None:
                if result.returncode == 0:
                    score = 10.0
                else:
                    # Estimate score based on message counts as last resort
                    error_count = len([m for m in messages if m.type in ('E', 'F')])
                    warning_count = len([m for m in messages if m.type == 'W'])
                    convention_count = len([m for m in messages if m.type in ('C', 'R')])
                    # Rough estimation: errors=-2, warnings=-1, conventions=-0.5
                    estimated_deduction = (error_count * 2) + (warning_count * 1) + (convention_count * 0.5)
                    score = max(0.0, 10.0 - estimated_deduction)
            
            # Build statistics
            statistics = {
                'error': len([m for m in messages if m.type == 'E']),
                'warning': len([m for m in messages if m.type == 'W']),
                'convention': len([m for m in messages if m.type == 'C']),
                'refactor': len([m for m in messages if m.type == 'R']),
                'fatal': len([m for m in messages if m.type == 'F']),
            }
            
            previous_score = self._previous_scores.get(abs_path)
            self._previous_scores[abs_path] = score or 0.0
            
            return PylintReport(
                score=score or 0.0,
                previous_score=previous_score,
                messages=messages,
                statistics=statistics,
                raw_output=result.stdout + result.stderr,
                file_path=filepath,
                success=True
            )
            
        except subprocess.TimeoutExpired:
            return PylintReport(
                score=0.0,
                previous_score=self._previous_scores.get(abs_path),
                messages=[],
                statistics={},
                raw_output="",
                file_path=filepath,
                success=False,
                error_message="Pylint analysis timed out"
            )
        except FileNotFoundError:
            return PylintReport(
                score=0.0,
                previous_score=self._previous_scores.get(abs_path),
                messages=[],
                statistics={},
                raw_output="",
                file_path=filepath,
                success=False,
                error_message=f"Python executable not found: {self._python_executable}"
            )
        except Exception as e:
            error_msg = str(e)
            # Check if pylint module is missing
            if "No module named pylint" in error_msg or "pylint" in error_msg.lower():
                error_msg = "Pylint is not installed. Run: pip install pylint"
            return PylintReport(
                score=0.0,
                previous_score=self._previous_scores.get(abs_path),
                messages=[],
                statistics={},
                raw_output="",
                file_path=filepath,
                success=False,
                error_message=error_msg
            )
    
    def _extract_score_from_output(self, output: str) -> Optional[float]:
        """Extract Pylint score from output text."""
        # Pattern: "Your code has been rated at X.XX/10"
        pattern = r"Your code has been rated at (-?\d+\.?\d*)/10"
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))
        return None
    
    def analyze_directory(self, directory: str = "") -> Dict[str, PylintReport]:
        """
        Analyze all Python files in a directory.
        
        Args:
            directory: Directory to analyze (relative to sandbox_root).
            
        Returns:
            Dictionary mapping file paths to their PylintReports.
        """
        target_dir = os.path.join(self.sandbox_root, directory) if directory else self.sandbox_root
        reports = {}
        
        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, self.sandbox_root)
                    reports[rel_path] = self.analyze_file(filepath)
        
        return reports
    
    def get_aggregate_score(self, reports: Dict[str, PylintReport]) -> float:
        """
        Calculate aggregate score from multiple reports.
        
        Args:
            reports: Dictionary of file paths to PylintReports.
            
        Returns:
            Average Pylint score across all files.
        """
        if not reports:
            return 0.0
        
        valid_scores = [r.score for r in reports.values() if r.success]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    
    def get_all_issues(self, reports: Dict[str, PylintReport]) -> List[Dict[str, Any]]:
        """
        Collect all issues from multiple reports.
        
        Args:
            reports: Dictionary of file paths to PylintReports.
            
        Returns:
            List of all issues with file path information.
        """
        all_issues = []
        for filepath, report in reports.items():
            for msg in report.messages:
                all_issues.append({
                    'file': filepath,
                    'type': msg.type,
                    'severity': msg.severity,
                    'symbol': msg.symbol,
                    'message': msg.message,
                    'line': msg.line,
                    'column': msg.column
                })
        return all_issues
    
    def format_issues_for_prompt(self, reports: Dict[str, PylintReport]) -> str:
        """
        Format issues in a way suitable for LLM prompts.
        
        Args:
            reports: Dictionary of file paths to PylintReports.
            
        Returns:
            Formatted string of all issues.
        """
        output = []
        
        for filepath, report in reports.items():
            if not report.messages:
                continue
                
            output.append(f"\n📄 File: {filepath}")
            output.append(f"   Score: {report.score:.2f}/10")
            output.append("   Issues:")
            
            # Group by severity
            for severity, issues in [
                ("Errors", report.errors),
                ("Warnings", report.warnings),
                ("Conventions", report.conventions),
                ("Refactoring", report.refactors)
            ]:
                if issues:
                    output.append(f"\n   {severity}:")
                    for issue in issues:
                        output.append(
                            f"     Line {issue.line}: [{issue.symbol}] {issue.message}"
                        )
        
        return "\n".join(output) if output else "No issues found."
    
    def check_syntax(self, filepath: str) -> Tuple[bool, str]:
        """
        Check if a Python file has valid syntax.
        
        Args:
            filepath: Path to the Python file.
            
        Returns:
            Tuple of (is_valid, error_message).
        """
        abs_path = os.path.abspath(filepath)
        
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                source = f.read()
            compile(source, abs_path, 'exec')
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)
