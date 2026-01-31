"""
Fixer Agent - Code Refactoring and Bug Fixing.

The Fixer reads the refactoring plan from the Auditor and applies fixes
to the code files, improving quality and fixing bugs.

OPTIMIZATION: Supports three fixing strategies:
1. FULL_FILE - Output complete file (original, for small files)
2. DIFF_BASED - Output unified diff only (80% token reduction)
3. FUNCTION_BY_FUNCTION - Fix one function at a time (for large files)
"""

import os
import re
import ast
import difflib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from src.agents.base_agent import BaseAgent
from src.agents.auditor import RefactoringPlan, CodeIssue
from src.tools.file_tools import FileTools
from src.tools.analysis_tools import AnalysisTools
from src.tools.test_tools import TestTools
from src.utils.logger import ActionType


@dataclass
class FixResult:
    """Result of applying a fix to code."""
    file_path: str
    success: bool
    original_code: str
    fixed_code: str
    issues_addressed: List[str]
    error_message: Optional[str] = None
    pylint_before: float = 0.0
    pylint_after: float = 0.0


@dataclass
class FixerReport:
    """Comprehensive report of all fixes applied."""
    files_fixed: List[str]
    total_fixes: int
    successful_fixes: int
    failed_fixes: int
    fix_results: List[FixResult]
    overall_improvement: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "files_fixed": self.files_fixed,
            "total_fixes": self.total_fixes,
            "successful_fixes": self.successful_fixes,
            "failed_fixes": self.failed_fixes,
            "overall_improvement": self.overall_improvement,
            "results": [
                {
                    "file": r.file_path,
                    "success": r.success,
                    "issues_addressed": r.issues_addressed,
                    "error": r.error_message,
                    "score_before": r.pylint_before,
                    "score_after": r.pylint_after
                }
                for r in self.fix_results
            ]
        }


class FixerAgent(BaseAgent):
    """
    The Fixer Agent applies code fixes based on the refactoring plan.
    
    Responsibilities:
    - Read the refactoring plan from Auditor
    - Apply fixes file by file
    - Handle errors gracefully with backups
    - Track improvements in code quality
    - Support iterative fixing based on Judge feedback
    """
    
    def __init__(
        self,
        sandbox_root: str,
        llm=None,
        model_name: str = "gemini-2.5-pro",
        verbose: bool = True
    ):
        """
        Initialize the Fixer Agent.
        
        Args:
            sandbox_root: Root directory containing code to fix.
            llm: Optional pre-configured LLM instance.
            model_name: Gemini model to use.
            verbose: Enable verbose output.
        """
        super().__init__(
            name="Fixer_Agent",
            llm=llm,
            model_name=model_name,
            verbose=verbose
        )
        
        self.sandbox_root = sandbox_root
        self.file_tools = FileTools(sandbox_root)
        self.analysis_tools = AnalysisTools(sandbox_root)
        self.test_tools = TestTools(sandbox_root)
        
        # Track files that didn't improve (for skipping in later iterations)
        # Key: filepath, Value: number of consecutive failures
        self.stuck_files: Dict[str, int] = {}
        self.STUCK_THRESHOLD = 3  # Skip file after 3 consecutive failures
        self.STYLE_SKIP_THRESHOLD = 8.25  # Skip files at or above this in STYLE_ONLY mode
        
        # Parallel processing settings
        self.MAX_PARALLEL_FILES = 3  # Max files to process in parallel
        
        # Thread-safe lock for logging and shared state
        self._log_lock = threading.Lock()

    def _log_safe(self, message: str, level: str = "INFO"):
        """Thread-safe logging method."""
        with self._log_lock:
            self._log(message, level)

    def execute_cosmetic(self, files: List[str]) -> Dict[str, Any]:
        """
        Execute COSMETIC-ONLY fixes before test generation.
        
        This mode ONLY renames LOCAL VARIABLES inside functions to be more descriptive:
        - Local variables: l → items, O → result, I → index, D → data, R → response
        - Fix wildcard imports: from typing import * → from typing import List, Dict
        - Remove unused imports
        
        PRESERVES (does NOT rename):
        - Function names (keep Process, getData, MyFunc exactly as they are)
        - Method names (keep MyMethod exactly as it is)
        - Class names (keep exactly as they are)
        - Global variables (keep exactly as they are)
        - Function parameters (keep exactly as they are)
        
        This ensures compatibility with external/teacher tests that expect specific API names.
        
        CRITICAL: This does NOT change any logic, operators, return values, or behavior.
        The goal is to get clean LOCAL variable names while preserving the public API.
        
        SAFETY: Files are processed sequentially to avoid issues with file dependencies.
        
        Args:
            files: List of file paths to apply cosmetic fixes to.
            
        Returns:
            Dictionary with results.
        """
        self._update_state(status="working", task="Applying cosmetic fixes (names only)")
        self._log("Starting COSMETIC fixes (renaming only, no logic changes)...")
        
        # SEQUENTIAL PROCESSING - to avoid issues with file dependencies
        self._log(f"  📝 Processing {len(files)} files sequentially (safe mode)...")
        
        results = []
        files_fixed = []
        
        for filepath in files:
            self._log(f"  Cosmetic fix: {os.path.basename(filepath)}...")
            try:
                result = self._apply_cosmetic_fix(filepath)
                results.append(result)
                
                if result.success:
                    files_fixed.append(filepath)
                    self._log(f"    ✅ Renamed: {os.path.basename(filepath)} ({result.pylint_before:.2f} → {result.pylint_after:.2f})", "SUCCESS")
                else:
                    self._log(f"    ⚠️ Skipped: {result.error_message}", "WARNING")
            except Exception as e:
                self._log(f"    ❌ Error on {os.path.basename(filepath)}: {e}", "ERROR")
                results.append(FixResult(
                    file_path=filepath,
                    success=False,
                    original_code="",
                    fixed_code="",
                    issues_addressed=[],
                    error_message=str(e)
                ))
        
        return {
            "success": True,
            "files_fixed": files_fixed,
            "results": results
        }
    
    def _apply_cosmetic_fix_safe(self, filepath: str) -> FixResult:
        """
        Thread-safe wrapper for _apply_cosmetic_fix.
        
        Each file has its own backup, so parallel processing is safe.
        """
        try:
            return self._apply_cosmetic_fix(filepath)
        except Exception as e:
            return FixResult(
                file_path=filepath,
                success=False,
                original_code="",
                fixed_code="",
                issues_addressed=[],
                error_message=str(e)
            )
    
    def _apply_cosmetic_fix(self, filepath: str) -> FixResult:
        """
        Apply cosmetic-only fixes to a single file with retry logic.
        
        Validates that the LLM didn't rename any functions, methods, classes,
        or global variables. If it did, reverts and retries up to 3 times.
        
        Args:
            filepath: Path to the file.
            
        Returns:
            FixResult with the outcome.
        """
        MAX_COSMETIC_RETRIES = 3
        
        try:
            original_code = self.file_tools.read_file(filepath)
            self.file_tools.backup_file(filepath)
            
            full_path = os.path.join(self.sandbox_root, filepath)
            initial_report = self.analysis_tools.analyze_file(full_path)
            pylint_before = initial_report.score if initial_report.success else 0.0
            
            # Extract original definitions to validate against
            orig_funcs, orig_classes = self._extract_definitions(original_code)
            
            last_error = ""
            
            for attempt in range(1, MAX_COSMETIC_RETRIES + 1):
                # Generate cosmetic-only fix using LLM
                fixed_code, llm_succeeded = self._generate_cosmetic_fix(filepath, original_code)
                
                if not llm_succeeded:
                    last_error = "LLM call failed"
                    continue
                
                # Normalize line endings
                fixed_code = fixed_code.replace('\r\n', '\n').replace('\r', '\n')
                if not fixed_code.endswith('\n'):
                    fixed_code += '\n'
                
                # Validate that no definitions were renamed or deleted
                is_valid, validation_error = self._validate_cosmetic_preserves_api(
                    original_code, fixed_code, orig_funcs, orig_classes
                )
                
                if not is_valid:
                    last_error = validation_error
                    if attempt < MAX_COSMETIC_RETRIES:
                        self._log(f"    ⚠️ Cosmetic fix renamed API (attempt {attempt}/{MAX_COSMETIC_RETRIES}): {validation_error}", "WARNING")
                    continue
                
                # Validate syntax
                self.file_tools.write_file(filepath, fixed_code)
                is_valid_syntax, syntax_error = self.analysis_tools.check_syntax(full_path)
                
                if not is_valid_syntax:
                    self.file_tools.restore_backup(filepath)
                    last_error = f"Syntax error: {syntax_error}"
                    continue
                
                # Success! Get new Pylint score
                final_report = self.analysis_tools.analyze_file(full_path)
                pylint_after = final_report.score if final_report.success else 0.0
                
                return FixResult(
                    file_path=filepath,
                    success=True,
                    original_code=original_code,
                    fixed_code=fixed_code,
                    issues_addressed=["Renamed local variables for clarity"],
                    pylint_before=pylint_before,
                    pylint_after=pylint_after
                )
            
            # All retries failed - restore original and return failure
            self.file_tools.restore_backup(filepath)
            self._log(f"    ⚠️ Cosmetic fix failed after {MAX_COSMETIC_RETRIES} attempts, keeping original", "WARNING")
            
            return FixResult(
                file_path=filepath,
                success=False,
                original_code=original_code,
                fixed_code=original_code,
                issues_addressed=[],
                error_message=f"All {MAX_COSMETIC_RETRIES} attempts failed: {last_error}",
                pylint_before=pylint_before,
                pylint_after=pylint_before
            )
            
        except Exception as e:
            try:
                self.file_tools.restore_backup(filepath)
            except Exception:
                pass
            
            return FixResult(
                file_path=filepath,
                success=False,
                original_code="",
                fixed_code="",
                issues_addressed=[],
                error_message=str(e)
            )
    
    def _validate_cosmetic_preserves_api(
        self,
        original_code: str,
        fixed_code: str,
        orig_funcs: set,
        orig_classes: set
    ) -> Tuple[bool, str]:
        """
        Validate that cosmetic fix preserved all function/class/method names.
        
        Args:
            original_code: Original source code.
            fixed_code: Fixed source code.
            orig_funcs: Set of original function names.
            orig_classes: Set of original class names.
            
        Returns:
            Tuple of (is_valid, error_message).
        """
        fixed_funcs, fixed_classes = self._extract_definitions(fixed_code)
        
        # Check for renamed functions (missing from fixed)
        missing_funcs = orig_funcs - fixed_funcs
        # Check for renamed classes (missing from fixed)
        missing_classes = orig_classes - fixed_classes
        # Check for new functions that weren't in original (likely renames)
        new_funcs = fixed_funcs - orig_funcs
        # Check for new classes that weren't in original (likely renames)
        new_classes = fixed_classes - orig_classes
        
        errors = []
        
        if missing_funcs:
            if new_funcs:
                # Likely renamed
                errors.append(f"Functions renamed: {', '.join(sorted(missing_funcs))} → possibly {', '.join(sorted(new_funcs))}")
            else:
                errors.append(f"Functions deleted: {', '.join(sorted(missing_funcs))}")
        
        if missing_classes:
            if new_classes:
                errors.append(f"Classes renamed: {', '.join(sorted(missing_classes))} → possibly {', '.join(sorted(new_classes))}")
            else:
                errors.append(f"Classes deleted: {', '.join(sorted(missing_classes))}")
        
        if errors:
            return False, "; ".join(errors)
        
        return True, ""
    
    def _generate_cosmetic_fix(self, filepath: str, original_code: str) -> Tuple[str, bool]:
        """
        Generate cosmetic-only fix using LLM.
        
        ONLY renames LOCAL variables and fixes imports, does NOT change any logic.
        Preserves all function names, method names, class names, and global variables
        to maintain compatibility with external tests.
        
        Args:
            filepath: Path to the file.
            original_code: Original code content.
            
        Returns:
            Tuple of (fixed code, success).
        """
        prompt = f"""You are a Python code INTERNAL RENAMER. Your ONLY job is to rename LOCAL variables inside functions.

FILE: {filepath}

ORIGINAL CODE:
```python
{original_code}
```

## YOUR TASK: RENAME LOCAL VARIABLES ONLY

### MUST RENAME (if present):
1. **Local variables inside functions** - fix ambiguous single letters: `l` → `items`, `O` → `result`, `I` → `index`, `D` → `data`, `R` → `response`
2. **Wildcard imports** - replace with explicit: `from typing import *` → `from typing import List, Dict` (only what's used)
3. **Unused imports** - remove them

### ⚠️ ABSOLUTELY DO NOT RENAME (PRESERVE EXACTLY):
- ❌ **Function names** - keep `Process`, `getData`, `MyFunc` exactly as they are
- ❌ **Method names** - keep `MyMethod`, `DoSomething` exactly as they are  
- ❌ **Class names** - keep exactly as they are
- ❌ **Global variables** - keep exactly as they are
- ❌ **Function parameters** - keep exactly as they are (even single letters like `a`, `b`)
- ❌ **Any public API** - the function signatures must stay the same

### ABSOLUTELY DO NOT CHANGE:
- ❌ ANY operators (+, -, *, /, ==, !=, <, >, etc.)
- ❌ ANY return values or return statements
- ❌ ANY if/else conditions or logic
- ❌ ANY loop logic or bounds
- ❌ ANY function behavior or algorithms
- ❌ ANY arithmetic or comparisons
- ❌ Number of functions, classes, or their structure
- ❌ Function signatures or parameter names

### EXAMPLE:
BEFORE:
```python
def Process(A, B):
    l = A + B  # BUG: wrong
    D = l * 2
    return D
```

AFTER (ONLY local variables `l` and `D` renamed, function name and params PRESERVED):
```python
def Process(A, B):
    result = A + B  # BUG: wrong
    data = result * 2
    return data
```

Notice: Function name `Process` and parameters `A`, `B` are PRESERVED. Only local variables renamed.

## OUTPUT
Return the COMPLETE Python file with ONLY local variable renames applied.
Every function, class, method that exists in the original MUST have the SAME NAME in your output.
No markdown code blocks. Start directly with Python code."""

        response = self._call_llm(
            prompt=prompt,
            action=ActionType.FIX,
            additional_details={
                "file_fixed": filepath,
                "fix_type": "COSMETIC"
            }
        )
        
        if not response.success:
            return original_code, False
        
        fixed_code = self._extract_code(response.content)
        return fixed_code, True

    def _extract_definitions(self, code: str) -> Tuple[set, set]:
        """
        Extract function and class names from Python code.
        
        Args:
            code: Python source code.
            
        Returns:
            Tuple of (function_names, class_names) as sets.
        """
        import ast
        try:
            tree = ast.parse(code)
            functions = set()
            classes = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    # Only top-level and class-level functions
                    functions.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.add(node.name)
            return functions, classes
        except SyntaxError:
            # If we can't parse, use regex fallback
            func_pattern = r'^\s*(?:async\s+)?def\s+(\w+)\s*\('
            class_pattern = r'^\s*class\s+(\w+)[:\(]'
            functions = set(re.findall(func_pattern, code, re.MULTILINE))
            classes = set(re.findall(class_pattern, code, re.MULTILINE))
            return functions, classes

    def _validate_no_deleted_definitions(
        self,
        original_code: str,
        fixed_code: str
    ) -> Tuple[bool, str]:
        """
        Check that fixed code hasn't deleted any functions or classes.
        
        Args:
            original_code: Original source code.
            fixed_code: Fixed source code.
            
        Returns:
            Tuple of (is_valid, error_message).
        """
        orig_funcs, orig_classes = self._extract_definitions(original_code)
        fixed_funcs, fixed_classes = self._extract_definitions(fixed_code)
        
        missing_funcs = orig_funcs - fixed_funcs
        missing_classes = orig_classes - fixed_classes
        
        if missing_funcs or missing_classes:
            errors = []
            if missing_funcs:
                errors.append(f"Missing functions: {', '.join(sorted(missing_funcs))}")
            if missing_classes:
                errors.append(f"Missing classes: {', '.join(sorted(missing_classes))}")
            return False, "; ".join(errors)
        
        return True, ""

    # Token threshold for using smart context extraction vs full file
    # Average token is ~4 chars, so 2000 tokens ≈ 8000 chars
    LARGE_FILE_CHAR_THRESHOLD = 8000
    
    def _extract_relevant_context(
        self,
        code: str,
        issues: List[CodeIssue],
        context_lines: int = 10
    ) -> Tuple[str, bool]:
        """
        Extract only the relevant portions of code around issues.
        
        For large files, this significantly reduces token usage by sending
        only the code sections that need fixing, with surrounding context.
        
        Args:
            code: Full source code.
            issues: List of issues with line numbers.
            context_lines: Number of context lines before/after each issue.
            
        Returns:
            Tuple of (extracted_code_with_markers, is_partial).
            is_partial is True if we extracted partial context, False if full file.
        """
        # For small files, just return the full content
        if len(code) <= self.LARGE_FILE_CHAR_THRESHOLD:
            return code, False
        
        lines = code.split('\n')
        total_lines = len(lines)
        
        # Collect line ranges that need to be included
        ranges_to_include = set()
        
        for issue in issues:
            line_num = issue.line_number
            # Ensure line number is valid
            if line_num < 1:
                line_num = 1
            if line_num > total_lines:
                line_num = total_lines
            
            # Add range around the issue (0-indexed internally)
            start = max(0, line_num - 1 - context_lines)
            end = min(total_lines, line_num + context_lines)
            
            for i in range(start, end):
                ranges_to_include.add(i)
        
        # If we'd include most of the file anyway, just send the full file
        if len(ranges_to_include) > total_lines * 0.7:
            return code, False
        
        # Always include imports and module docstring (first 20 lines or until first def/class)
        header_end = 0
        for i, line in enumerate(lines[:30]):
            stripped = line.strip()
            if stripped.startswith(('def ', 'class ', 'async def ')):
                header_end = i
                break
            header_end = i + 1
        
        for i in range(min(header_end, 20)):
            ranges_to_include.add(i)
        
        # Build the extracted code with markers
        sorted_ranges = sorted(ranges_to_include)
        result_lines = []
        last_included = -2
        
        for i in sorted_ranges:
            # Add ellipsis marker if there's a gap
            if i > last_included + 1:
                if result_lines:  # Don't add at the very beginning
                    gap_size = i - last_included - 1
                    result_lines.append(f"\n# ... [{gap_size} lines omitted] ...\n")
            
            result_lines.append(f"{lines[i]}  # L{i + 1}")
            last_included = i
        
        # Add marker if we're not at the end
        if last_included < total_lines - 1:
            remaining = total_lines - last_included - 1
            result_lines.append(f"\n# ... [{remaining} lines until end of file] ...")
        
        extracted = '\n'.join(result_lines)
        return extracted, True

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Fixer agent."""
        return """You are an expert Python developer specializing in bug fixing and code refactoring.

Your role is to:
1. FIX BUGS - Implement correct, general-purpose algorithms (HIGHEST PRIORITY)
2. Fix runtime errors and exceptions
3. Improve code quality following PEP 8 and best practices
4. Add missing docstrings and type hints
5. Remove unused imports and variables

CRITICAL - GENERAL CORRECTNESS, NOT OVERFITTING:
- Write GENERAL algorithms that work for ANY valid input, not just test values
- Example: If tests show add(2,3)==5, implement "return a + b", NOT "return 5"
- Example: If tests show find_max([1,5,3])==5, implement a real max-finding loop
- The tests are examples of correct behavior, not the only cases to handle

When fixing code:
- Understand the INTENT from function names, docstrings, and test patterns
- Implement standard algorithms (sorting, searching, arithmetic, etc.)
- Handle edge cases properly (empty lists, zero values, negative numbers)
- Make minimal changes - don't rewrite everything unnecessarily

CRITICAL RULES:
- DO NOT rename any public functions, classes, or methods
- DO NOT change function signatures (parameter names, order, or count)
- DO NOT DELETE any functions, classes, or global variables
- DO NOT change import style (absolute to relative or vice versa)
  - If code uses "from module import X", keep it as absolute import
  - If code uses "from .module import X", keep it as relative import
  - Changing import style can break the code if the package structure is wrong
- Keep all public APIs intact

IMPORTANT: Return ONLY the complete fixed Python code.
Do NOT include explanations, markdown code blocks, or any other text.
Return ONLY valid Python code that can be directly written to a file."""
    
    def execute(
        self,
        plan: RefactoringPlan = None,
        fix_type: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the fixing process based on the refactoring plan.
        
        SAFETY: Files are processed sequentially to avoid issues with file dependencies.
        
        Args:
            plan: RefactoringPlan from the Auditor agent.
            fix_type: Override fix_type ("STYLE_ONLY" or "BUGS"). If None, uses plan.fix_type.
            
        Returns:
            Dictionary containing the fixer report and results.
        """
        if plan is None:
            return {
                "success": False,
                "error": "No refactoring plan provided",
                "report": None
            }
        
        # Determine fix mode
        mode = fix_type or getattr(plan, 'fix_type', 'BUGS')
        
        # Get files that need style fixes (only relevant in STYLE_ONLY mode)
        files_needing_style = getattr(plan, 'files_needing_style', [])
        
        self._update_state(status="working", task=f"Applying fixes (mode: {mode})")
        self._log(f"Starting code fixing process (mode: {mode})...")
        self._log(f"Plan has {plan.total_issues} issues across {len(plan.files_analyzed)} files")
        
        if mode == "STYLE_ONLY" and files_needing_style:
            self._log(f"📝 Files needing style fixes ({len(files_needing_style)}): {files_needing_style}")
        
        try:
            fix_results = []
            files_fixed = []
            
            # Collect files to process (filter out skipped files)
            files_to_fix = []
            for filepath in plan.priority_order:
                file_issues = plan.get_issues_for_file(filepath)
                
                if not file_issues:
                    continue
                
                # In STYLE_ONLY mode, skip files that already meet threshold
                if mode == "STYLE_ONLY" and filepath not in files_needing_style:
                    self._log(f"⏭️ Skipping {filepath} (already meets threshold, Pylint >= {self.STYLE_SKIP_THRESHOLD})")
                    continue
                
                # Skip files that are stuck (didn't improve in several iterations)
                if filepath in self.stuck_files and self.stuck_files[filepath] >= self.STUCK_THRESHOLD:
                    self._log(f"⏭️ Skipping {filepath} (stuck - no improvement in {self.STUCK_THRESHOLD} iterations)")
                    continue
                
                files_to_fix.append((filepath, file_issues))
            
            if not files_to_fix:
                self._log("No files to fix after filtering")
                report = FixerReport(
                    files_fixed=[],
                    total_fixes=0,
                    successful_fixes=0,
                    failed_fixes=0,
                    fix_results=[],
                    overall_improvement=0.0
                )
                return {
                    "success": True,
                    "report": report,
                    "files_fixed": [],
                    "improvement": 0.0,
                    "fix_type": mode
                }
            
            # SEQUENTIAL PROCESSING - to avoid issues with file dependencies
            self._log(f"  📝 Processing {len(files_to_fix)} files sequentially (safe mode)...")
            
            for filepath, file_issues in files_to_fix:
                self._log(f"  Fixing {os.path.basename(filepath)} ({len(file_issues)} issues) [mode: {mode}]...")
                try:
                    result = self._fix_file(filepath, file_issues, fix_type=mode)
                    fix_results.append(result)
                    
                    if result.success:
                        files_fixed.append(filepath)
                        self._log(f"    ✅ {os.path.basename(filepath)}: {result.pylint_before:.2f} → {result.pylint_after:.2f}", "SUCCESS")
                    else:
                        self._log(f"    ❌ {os.path.basename(filepath)}: {result.error_message}", "ERROR")
                except Exception as e:
                    self._log(f"    ❌ {os.path.basename(filepath)}: Exception - {e}", "ERROR")
                    fix_results.append(FixResult(
                        file_path=filepath,
                        success=False,
                        original_code="",
                        fixed_code="",
                        issues_addressed=[],
                        error_message=str(e)
                    ))
            
            # Calculate overall improvement
            total_before = sum(r.pylint_before for r in fix_results)
            total_after = sum(r.pylint_after for r in fix_results)
            improvement = (total_after - total_before) / len(fix_results) if fix_results else 0
            
            report = FixerReport(
                files_fixed=files_fixed,
                total_fixes=len(fix_results),
                successful_fixes=sum(1 for r in fix_results if r.success),
                failed_fixes=sum(1 for r in fix_results if not r.success),
                fix_results=fix_results,
                overall_improvement=improvement
            )
            
            self._update_state(status="completed", result=report.to_dict())
            self._log(f"Fixing completed: {report.successful_fixes}/{report.total_fixes} files fixed", "SUCCESS")
            
            return {
                "success": True,
                "report": report,
                "files_fixed": files_fixed,
                "improvement": improvement,
                "fix_type": mode
            }
            
        except Exception as e:
            self._update_state(status="failed", error=str(e))
            self._log(f"Fixing failed: {e}", "ERROR")
            return {
                "success": False,
                "error": str(e),
                "report": None
            }
    
    def _fix_file_safe(
        self, filepath: str, issues: List[CodeIssue], fix_type: str = "BUGS"
    ) -> FixResult:
        """
        Thread-safe wrapper for _fix_file.
        Catches exceptions and returns a FixResult even on failure.
        """
        try:
            return self._fix_file(filepath, issues, fix_type)
        except Exception as e:
            self._log_safe(f"  ⚠️ Exception fixing {filepath}: {e}")
            return FixResult(
                filepath=filepath,
                success=False,
                pylint_before=0.0,
                pylint_after=0.0,
                issues_fixed=0,
                issues_remaining=len(issues),
                error_message=str(e)
            )

    
    def _fix_file(
        self, filepath: str, issues: List[CodeIssue], fix_type: str = "BUGS"
    ) -> FixResult:
        """
        Fix a single file based on its issues.
        
        Args:
            filepath: Path to the file.
            issues: List of issues to fix.
            fix_type: Type of fixes - "STYLE_ONLY" or "BUGS".
            
        Returns:
            FixResult with the outcome.
        """
        try:
            # Read original content
            original_code = self.file_tools.read_file(filepath)
            
            # Create backup
            self.file_tools.backup_file(filepath)
            
            # Get initial Pylint score
            full_path = os.path.join(self.sandbox_root, filepath)
            initial_report = self.analysis_tools.analyze_file(full_path)
            pylint_before = initial_report.score if initial_report.success else 0.0
            
            # Try to find associated test file
            # In BUGS mode: tests show what the code SHOULD do
            # In STYLE_ONLY mode: tests show what behavior MUST be preserved
            test_code = None
            try:
                dirname = os.path.dirname(filepath)
                basename = os.path.basename(filepath)
                test_filename = f"test_{basename}"
                test_filepath = os.path.join(dirname, test_filename)
                
                if test_filename in self.file_tools.list_files(dirname):
                    test_code = self.file_tools.read_file(test_filepath)
                    self._log(f"  ℹ️ Found associated test file: {test_filepath}")
            except Exception:
                pass

            # Retry loop for syntax errors
            max_retries = 3
            current_code = original_code
            
            llm_succeeded = False
            for attempt in range(max_retries):
                 # Generate fixed code using LLM
                if attempt == 0:
                    fixed_code, llm_succeeded = self._generate_fixed_code(
                        filepath, current_code, issues, test_code, fix_type
                    )
                else:
                    self._log(f"  🔄 Retry {attempt}/{max_retries} due to syntax error...")
                    fixed_code, llm_succeeded = self._fix_syntax_error(filepath, fixed_code, error_msg)
                
                # If LLM call failed, don't bother with syntax check - abort early
                if not llm_succeeded:
                    self.file_tools.restore_backup(filepath)
                    return FixResult(
                        file_path=filepath,
                        success=False,
                        original_code=original_code,
                        fixed_code=original_code,
                        issues_addressed=[],
                        error_message="LLM call failed - no fix applied",
                        pylint_before=pylint_before,
                        pylint_after=pylint_before
                    )

                # Normalize line endings (convert CRLF to LF) and ensure final newline
                fixed_code = fixed_code.replace('\r\n', '\n').replace('\r', '\n')
                if not fixed_code.endswith('\n'):
                    fixed_code += '\n'

                # Write and check syntax
                self.file_tools.write_file(filepath, fixed_code)
                is_valid, error_msg = self.analysis_tools.check_syntax(full_path)
                
                if is_valid:
                    # Check that no functions/classes were deleted
                    defs_valid, defs_error = self._validate_no_deleted_definitions(
                        original_code, fixed_code
                    )
                    if not defs_valid:
                        self._log(f"  ⚠️ LLM deleted definitions: {defs_error}", "WARN")
                        # Treat as an error and retry
                        error_msg = f"Code truncated - {defs_error}"
                        is_valid = False
                    else:
                        # Check if code actually changed
                        if fixed_code.strip() == original_code.strip():
                            # If no change, report as failure
                            self._log(f"  ⚠️ LLM returned unchanged code - no fix applied", "WARN")
                            self.file_tools.restore_backup(filepath)
                            return FixResult(
                                file_path=filepath,
                                success=False,
                                original_code=original_code,
                                fixed_code=original_code,
                                issues_addressed=[],
                                error_message="LLM returned unchanged code - no fix applied",
                                pylint_before=pylint_before,
                                pylint_after=pylint_before
                            )
                        # If valid and complete, proceed to Pylint check
                        break
                
                # If invalid, and this was the last attempt, restore backup
                if attempt == max_retries - 1:
                    self.file_tools.restore_backup(filepath)
                    return FixResult(
                        file_path=filepath,
                        success=False,
                        original_code=original_code,
                        fixed_code=fixed_code,
                        issues_addressed=[],
                        error_message=f"Fixed code has syntax error after retries: {error_msg}",
                        pylint_before=pylint_before,
                        pylint_after=pylint_before
                    )
                # Otherwise loop continues to retry based on error_msg
            
            # Get new Pylint score
            final_report = self.analysis_tools.analyze_file(full_path)
            pylint_after = final_report.score if final_report.success else 0.0
            
            # CRITICAL SAFETY CHECK for STYLE_ONLY mode:
            # If we're in STYLE_ONLY mode, tests MUST still pass after changes.
            # If tests fail, we accidentally broke logic - ROLLBACK immediately!
            if fix_type == "STYLE_ONLY":
                test_results = self.test_tools.run_tests()
                if not test_results.all_passed:
                    self._log(f"  ⚠️ STYLE_ONLY change broke tests! Rolling back...", "WARN")
                    self.file_tools.restore_backup(filepath)
                    self.stuck_files[filepath] = self.stuck_files.get(filepath, 0) + 1
                    return FixResult(
                        file_path=filepath,
                        success=False,
                        original_code=original_code,
                        fixed_code=fixed_code,
                        issues_addressed=[],
                        error_message="STYLE_ONLY changes broke tests - rolled back to preserve functionality",
                        pylint_before=pylint_before,
                        pylint_after=pylint_before
                    )
            
            # ROLLBACK if score dropped - but ONLY in STYLE_ONLY mode!
            # In BUGS mode, we prioritize fixing bugs over maintaining Pylint score
            if pylint_after < pylint_before and fix_type == "STYLE_ONLY":
                self._log(f"  ⚠️ Score dropped ({pylint_before:.2f} → {pylint_after:.2f}), rolling back", "WARN")
                self.file_tools.restore_backup(filepath)
                # Mark as stuck since this fix made it worse
                self.stuck_files[filepath] = self.stuck_files.get(filepath, 0) + 1
                return FixResult(
                    file_path=filepath,
                    success=False,
                    original_code=original_code,
                    fixed_code=fixed_code,
                    issues_addressed=[],
                    error_message=f"Score dropped from {pylint_before:.2f} to {pylint_after:.2f}, rolled back",
                    pylint_before=pylint_before,
                    pylint_after=pylint_before  # Report original score since we rolled back
                )
            
            # In BUGS mode, log if score dropped but don't roll back
            if pylint_after < pylint_before and fix_type == "BUGS":
                self._log(f"  ℹ️ Pylint dropped ({pylint_before:.2f} → {pylint_after:.2f}) but keeping bug fix")
            
            # Track stuck files (no improvement in STYLE_ONLY mode)
            if pylint_after == pylint_before and fix_type == "STYLE_ONLY":
                self.stuck_files[filepath] = self.stuck_files.get(filepath, 0) + 1
            elif pylint_after > pylint_before:
                # Clear stuck counter on improvement
                self.stuck_files.pop(filepath, None)
            
            return FixResult(
                file_path=filepath,
                success=True,
                original_code=original_code,
                fixed_code=fixed_code,
                issues_addressed=[i.description for i in issues],
                pylint_before=pylint_before,
                pylint_after=pylint_after
            )
            
        except Exception as e:
            # Restore backup on error
            try:
                self.file_tools.restore_backup(filepath)
            except Exception as restore_error:
                self._log(f"  ⚠️ Failed to restore backup for {filepath}: {restore_error}", "WARN")
            
            return FixResult(
                file_path=filepath,
                success=False,
                original_code="",
                fixed_code="",
                issues_addressed=[],
                error_message=str(e),
                pylint_before=0.0,
                pylint_after=0.0
            )
    
    def _generate_fixed_code(
        self,
        filepath: str,
        original_code: str,
        issues: List[CodeIssue],
        test_code: Optional[str] = None,
        fix_type: str = "BUGS"
    ) -> Tuple[str, bool]:
        """
        Generate fixed code using FULL_FILE mode (always).
        
        Args:
            filepath: Path to the file.
            original_code: Original code content.
            issues: List of issues to fix.
            test_code: Optional test file content.
            fix_type: Type of fixes - "STYLE_ONLY" or "BUGS".
            
        Returns:
            Tuple of (fixed code string, success boolean).
        """
        line_count = len(original_code.split('\n'))
        self._log(f"  📄 Using FULL_FILE mode ({line_count} lines)")
        return self._generate_full_file_fix(filepath, original_code, issues, test_code, fix_type)
    
    def _generate_full_file_fix(
        self,
        filepath: str,
        original_code: str,
        issues: List[CodeIssue],
        test_code: Optional[str] = None,
        fix_type: str = "BUGS"
    ) -> Tuple[str, bool]:
        """
        Generate fixed code by asking LLM to return the complete file.
        
        For large files, uses smart context extraction to reduce token usage
        while still requiring the LLM to output the complete fixed file.
        
        Args:
            filepath: Path to the file.
            original_code: Original code content.
            issues: List of issues to fix.
            test_code: Optional test file content.
            fix_type: Type of fixes - "STYLE_ONLY" or "BUGS".
            
        Returns:
            Tuple of (fixed code string, success boolean).
        """
        # Format issues for the prompt
        issues_text = "\n".join([
            f"- Line {i.line_number} [{i.severity}]: {i.description}\n  Fix: {i.suggested_fix}"
            for i in issues
        ])
        
        # Smart context extraction for large files to reduce token usage
        code_for_prompt, is_partial = self._extract_relevant_context(original_code, issues)
        
        # Additional instructions if using partial context
        partial_context_note = ""
        if is_partial:
            partial_context_note = """
⚠️ PARTIAL CODE SHOWN: Only relevant sections are shown with line numbers (# L123).
Lines marked with "# ... [N lines omitted] ..." indicate skipped code.
You MUST still output the COMPLETE file - reconstruct omitted sections unchanged."""
        
        if fix_type == "STYLE_ONLY":
            # Prepare test context for STYLE_ONLY mode
            test_section = ""
            if test_code:
                # Also apply smart extraction to test code for large test files
                test_for_prompt = test_code
                if len(test_code) > self.LARGE_FILE_CHAR_THRESHOLD:
                    test_for_prompt = test_code[:self.LARGE_FILE_CHAR_THRESHOLD] + "\n# ... [test file truncated for brevity] ..."
                test_section = f"""
REFERENCE - EXISTING TESTS (READ-ONLY CONTEXT):
The code below ALREADY PASSES all these tests. This is just for your reference
to understand what each function does. DO NOT modify any logic.
```python
{test_for_prompt}
```
"""
            
            # STYLE-ONLY MODE: Only improve style, do NOT change logic
            prompt = f"""Improve ONLY the STYLE and DOCUMENTATION of this Python code.

FILE: {filepath}
{partial_context_note}
ISSUES TO FIX (STYLE ONLY):
{issues_text}

ORIGINAL CODE:
```python
{code_for_prompt}
```
{test_section}
⚠️ CRITICAL - THIS CODE IS ALREADY CORRECT ⚠️
The code above is FUNCTIONALLY PERFECT. All tests pass. Your job is ONLY cosmetic.

ABSOLUTE RULES - ZERO TOLERANCE:
1. PRESERVE ALL LOGIC EXACTLY - every operator, comparison, return value must stay the same
2. DO NOT "fix" anything that looks like a bug - it's intentional and correct
3. DO NOT change any: if/else conditions, loop bounds, arithmetic, return statements
4. DO NOT reorder statements or change control flow
5. DO NOT add error handling or edge cases - the code handles everything correctly

✅ YOU MUST DO THESE STYLE FIXES:
- REPLACE wildcard imports: `from typing import *` → `from typing import List, Dict, Optional` (only what's used)
- RENAME ambiguous variable names: l → items, O → result, I → index (Pylint E741)
- Add a module docstring at the very top
- Add docstrings to functions/classes (describe what they DO, not what they SHOULD do)
- Add type hints (: int, -> str, etc.)
- Fix whitespace around operators (a=b → a = b)
- Remove completely unused imports
- Fix multiple statements on one line (x=1;y=2 → separate lines)
- Replace bare except with specific exceptions

⛔ DO NOT CHANGE LOGIC ⛔
- Keep the SAME algorithm - just rename variables and add documentation
- If renaming a variable, rename it EVERYWHERE consistently
- reverseString stays reverseString (keep camelCase if original uses it)

## OUTPUT FORMAT
Return the COMPLETE Python file. Include every function, class, and import.
Do not truncate or use placeholders like "..." or "# rest of code".
No markdown code blocks. Start directly with Python code."""
        else:
            # BUGS MODE: Fix logic issues and bugs
            # Prepare test code section if available
            test_section = ""
            if test_code:
                # Also apply smart extraction to test code for large test files
                test_for_prompt = test_code
                if len(test_code) > self.LARGE_FILE_CHAR_THRESHOLD:
                    test_for_prompt = test_code[:self.LARGE_FILE_CHAR_THRESHOLD] + "\n# ... [test file truncated for brevity] ..."
                test_section = f"""
ASSOCIATED TESTS (USE AS SPECIFICATION):
The tests below show expected behavior. Use them to understand what the code SHOULD do.
```python
{test_for_prompt}
```
"""

            prompt = f"""Fix the following Python code according to the issues listed below.

FILE: {filepath}
{partial_context_note}
ISSUES TO FIX:
{issues_text}

ORIGINAL CODE:
```python
{code_for_prompt}
```
{test_section}
REQUIREMENTS (IN PRIORITY ORDER):
1. IMPLEMENT CORRECT GENERAL LOGIC - Write code that works for ALL inputs, not just test cases
2. Understand the INTENT of each function from its name, docstring, and tests, then implement correctly
3. Use standard algorithms (e.g., proper max/min finding, correct arithmetic operations)
4. Fix ALL listed issues
5. Add docstrings to ALL functions, classes, and the module (Google style)
6. Add type hints to function parameters and return values
7. Follow PEP 8 style guidelines
8. Make sure the code is syntactically correct

CRITICAL - DO NOT OVERFIT:
- DO NOT write code that only handles the specific test values
- DO write general algorithms that work for ANY valid input
- Example: For find_max([1,5,3]), don't return 5 literally - implement a real max algorithm

⛔ API PRESERVATION RULES ⛔
- KEEP EXACT NAMES: reverseString stays reverseString, NOT reverse_string
- KEEP original naming convention (camelCase or snake_case)
- DO NOT rename functions, classes, methods, or variables
- DO NOT change parameter names or function signatures
- DO NOT delete any existing functions or classes

## OUTPUT FORMAT
Return the COMPLETE Python file. Include every function, class, and import.
Do not truncate or use placeholders like "..." or "# rest of code".
No markdown code blocks. Start directly with Python code."""

        response = self._call_llm(
            prompt=prompt,
            action=ActionType.FIX,
            additional_details={
                "file_fixed": filepath,
                "issues_count": len(issues)
            }
        )
        
        if not response.success:
            self._log(f"  ⚠️ LLM call failed: {response.error}", "WARN")
            return original_code, False
        
        # Extract code from response
        fixed_code = self._extract_code(response.content)
        
        return fixed_code, True
    
    def _fix_syntax_error(
        self,
        filepath: str,
        broken_code: str,
        error_msg: str
    ) -> Tuple[str, bool]:
        """
        Fix syntax error in generated code using LLM.
        
        Args:
            filepath: Path to the file.
            broken_code: The code with syntax error.
            error_msg: The syntax error message.
            
        Returns:
            Tuple of (fixed code string, success boolean).
        """
        prompt = f"""The following Python code has a syntax error. Fix it.

FILE: {filepath}

BROKEN CODE:
```python
{broken_code}
```

SYNTAX ERROR:
{error_msg}

⚠️ CRITICAL: OUTPUT THE COMPLETE FILE ⚠️
- Your response MUST contain ALL functions and classes from the original code
- Do NOT truncate, abbreviate, or skip any functions
- Include EVERY function, EVERY class from the original

Return ONLY the complete fixed Python code, nothing else.
Start directly with the Python code."""

        response = self._call_llm(
            prompt=prompt,
            action=ActionType.FIX,
            additional_details={
                "file_fixed": filepath,
                "fix_type": "syntax_error",
                "error": error_msg
            }
        )
        
        if not response.success:
            self._log(f"  ⚠️ LLM call failed: {response.error}", "WARN")
            return broken_code, False
            
        return self._extract_code(response.content), True

    def _extract_code(self, text: str) -> str:
        """
        Extract Python code from LLM response.
        
        Args:
            text: LLM response text.
            
        Returns:
            Extracted Python code.
        """
        # Try to extract from markdown code block
        patterns = [
            r"```python\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # If no code block, return the text as-is (hopefully it's just code)
        # Remove any leading/trailing non-code text
        lines = text.strip().split('\n')
        
        # Find the first line that looks like Python code
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith(('"""', "'''", '#', 'import ', 'from ', 'def ', 'class ', '@')):
                start_idx = i
                break
        
        return '\n'.join(lines[start_idx:]).strip()
    
    def fix_single_file(
        self,
        filepath: str,
        issues: Optional[List[Dict[str, Any]]] = None,
        error_context: Optional[str] = None
    ) -> FixResult:
        """
        Fix a single file (for use in self-healing loop).
        
        Args:
            filepath: Path to the file.
            issues: Optional list of specific issues.
            error_context: Optional error message from tests.
            
        Returns:
            FixResult with the outcome.
        """
        self._log(f"Fixing single file: {filepath}")
        
        # Convert dict issues to CodeIssue objects
        code_issues = []
        if issues:
            for i in issues:
                code_issues.append(CodeIssue(
                    file_path=filepath,
                    line_number=i.get("line", 1),
                    issue_type=i.get("type", "bug"),
                    severity=i.get("severity", "high"),
                    description=i.get("description", "Unknown issue"),
                    suggested_fix=i.get("fix", "Review and fix this issue")
                ))
        
        # If we have error context from tests, add it as an issue
        if error_context:
            code_issues.append(CodeIssue(
                file_path=filepath,
                line_number=1,
                issue_type="bug",
                severity="critical",
                description=f"Test failure: {error_context}",
                suggested_fix="Fix the code to make tests pass"
            ))
        
        return self._fix_file(filepath, code_issues)
    
    def fix_based_on_test_failure(
        self,
        filepath: str,
        test_output: str,
        traceback: str,
        previously_passing_tests: Optional[List[str]] = None
    ) -> FixResult:
        """
        Fix a file based on test failure feedback.
        
        Args:
            filepath: Path to the file that needs fixing.
            test_output: Test execution output.
            traceback: Error traceback from the test.
            previously_passing_tests: List of test names that were passing before.
                                     These should NOT be broken by the fix.
            
        Returns:
            FixResult with the outcome.
        """
        self._log(f"Fixing {filepath} based on test failure...")
        
        try:
            original_code = self.file_tools.read_file(filepath)
            self.file_tools.backup_file(filepath)
            
            full_path = os.path.join(self.sandbox_root, filepath)
            initial_report = self.analysis_tools.analyze_file(full_path)
            pylint_before = initial_report.score if initial_report.success else 0.0
            
            # Try to find associated test file
            test_code = None
            try:
                dirname = os.path.dirname(filepath)
                basename = os.path.basename(filepath)
                test_filename = f"test_{basename}"
                test_filepath = os.path.join(dirname, test_filename)
                
                if test_filename in self.file_tools.list_files(dirname):
                    test_code = self.file_tools.read_file(test_filepath)
            except Exception:
                pass
            
            # Choose strategy based on file size
            line_count = len(original_code.split('\n'))
            
            # Use full-file test fix for all sizes (DIFF_BASED disabled - unreliable)
            self._log(f"  📄 Test fix: FULL_FILE mode ({line_count} lines)")
            fixed_code, llm_succeeded = self._generate_full_file_test_fix(
                filepath, original_code, test_output, traceback, test_code,
                previously_passing_tests=previously_passing_tests
            )
            
            # If LLM call failed, abort early
            if not llm_succeeded:
                self.file_tools.restore_backup(filepath)
                return FixResult(
                    file_path=filepath,
                    success=False,
                    original_code=original_code,
                    fixed_code=original_code,
                    issues_addressed=[],
                    error_message="LLM call failed - no fix applied",
                    pylint_before=pylint_before,
                    pylint_after=pylint_before
                )
            
            # Normalize line endings (convert CRLF to LF) and ensure final newline
            fixed_code = fixed_code.replace('\r\n', '\n').replace('\r', '\n')
            if not fixed_code.endswith('\n'):
                fixed_code += '\n'
            
            # Check if any actual change was made
            if fixed_code.strip() == original_code.strip():
                self._log(f"  ⚠️ LLM returned unchanged code - no fix applied", "WARN")
                self.file_tools.restore_backup(filepath)
                return FixResult(
                    file_path=filepath,
                    success=False,
                    original_code=original_code,
                    fixed_code=original_code,
                    issues_addressed=[],
                    error_message="LLM returned unchanged code - no fix applied",
                    pylint_before=pylint_before,
                    pylint_after=pylint_before
                )
            
            # Write and validate
            self.file_tools.write_file(filepath, fixed_code)
            
            is_valid, error_msg = self.analysis_tools.check_syntax(full_path)
            if not is_valid:
                self.file_tools.restore_backup(filepath)
                return FixResult(
                    file_path=filepath,
                    success=False,
                    original_code=original_code,
                    fixed_code=fixed_code,
                    issues_addressed=[],
                    error_message=f"Syntax error: {error_msg}",
                    pylint_before=pylint_before,
                    pylint_after=pylint_before
                )
            
            # Check that no functions/classes were deleted
            defs_valid, defs_error = self._validate_no_deleted_definitions(
                original_code, fixed_code
            )
            if not defs_valid:
                self._log(f"  ⚠️ Fix deleted definitions: {defs_error}", "WARN")
                self.file_tools.restore_backup(filepath)
                return FixResult(
                    file_path=filepath,
                    success=False,
                    original_code=original_code,
                    fixed_code=fixed_code,
                    issues_addressed=[],
                    error_message=f"Code truncated - {defs_error}",
                    pylint_before=pylint_before,
                    pylint_after=pylint_before
                )
            
            final_report = self.analysis_tools.analyze_file(full_path)
            pylint_after = final_report.score if final_report.success else 0.0
            
            # If we successfully fixed based on tests, unstuck the file!
            self.stuck_files.pop(filepath, None)
            
            return FixResult(
                file_path=filepath,
                success=True,
                original_code=original_code,
                fixed_code=fixed_code,
                issues_addressed=[f"Test failure fix: {traceback[:100]}"],
                pylint_before=pylint_before,
                pylint_after=pylint_after
            )
            
        except Exception as e:
            try:
                self.file_tools.restore_backup(filepath)
            except Exception as restore_error:
                self._log(f"  ⚠️ Failed to restore backup for {filepath}: {restore_error}", "WARN")
            
            return FixResult(
                file_path=filepath,
                success=False,
                original_code="",
                fixed_code="",
                issues_addressed=[],
                error_message=str(e)
            )

    def _generate_full_file_test_fix(
        self,
        filepath: str,
        original_code: str,
        test_output: str,
        traceback: str,
        test_code: Optional[str] = None,
        previously_passing_tests: Optional[List[str]] = None
    ) -> Tuple[str, bool]:
        """
        Generate test fix by asking LLM to return the complete file.
        
        This is the original approach, used for smaller files.
        
        Args:
            filepath: Path to the file.
            original_code: Original code content.
            test_output: Test failure output.
            traceback: Error traceback.
            test_code: Optional test file content.
            previously_passing_tests: List of tests that were passing and should NOT be broken.
            
        Returns:
            Tuple of (fixed code string, success boolean).
        """
        # Prepare test code section
        test_section = ""
        if test_code:
            test_section = f"""
## TEST FILE (for understanding expected behavior)
```python
{test_code}
```
"""
        
        # Prepare previously passing tests warning
        passing_tests_warning = ""
        if previously_passing_tests:
            # Only include tests relevant to this file
            filename = os.path.basename(filepath)
            relevant_tests = [t for t in previously_passing_tests if filename.replace('.py', '') in t]
            if relevant_tests:
                tests_list = '\n'.join(f"  - {t}" for t in relevant_tests[:20])  # Limit to 20
                passing_tests_warning = f"""
## ⚠️ CRITICAL: DO NOT BREAK THESE PASSING TESTS ⚠️
The following {len(relevant_tests)} test(s) are currently PASSING.
Your fix MUST NOT break them. If your fix would cause any of these to fail, find a different approach.
{tests_list}
"""

        # Truncate traceback to avoid context overflow (keep most relevant parts)
        truncated_traceback = traceback
        if len(traceback) > 2000:
            # Keep first 500 chars (context) and last 1500 chars (actual error)
            truncated_traceback = traceback[:500] + "\n... [truncated] ...\n" + traceback[-1500:]
        
        # Truncate test output similarly
        truncated_test_output = test_output
        if len(test_output) > 1500:
            truncated_test_output = test_output[:500] + "\n... [truncated] ...\n" + test_output[-1000:]

        prompt = f"""You are an expert Python debugger. A test is failing and you must fix it.
{passing_tests_warning}
## SOURCE CODE
FILE: {filepath}
```python
{original_code}
```

## ERROR TRACEBACK (READ THIS CAREFULLY!)
```
{truncated_traceback}
```

## TEST OUTPUT
```
{truncated_test_output}
```
{test_section}
## DEBUGGING INSTRUCTIONS

You MUST follow this debugging process step by step:

### STEP 1: READ THE TRACEBACK
- What is the EXACT error type? (e.g., AttributeError, TypeError, AssertionError)
- What is the EXACT error message?
- Which LINE in the source code caused the error?
- What FUNCTION/METHOD was being called when it failed?

### STEP 2: UNDERSTAND THE TEST
- Find the failing test in the test file
- What INPUTS does the test provide?
- What BEHAVIOR does the test expect?
- What assertion or expectation failed?

### STEP 3: TRACE THE EXECUTION
- Follow the code path from the test to the error
- What values are being passed?
- Where does the actual behavior diverge from expected?

### STEP 4: IDENTIFY ROOT CAUSE
- The traceback shows the SYMPTOM
- What is the underlying BUG that causes this symptom?
- Is it a logic error? Missing check? Wrong operator? Unhandled case?

### STEP 5: FIX
- Fix ONLY the root cause
- Do NOT refactor unrelated code
- Ensure the fix handles the edge case the test is checking

## YOUR ANALYSIS (Think step by step before fixing)
Before writing code, briefly state:
1. Error type and message: [what error?]
2. Failing test: [which test?]
3. Root cause: [why does it fail?]
4. Fix strategy: [what will you change?]

Then provide the fixed code.

## OUTPUT FORMAT
⛔ API PRESERVATION - keep all names, signatures, and conventions unchanged.
Return the COMPLETE Python file with your fix applied.
No markdown blocks. Start directly with Python code."""

        response = self._call_llm(
            prompt=prompt,
            action=ActionType.FIX,
            additional_details={
                "file_fixed": filepath,
                "fix_type": "test_failure",
                "error": traceback[:500]
            }
        )
        
        if not response.success:
            error_msg = response.error or "Unknown error"
            self._log(f"  ⚠️ LLM call failed: {error_msg}", "WARN")
            
            # Check if it's a MAX_TOKENS issue - file might be too large
            if "MAX_TOKENS" in error_msg:
                file_size = len(original_code)
                self._log(f"  ⚠️ File too large ({file_size} chars) - consider splitting or simplifying", "WARN")
            
            return original_code, False
            
        return self._extract_code(response.content), True

