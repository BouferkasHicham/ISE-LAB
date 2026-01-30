"""
Test Tools - Unit test execution using Pytest.

This module provides integration with pytest for running unit tests,
capturing results, and providing feedback to the self-healing loop.
"""

import subprocess
import os
import sys
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TestResult:
    """Represents the result of a single test."""
    name: str
    outcome: str  # "passed", "failed", "error", "skipped"
    duration: float
    file_path: str
    line_number: Optional[int] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    
    @property
    def passed(self) -> bool:
        return self.outcome == "passed"
    
    @property
    def failed(self) -> bool:
        return self.outcome in ("failed", "error")


@dataclass
class TestSuiteResult:
    """Comprehensive test execution results."""
    total: int
    passed: int
    failed: int
    errors: int
    skipped: int
    duration: float
    test_results: List[TestResult]
    raw_output: str
    success: bool
    error_message: Optional[str] = None
    
    @property
    def pass_rate(self) -> float:
        """Calculate the pass rate as a percentage."""
        if self.total == 0:
            return 100.0
        return (self.passed / self.total) * 100
    
    @property
    def all_passed(self) -> bool:
        """Check if all tests passed (requires actual tests to have run successfully)."""
        return self.success and self.total > 0 and self.failed == 0 and self.errors == 0
    
    @property
    def has_execution_error(self) -> bool:
        """Check if there was a timeout or other execution error (not a test failure)."""
        return not self.success and self.error_message is not None
    
    def get_failed_tests(self) -> List[TestResult]:
        """Get only the failed tests."""
        return [t for t in self.test_results if t.failed]
    
    def get_summary(self) -> str:
        """Get a formatted summary of test results."""
        status = "✅ ALL TESTS PASSED" if self.all_passed else "❌ TESTS FAILED"
        
        summary = f"""
{status}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Tests: {self.total}
  ✅ Passed:  {self.passed}
  ❌ Failed:  {self.failed}
  💥 Errors:  {self.errors}
  ⏭️  Skipped: {self.skipped}
  ⏱️  Duration: {self.duration:.2f}s
  📊 Pass Rate: {self.pass_rate:.1f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return summary
    
    def get_failure_details(self) -> str:
        """Get detailed information about failed tests."""
        if self.all_passed:
            return "No failures."
        
        details = ["Failed Tests Details:", "=" * 50]
        
        for test in self.get_failed_tests():
            details.append(f"\n❌ {test.name}")
            details.append(f"   File: {test.file_path}")
            if test.line_number:
                details.append(f"   Line: {test.line_number}")
            if test.error_message:
                details.append(f"   Error: {test.error_message}")
            if test.traceback:
                details.append(f"   Traceback:\n{test.traceback}")
        
        return "\n".join(details)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for logging."""
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "duration": self.duration,
            "pass_rate": self.pass_rate,
            "all_passed": self.all_passed,
            "raw_output": self.raw_output,  # Include full pytest output for LLM
            "failed_tests": [
                {
                    "name": t.name,
                    "file": t.file_path,
                    "error": t.error_message,
                    "traceback": t.traceback
                }
                for t in self.get_failed_tests()
            ]
        }


class TestTools:
    """
    Test execution tools using pytest.
    
    Provides functionality to run tests, capture results,
    and provide detailed feedback for the self-healing loop.
    """
    
    def __init__(self, sandbox_root: str):
        """
        Initialize test tools.
        
        Args:
            sandbox_root: Root directory for test execution.
        """
        self.sandbox_root = os.path.abspath(sandbox_root)
    
    def run_tests(
        self,
        test_path: str = "",
        verbose: bool = True,
        timeout: int = 600
    ) -> TestSuiteResult:
        """
        Run pytest on the specified path.
        
        Args:
            test_path: Path to test file or directory (relative to sandbox).
            verbose: Enable verbose output.
            timeout: Maximum execution time in seconds (default 10 minutes for large test suites).
            
        Returns:
            TestSuiteResult with comprehensive test results.
        """
        if test_path:
            target = os.path.join(self.sandbox_root, test_path)
        else:
            target = self.sandbox_root
        
        if not os.path.exists(target):
            return TestSuiteResult(
                total=0,
                passed=0,
                failed=0,
                errors=0,
                skipped=0,
                duration=0,
                test_results=[],
                raw_output="",
                success=False,
                error_message=f"Test path not found: {target}"
            )
        
        try:
            # Build pytest command with JSON report
            cmd = [
                sys.executable, "-m", "pytest",
                target,
                "-v" if verbose else "",
                "--tb=short",  # Shorter tracebacks
                "-q",
                "--no-header",
            ]
            # Remove empty strings from command
            cmd = [c for c in cmd if c]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.sandbox_root
            )
            
            return self._parse_pytest_output(result.stdout + result.stderr, result.returncode)
            
        except subprocess.TimeoutExpired:
            return TestSuiteResult(
                total=0,
                passed=0,
                failed=0,
                errors=0,
                skipped=0,
                duration=timeout,
                test_results=[],
                raw_output="",
                success=False,
                error_message=f"Test execution timed out after {timeout}s"
            )
        except Exception as e:
            return TestSuiteResult(
                total=0,
                passed=0,
                failed=0,
                errors=0,
                skipped=0,
                duration=0,
                test_results=[],
                raw_output="",
                success=False,
                error_message=str(e)
            )
    
    def _parse_pytest_output(self, output: str, return_code: int) -> TestSuiteResult:
        """
        Parse pytest output to extract test results.
        
        Args:
            output: Combined stdout and stderr from pytest.
            return_code: Process return code.
            
        Returns:
            Parsed TestSuiteResult.
        """
        test_results = []
        
        # Parse individual test results from verbose output
        # Patterns to match various pytest output formats:
        # - test_file.py::test_name PASSED/FAILED
        # - test_file.py::ClassName::test_name PASSED/FAILED  
        # - test_file.py::ClassName::test_name[param] PASSED/FAILED
        # - tests/test_file.py::ClassName::NestedClass::test_name PASSED/FAILED
        # Capture: (file_path)::(test_identifier) (outcome)
        # Pattern handles various spacing/formatting from different pytest versions
        test_pattern = r"(\S+\.py)::(\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|passed|failed|error|skipped)"
        for match in re.finditer(test_pattern, output):
            file_path, test_id, outcome = match.groups()
            # Extract just the test name (last component after ::)
            test_name = test_id.split("::")[-1]
            # Remove parametrize brackets if present
            if "[" in test_name:
                test_name = test_name.split("[")[0]
            test_results.append(TestResult(
                name=test_name,
                outcome=outcome.lower(),
                duration=0.0,  # Duration not easily extractable from simple output
                file_path=file_path
            ))
        
        # Parse summary line: "X passed, Y failed, Z errors in X.XXs"
        summary_pattern = r"(\d+)\s+passed|(\d+)\s+failed|(\d+)\s+error|(\d+)\s+skipped"
        passed = failed = errors = skipped = 0
        
        for match in re.finditer(summary_pattern, output, re.IGNORECASE):
            if match.group(1):
                passed = int(match.group(1))
            if match.group(2):
                failed = int(match.group(2))
            if match.group(3):
                errors = int(match.group(3))
            if match.group(4):
                skipped = int(match.group(4))
        
        # Also check for simpler patterns
        simple_passed = re.search(r"(\d+) passed", output)
        simple_failed = re.search(r"(\d+) failed", output)
        simple_error = re.search(r"(\d+) error", output)
        simple_skipped = re.search(r"(\d+) skipped", output)
        
        if simple_passed:
            passed = int(simple_passed.group(1))
        if simple_failed:
            failed = int(simple_failed.group(1))
        if simple_error:
            errors = int(simple_error.group(1))
        if simple_skipped:
            skipped = int(simple_skipped.group(1))
        
        # Extract duration
        duration_pattern = r"in\s+(\d+\.?\d*)s"
        duration_match = re.search(duration_pattern, output)
        duration = float(duration_match.group(1)) if duration_match else 0.0
        
        total = passed + failed + errors + skipped
        
        # Raw output is passed directly to LLM - no parsing needed
        # The LLM can understand any test output format
        
        # Check for collection errors (no tests found but with errors)
        if "error" in output.lower() and total == 0:
            error_match = re.search(r"(ModuleNotFoundError|ImportError|SyntaxError)[^\n]+", output)
            if error_match:
                return TestSuiteResult(
                    total=0,
                    passed=0,
                    failed=0,
                    errors=1,
                    skipped=0,
                    duration=duration,
                    test_results=[TestResult(
                        name="collection_error",
                        outcome="error",
                        duration=0,
                        file_path="",
                        error_message=error_match.group(0)
                    )],
                    raw_output=output,
                    success=False,
                    error_message=error_match.group(0)
                )
        
        return TestSuiteResult(
            total=total,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            duration=duration,
            test_results=test_results,
            raw_output=output,
            success=return_code == 0
        )
    
    
    def discover_tests(self, directory: str = "") -> List[str]:
        """
        Discover all test files in a directory.
        
        Args:
            directory: Directory to search (relative to sandbox).
            
        Returns:
            List of test file paths.
        """
        search_dir = os.path.join(self.sandbox_root, directory) if directory else self.sandbox_root
        test_files = []
        
        for root, _, files in os.walk(search_dir):
            for file in files:
                if file.startswith("test_") and file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.sandbox_root)
                    test_files.append(rel_path)
        
        return sorted(test_files)
    
    def run_single_test(
        self,
        test_file: str,
        test_name: str,
        timeout: int = 60
    ) -> TestResult:
        """
        Run a single specific test.
        
        Args:
            test_file: Path to the test file.
            test_name: Name of the test function.
            timeout: Execution timeout.
            
        Returns:
            TestResult for the specific test.
        """
        test_path = os.path.join(self.sandbox_root, test_file)
        test_spec = f"{test_path}::{test_name}"
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_spec, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.sandbox_root
            )
            
            suite_result = self._parse_pytest_output(
                result.stdout + result.stderr,
                result.returncode
            )
            
            if suite_result.test_results:
                return suite_result.test_results[0]
            
            return TestResult(
                name=test_name,
                outcome="passed" if result.returncode == 0 else "failed",
                duration=suite_result.duration,
                file_path=test_file,
                error_message=suite_result.error_message
            )
            
        except Exception as e:
            return TestResult(
                name=test_name,
                outcome="error",
                duration=0,
                file_path=test_file,
                error_message=str(e)
            )
    
    def check_test_file_syntax(self, test_file: str) -> Tuple[bool, str]:
        """
        Check if a test file has valid Python syntax.
        
        Args:
            test_file: Path to the test file.
            
        Returns:
            Tuple of (is_valid, error_message).
        """
        test_path = os.path.join(self.sandbox_root, test_file)
        
        try:
            with open(test_path, 'r', encoding='utf-8') as f:
                source = f.read()
            compile(source, test_path, 'exec')
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)
    
    def format_results_for_prompt(self, results: TestSuiteResult) -> str:
        """
        Format test results for LLM prompts.
        
        Args:
            results: Test suite results.
            
        Returns:
            Formatted string suitable for LLM context.
        """
        output = [results.get_summary()]
        
        if not results.all_passed:
            output.append("\n" + results.get_failure_details())
        
        return "\n".join(output)
