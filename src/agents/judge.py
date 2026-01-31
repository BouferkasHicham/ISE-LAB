"""
Judge Agent - Test Execution and Validation.

The Judge executes unit tests and validates whether fixes are successful.
If tests fail, it provides feedback for the self-healing loop.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.agents.base_agent import BaseAgent
from src.tools.file_tools import FileTools
from src.tools.test_tools import TestTools, TestSuiteResult
from src.tools.analysis_tools import AnalysisTools
from src.utils.logger import ActionType


class JudgmentResult(Enum):
    """Possible judgment outcomes."""
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    NO_TESTS = "NO_TESTS"


@dataclass
class Judgment:
    """Result of the Judge's evaluation."""
    result: JudgmentResult
    test_results: Optional[TestSuiteResult]
    pylint_scores: Dict[str, float]
    average_pylint_score: float
    feedback: str
    files_to_fix: List[str]  # Files with low Pylint scores
    files_with_test_failures: List[str] = field(default_factory=list)  # Files with failing tests
    error_details: Optional[str] = None
    test_file_errors: List[Dict[str, Any]] = field(default_factory=list)  # Errors IN test files (not source)
    
    @property
    def passed(self) -> bool:
        """Check if the judgment passed."""
        return self.result == JudgmentResult.PASS
    
    @property
    def needs_fixing(self) -> bool:
        """Check if more fixes are needed."""
        return self.result in (JudgmentResult.FAIL, JudgmentResult.ERROR)
    
    @property
    def has_test_file_errors(self) -> bool:
        """Check if there are errors in test files themselves."""
        return len(self.test_file_errors) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert judgment to dictionary."""
        return {
            "result": self.result.value,
            "passed": self.passed,  # Include the passed property
            "test_results": self.test_results.to_dict() if self.test_results else None,
            "pylint_scores": self.pylint_scores,
            "average_pylint_score": self.average_pylint_score,
            "feedback": self.feedback,
            "files_to_fix": self.files_to_fix,
            "files_with_test_failures": self.files_with_test_failures,
            "error_details": self.error_details,
            "test_file_errors": self.test_file_errors,
            "has_test_file_errors": self.has_test_file_errors
        }


class JudgeAgent(BaseAgent):
    """
    The Judge Agent validates code fixes through testing.
    
    Responsibilities:
    - Execute unit tests using pytest
    - Verify code quality improvements via Pylint
    - Provide detailed feedback for failed tests
    - Determine if the self-healing loop should continue
    - Confirm mission completion when all tests pass
    """
    
    def __init__(
        self,
        sandbox_root: str,
        pylint_threshold: float = 8.5,
        llm=None,
        model_name: str = "gemini-2.5-pro",
        verbose: bool = True
    ):
        """
        Initialize the Judge Agent.
        
        Args:
            sandbox_root: Root directory containing code to test.
            pylint_threshold: Minimum acceptable Pylint score.
            llm: Optional pre-configured LLM instance.
            model_name: Gemini model to use.
            verbose: Enable verbose output.
        """
        super().__init__(
            name="Judge_Agent",
            llm=llm,
            model_name=model_name,
            verbose=verbose
        )
        
        self.sandbox_root = sandbox_root
        self.pylint_threshold = pylint_threshold
        self.file_tools = FileTools(sandbox_root)
        self.test_tools = TestTools(sandbox_root)
        self.analysis_tools = AnalysisTools(sandbox_root)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Judge agent."""
        return """You are an expert code reviewer and quality assurance specialist.

Your role is to:
1. Analyze test results and determine pass/fail status
2. Identify root causes of test failures
3. Provide specific, actionable feedback for fixing issues
4. Validate code quality against Pylint standards
5. Map test failures to specific source files

When analyzing test failures:
- READ THE TRACEBACK carefully to identify the exact error
- Identify which SOURCE file (not test file) contains the bug
- Explain what the code is doing wrong vs what it should do
- Suggest a specific, minimal fix
- Consider if the fix might break other tests

When analyzing Pylint scores:
- Scores below 8.5 need improvement
- Common issues: missing docstrings, unused imports, naming conventions
- Prioritize issues by severity (errors > warnings > conventions)

Output format:
- Be precise and technical
- Use JSON format when requested
- List files by their relative path, not full path
- Return only source files, never test files"""
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the full judgment process.
        
        Returns:
            Dictionary containing the judgment and detailed results.
        """
        self._update_state(status="working", task="Running tests and validation")
        self._log("Starting code validation...")
        
        try:
            # Step 1: Run all tests
            test_results = self._run_tests()
            
            # Step 2: Run Pylint analysis
            pylint_scores = self._analyze_code_quality()
            
            # Step 3: Make judgment
            judgment = self._make_judgment(test_results, pylint_scores)
            
            self._update_state(status="completed", result=judgment.to_dict())
            
            status = "✅ PASSED" if judgment.passed else "❌ FAILED"
            self._log(f"Judgment: {status}", "SUCCESS" if judgment.passed else "ERROR")
            
            return {
                "success": True,
                "judgment": judgment,
                "passed": judgment.passed,
                "needs_fixing": judgment.needs_fixing,
                "feedback": judgment.feedback
            }
            
        except Exception as e:
            self._update_state(status="failed", error=str(e))
            self._log(f"Validation failed: {e}", "ERROR")
            return {
                "success": False,
                "error": str(e),
                "judgment": None
            }
    
    def _run_tests(self) -> TestSuiteResult:
        """
        Run all tests in the sandbox.
        
        Returns:
            TestSuiteResult with test execution results.
        """
        self._log("Running unit tests...")
        
        # First, discover test files
        test_files = self.test_tools.discover_tests()
        
        if not test_files:
            self._log("No test files found", "WARN")
            return TestSuiteResult(
                total=0,
                passed=0,
                failed=0,
                errors=0,
                skipped=0,
                duration=0,
                test_results=[],
                raw_output="No test files found",
                success=True
            )
        
        self._log(f"Found {len(test_files)} test file(s)")
        
        # Run tests
        results = self.test_tools.run_tests(verbose=True)
        
        # Log results
        if results.has_execution_error:
            self._log(f"Test execution error: {results.error_message}", "ERROR")
        elif results.all_passed:
            self._log(f"All tests passed! ({results.passed}/{results.total})", "SUCCESS")
        elif results.total == 0:
            self._log(f"No tests collected (test files exist but 0 tests found)", "WARN")
        else:
            self._log(f"Tests failed: {results.failed} failed, {results.errors} errors", "ERROR")
        
        return results
    
    def _analyze_code_quality(self) -> Dict[str, float]:
        """
        Analyze code quality with Pylint.
        
        Only analyzes SOURCE files, not test files.
        Test files are the ground truth and should not be modified.
        
        Returns:
            Dictionary mapping file paths to Pylint scores.
        """
        self._log("Analyzing code quality with Pylint...")
        
        all_python_files = self.file_tools.list_python_files()
        
        # Exclude test files - they are ground truth, not to be judged or fixed
        python_files = [
            f for f in all_python_files
            if not os.path.basename(f).startswith('test_')
            and not os.path.basename(f).endswith('_test.py')
        ]
        
        scores = {}
        
        for filepath in python_files:
            # Use os.path.join for proper path construction
            full_path = os.path.join(self.sandbox_root, filepath)
            report = self.analysis_tools.analyze_file(full_path)
            scores[filepath] = report.score if report.success else 0.0
        
        if scores:
            avg_score = sum(scores.values()) / len(scores)
            self._log(f"Average Pylint score: {avg_score:.2f}/10")
        
        return scores
    
    def _identify_files_from_test_output(
        self,
        raw_test_output: str,
        available_source_files: List[str]
    ) -> List[str]:
        """
        Use LLM to identify which source files need fixing based on raw test output.
        
        This is more robust than regex parsing because:
        - Works with any test framework output format
        - Handles edge cases and unusual formats
        - Can understand context and error messages
        
        Args:
            raw_test_output: Raw output from pytest/unittest/etc.
            available_source_files: List of source files that could need fixing.
            
        Returns:
            List of source file paths that need fixing.
        """
        if not raw_test_output or not available_source_files:
            return []
        
        # Format available files for the prompt
        files_list = "\n".join(f"- {f}" for f in available_source_files)
        
        prompt = f"""Analyze this test output and identify which SOURCE files need to be fixed.

## AVAILABLE SOURCE FILES (choose from these only):
{files_list}

## RAW TEST OUTPUT:
```
{raw_test_output}
```

## INSTRUCTIONS:
1. Read the test output carefully
2. Identify which tests failed
3. Determine which SOURCE file(s) contain the bugs causing the failures
4. Only list files from the AVAILABLE SOURCE FILES list above
5. Do NOT list test files (test_*.py) - only source files

## RESPONSE FORMAT:
Return ONLY a JSON array of source file paths that need fixing.
Example: ["calculator.py", "utils.py"]
If no files can be identified, return: []

YOUR RESPONSE (JSON array only):"""

        try:
            llm_response = self._call_llm(
                prompt=prompt,
                action=ActionType.DEBUG,
                additional_details={"task": "identify_files_from_test_output"}
            )
            
            if not llm_response.success:
                self._log(f"LLM call failed: {llm_response.error}", "WARN")
                return []
            
            # Parse JSON response
            import json
            # Clean up response - extract JSON array
            response_text = llm_response.content.strip()
            # Find the JSON array in the response
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                files = json.loads(json_str)
                
                # Validate that returned files are in available_source_files
                valid_files = []
                for f in files:
                    # Match by basename or full path
                    for available in available_source_files:
                        if f == available or os.path.basename(available) == f:
                            if available not in valid_files:
                                valid_files.append(available)
                            break
                
                return valid_files
        except Exception as e:
            self._log(f"LLM file identification failed: {e}", "WARN")
        
        return []

    def _make_judgment(
        self,
        test_results: TestSuiteResult,
        pylint_scores: Dict[str, float]
    ) -> Judgment:
        """
        Make the final judgment based on tests AND code quality.
        
        Lab requirements:
        - Tests must pass (40% of grade)
        - Pylint score must meet threshold
        
        Args:
            test_results: Results from running tests.
            pylint_scores: Pylint scores for each file.
            
        Returns:
            Judgment with the final decision.
        """
        # Calculate average Pylint score
        avg_pylint = sum(pylint_scores.values()) / len(pylint_scores) if pylint_scores else 0.0
        
        # Identify files with low Pylint (below threshold)
        files_to_fix = []
        for filepath, score in pylint_scores.items():
            if score < self.pylint_threshold:
                files_to_fix.append(filepath)
        
        # Track files with test failures - will be identified by LLM
        files_with_test_failures: List[str] = []
        
        # Get raw test output for LLM analysis
        raw_output = test_results.raw_output if hasattr(test_results, 'raw_output') else ""
        available_source_files = list(pylint_scores.keys())
        
        # CRITICAL: Detect errors IN test files themselves (not source code bugs)
        # These cannot be fixed by the Fixer and require test regeneration
        test_file_errors = self._detect_test_file_errors(raw_output)
        if test_file_errors:
            self._log(f"⚠️ CRITICAL: Found {len(test_file_errors)} error(s) IN TEST FILES", "ERROR")
            for err in test_file_errors:
                self._log(f"  → {err['type']}: {err['message']}", "ERROR")
        
        # Check for test execution errors
        if test_results.errors > 0:
            result = JudgmentResult.ERROR
            feedback = self._generate_error_feedback(test_results)
            
            # If test file errors exist, report them prominently
            if test_file_errors:
                feedback += "\n\n🚨 CRITICAL: ERRORS IN TEST FILES (not source code):"
                feedback += "\nThese cannot be fixed by modifying source code. Test regeneration required."
                for err in test_file_errors:
                    feedback += f"\n  • [{err['type']}] {err.get('test_file', 'unknown')}: {err['message']}"
            
            # Use LLM to identify which files need fixing from raw output
            files_with_test_failures = self._identify_files_from_test_output(
                raw_output, available_source_files
            )
            
            # Do NOT fallback if LLM couldn't identify files - better to skip than fix wrong files
            if not files_with_test_failures:
                self._log("⚠️ Could not identify specific files with test failures", "WARN")
            
            return Judgment(
                result=result,
                test_results=test_results,
                pylint_scores=pylint_scores,
                average_pylint_score=avg_pylint,
                feedback=feedback,
                files_to_fix=files_to_fix,
                files_with_test_failures=files_with_test_failures,
                error_details="Test execution errors",
                test_file_errors=test_file_errors
            )
        
        # Check for test failures
        if not test_results.all_passed and test_results.total > 0:
            result = JudgmentResult.FAIL
            feedback = self._generate_failure_feedback(test_results)
            feedback += f"\n\nPylint average: {avg_pylint:.2f}/10"
            
            # Use LLM to identify which files need fixing from raw output
            files_with_test_failures = self._identify_files_from_test_output(
                raw_output, available_source_files
            )
            
            # Do NOT fallback if LLM couldn't identify files - better to skip than fix wrong files
            if not files_with_test_failures:
                self._log("⚠️ Could not identify specific files with test failures", "WARN")
            
            return Judgment(
                result=result,
                test_results=test_results,
                pylint_scores=pylint_scores,
                average_pylint_score=avg_pylint,
                feedback=feedback,
                files_to_fix=files_to_fix,
                files_with_test_failures=files_with_test_failures,
                error_details=None
            )
        
        # Check for test execution errors (timeout, import errors, etc.)
        if test_results.has_execution_error:
            result = JudgmentResult.ERROR
            feedback = f"⚠️ Test execution error: {test_results.error_message}"
            self._log(f"Test execution error: {test_results.error_message}", "ERROR")
            
            return Judgment(
                result=result,
                test_results=test_results,
                pylint_scores=pylint_scores,
                average_pylint_score=avg_pylint,
                feedback=feedback,
                files_to_fix=files_to_fix,
                files_with_test_failures=[],
                error_details=test_results.error_message
            )
        
        # If no tests exist (and no execution error), judgment is based on Pylint only
        if test_results.total == 0:
            all_files_pass = all(score >= self.pylint_threshold for score in pylint_scores.values()) if pylint_scores else False
            if all_files_pass and not files_to_fix:
                result = JudgmentResult.NO_TESTS
                feedback = f"No tests found. All files meet Pylint threshold ({self.pylint_threshold}/10) ✅"
            else:
                result = JudgmentResult.FAIL
                low_files = [f"{f}: {pylint_scores[f]:.2f}" for f in files_to_fix]
                feedback = f"No tests found. Code quality needs improvement (Avg: {avg_pylint:.2f}/10). Low scores: {', '.join(low_files)}"
            
            return Judgment(
                result=result,
                test_results=test_results,
                pylint_scores=pylint_scores,
                average_pylint_score=avg_pylint,
                feedback=feedback,
                files_to_fix=files_to_fix,
                files_with_test_failures=[],  # No tests, so no test failures
                error_details=None
            )
        
        # All tests passed - now check Pylint (each file must meet threshold)
        all_files_pass = all(score >= self.pylint_threshold for score in pylint_scores.values()) if pylint_scores else False
        if all_files_pass and not files_to_fix:
            result = JudgmentResult.PASS
            feedback = f"All tests passed ✅ and all files meet Pylint threshold ({self.pylint_threshold}/10)! Average: {avg_pylint:.2f}/10"
        else:
            result = JudgmentResult.FAIL
            low_files = [f"{os.path.basename(f)}: {pylint_scores[f]:.2f}" for f in files_to_fix]
            feedback = f"Tests passed but {len(files_to_fix)} file(s) below threshold ({self.pylint_threshold}/10): {', '.join(low_files)}"
        
        return Judgment(
            result=result,
            test_results=test_results,
            pylint_scores=pylint_scores,
            average_pylint_score=avg_pylint,
            feedback=feedback,
            files_to_fix=files_to_fix,
            files_with_test_failures=[],  # Tests passed, no failures
            error_details=None
        )
    
    def _generate_error_feedback(self, test_results: TestSuiteResult) -> str:
        """Generate feedback for test errors - just summarize, raw output has details."""
        error_count = sum(1 for t in test_results.get_failed_tests() if t.outcome == "error")
        return f"Test execution encountered {error_count} error(s). See raw test output for details."
    
    def _generate_failure_feedback(self, test_results: TestSuiteResult) -> str:
        """Generate feedback summary for test failures - raw output has full details."""
        return f"❌ {test_results.failed} test(s) failed. See raw test output for details."
    
    def analyze_failure(
        self,
        test_results: TestSuiteResult,
        source_code: str,
        test_code: str
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze test failure and suggest fixes.
        
        Args:
            test_results: Failed test results.
            source_code: The source code being tested.
            test_code: The test code that failed.
            
        Returns:
            Dictionary with analysis and suggestions.
        """
        self._log("Analyzing test failure with LLM...")
        
        failed_tests = test_results.get_failed_tests()
        if not failed_tests:
            return {"error": "No failed tests to analyze"}
        
        failure_info = []
        for test in failed_tests:
            info = f"Test: {test.name}\n"
            if test.error_message:
                info += f"Error: {test.error_message}\n"
            if test.traceback:
                info += f"Traceback:\n{test.traceback}\n"
            failure_info.append(info)
        
        prompt = f"""Analyze this test failure and identify the root cause.

SOURCE CODE:
```python
{source_code}
```

TEST CODE:
```python
{test_code}
```

FAILURE DETAILS:
{chr(10).join(failure_info)}

Respond with JSON containing:
{{
    "root_cause": "Brief description of what's wrong",
    "affected_lines": [list of line numbers in source that need fixing],
    "suggested_fix": "Specific code change to fix the issue",
    "explanation": "Detailed explanation of why this fix works"
}}"""

        response = self._call_llm_json(
            prompt=prompt,
            action=ActionType.DEBUG,
            additional_details={
                "failed_tests": len(failed_tests),
                "test_names": [t.name for t in failed_tests]
            }
        )
        
        return response
    
    def quick_check(self) -> Tuple[bool, str]:
        """
        Quick check if tests pass without full analysis.
        
        Returns:
            Tuple of (all_passed, message).
        """
        test_results = self.test_tools.run_tests(verbose=False)
        
        if test_results.all_passed:
            return True, f"All {test_results.passed} tests passed"
        else:
            return False, f"Failed: {test_results.failed} tests, {test_results.errors} errors"
    
    def validate_single_file(self, filepath: str) -> Dict[str, Any]:
        """
        Validate a single file's tests and quality.
        
        Args:
            filepath: Path to the source file.
            
        Returns:
            Validation results for the file.
        """
        self._log(f"Validating single file: {filepath}")
        
        # Check Pylint score
        full_path = os.path.join(self.sandbox_root, filepath)
        report = self.analysis_tools.analyze_file(full_path)
        
        # Find corresponding test file (test_<basename> in same directory)
        dirname = os.path.dirname(filepath)
        basename = os.path.basename(filepath)
        test_filename = f"test_{basename}"
        test_file = os.path.join(dirname, test_filename) if dirname else test_filename
        test_results = None
        
        if self.file_tools.file_exists(test_file):
            test_results = self.test_tools.run_tests(test_file)
        
        return {
            "filepath": filepath,
            "pylint_score": report.score,
            "pylint_passed": report.score >= self.pylint_threshold,
            "tests_passed": test_results.all_passed if test_results else None,
            "test_count": test_results.total if test_results else 0,
            "issues": len(report.messages)
        }
    
    def get_detailed_feedback(self, judgment: Judgment) -> str:
        """
        Generate detailed feedback for the Fixer agent.
        
        Args:
            judgment: The judgment result.
            
        Returns:
            Detailed feedback string.
        """
        feedback = [f"JUDGE FEEDBACK - Result: {judgment.result.value}"]
        feedback.append("=" * 50)
        feedback.append(f"\n{judgment.feedback}")
        
        if judgment.files_to_fix:
            feedback.append(f"\n\nFiles requiring attention ({len(judgment.files_to_fix)}):")
            for filepath in judgment.files_to_fix:
                score = judgment.pylint_scores.get(filepath, 0.0)
                feedback.append(f"  • {filepath} (Pylint: {score:.2f}/10)")
        
        if judgment.test_results and not judgment.test_results.all_passed:
            feedback.append("\n\nFailed Tests:")
            for test in judgment.test_results.get_failed_tests():
                feedback.append(f"  ❌ {test.name}")
                if test.error_message:
                    feedback.append(f"     → {test.error_message}")
        
        feedback.append("\n" + "=" * 50)
        
        return "\n".join(feedback)

    def _detect_test_file_errors(self, raw_output: str) -> List[Dict[str, Any]]:
        """
        Detect errors that are IN the test files themselves (not source code bugs).
        
        These are structural/syntax errors in test files that the Fixer cannot fix
        because the problem is in the LLM-generated tests, not the source code.
        
        Common patterns:
        - "fixture 'self' not found" - test function has 'self' param outside class
        - "SyntaxError" in test file
        - "NameError" for undefined test helpers
        
        Args:
            raw_output: Raw pytest output.
            
        Returns:
            List of dicts with test file error details.
        """
        test_file_errors = []
        
        # Pattern 1: fixture 'self' not found (orphan self parameter)
        if "fixture 'self' not found" in raw_output:
            # Try to extract which test file
            import re
            # Look for patterns like "test_calculator.py::test_something"
            matches = re.findall(r'(test_\w+\.py)::(\w+)', raw_output)
            for test_file, test_name in matches:
                if "fixture 'self' not found" in raw_output:
                    test_file_errors.append({
                        "type": "orphan_self_parameter",
                        "test_file": test_file,
                        "test_function": test_name,
                        "message": "Test function has 'self' parameter but is not inside a class",
                        "fixable_by_fixer": False,
                        "requires_test_regeneration": True
                    })
        
        # Pattern 2: Syntax errors in test files
        import re
        syntax_errors = re.findall(
            r'(test_\w+\.py).*?SyntaxError:\s*(.+?)(?:\n|$)',
            raw_output,
            re.DOTALL
        )
        for test_file, error_msg in syntax_errors:
            test_file_errors.append({
                "type": "syntax_error",
                "test_file": test_file,
                "message": f"Syntax error in test file: {error_msg.strip()}",
                "fixable_by_fixer": False,
                "requires_test_regeneration": True
            })
        
        # Pattern 3: IndentationError in test files  
        indent_errors = re.findall(
            r'(test_\w+\.py).*?IndentationError:\s*(.+?)(?:\n|$)',
            raw_output,
            re.DOTALL
        )
        for test_file, error_msg in indent_errors:
            test_file_errors.append({
                "type": "indentation_error",
                "test_file": test_file,
                "message": f"Indentation error in test file: {error_msg.strip()}",
                "fixable_by_fixer": False,
                "requires_test_regeneration": True
            })
        
        # Pattern 4: Collection errors (pytest couldn't even collect tests)
        if "ERROR" in raw_output and "collecting" in raw_output.lower():
            collect_errors = re.findall(
                r'ERROR\s+collecting\s+(test_\w+\.py)',
                raw_output
            )
            for test_file in collect_errors:
                if not any(e.get("test_file") == test_file for e in test_file_errors):
                    test_file_errors.append({
                        "type": "collection_error",
                        "test_file": test_file,
                        "message": f"Pytest could not collect tests from {test_file}",
                        "fixable_by_fixer": False,
                        "requires_test_regeneration": True
                    })
        
        return test_file_errors
