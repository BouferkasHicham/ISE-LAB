"""
Test Generator Agent - Automatic Test File Generation.

The TestGenerator reads the Auditor's analysis and source code,
then generates comprehensive pytest test files for each source file.
"""

import os
import re
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agents.base_agent import BaseAgent
from src.agents.auditor import RefactoringPlan, CodeIssue
from src.tools.file_tools import FileTools
from src.utils.logger import ActionType


@dataclass
class GeneratedTest:
    """Represents a generated or validated test file."""
    source_file: str
    test_file: str
    test_code: str
    functions_tested: List[str]
    success: bool
    error_message: Optional[str] = None
    was_existing: bool = False  # True if test file already existed and was validated


@dataclass
class TestGenerationReport:
    """Report of all test generation results."""
    files_processed: List[str]
    tests_generated: List[str]
    total_functions_tested: int
    successful: int
    failed: int
    tests_reused: int = 0  # Count of existing tests that were reused
    results: List[GeneratedTest] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "files_processed": self.files_processed,
            "tests_generated": self.tests_generated,
            "total_functions_tested": self.total_functions_tested,
            "successful": self.successful,
            "failed": self.failed,
            "tests_reused": self.tests_reused,
            "results": [
                {
                    "source_file": r.source_file,
                    "test_file": r.test_file,
                    "functions_tested": r.functions_tested,
                    "success": r.success,
                    "error": r.error_message,
                    "was_existing": r.was_existing
                }
                for r in self.results
            ]
        }


class TestGeneratorAgent(BaseAgent):
    """
    The TestGenerator Agent creates pytest test files for source code.
    
    Responsibilities:
    - Read source code and understand function signatures
    - Use Auditor's analysis to understand expected behavior
    - Generate comprehensive pytest test files
    - Cover normal cases, edge cases, and error conditions
    - Write test files to the sandbox directory
    
    This agent runs AFTER the Auditor and BEFORE the Fixer.
    """
    
    def __init__(
        self,
        sandbox_root: str,
        llm=None,
        model_name: str = "gemini-2.5-pro",
        verbose: bool = True
    ):
        """
        Initialize the TestGenerator Agent.
        
        Args:
            sandbox_root: Root directory containing code to generate tests for.
            llm: Optional pre-configured LLM instance.
            model_name: Gemini model to use.
            verbose: Enable verbose output.
        """
        super().__init__(
            name="TestGenerator_Agent",
            llm=llm,
            model_name=model_name,
            verbose=verbose
        )
        
        self.sandbox_root = sandbox_root
        self.file_tools = FileTools(sandbox_root)
        
        # Parallel processing settings
        self.MAX_PARALLEL_FILES = 3  # Max files to process in parallel
        self._log_lock = threading.Lock()
    
    def _log_safe(self, message: str, level: str = "INFO"):
        """Thread-safe logging method."""
        with self._log_lock:
            self._log(message, level)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the TestGenerator agent."""
        return """You are an expert Python test engineer specializing in pytest.

Generate tests for the code. Test that it works correctly - doesn't crash, handles edge cases, 
returns appropriate types, handles errors properly.

CRITICAL PRINCIPLE:
You can test specific output values when the code's intent is absolutely clear 
(standard algorithms like fibonacci, well-known formulas, obvious calculations).
But if you're not certain what the "correct" output should be, test behavior instead of values.

Examples:
- fibonacci(10) == 55 → OK to test (standard algorithm, well-known values)
- calculate_discount(100) == 85 → DON'T test specific value (you don't know the rate)
  Instead: test it returns a number, doesn't crash, handles edge cases
- sort_users() returns list → OK to test type and that it doesn't crash

It's better to generate fewer confident tests than many tests with guessed values.
If the code looks correct, the tests should pass. Your tests should catch real bugs, 
not enforce your assumptions about business logic.

CRITICAL RULES - AVOID THESE COMMON MISTAKES:

1. ALWAYS IMPORT FROM SOURCE - Never redefine classes from the source module:
   ✓ CORRECT: from mymodule import MyClass, Event, Handler
   ✗ WRONG: class Event: ...  # Don't redefine source classes in tests!
   
2. USE REAL IMPORTS - Don't create mock classes that shadow source classes.
   If you need a fake/stub, name it explicitly: FakeDatabase, MockHandler, StubEvent

3. TEST BEHAVIOR, NOT PYTHON INTERNALS:
   ✗ WRONG: assert (a < b) is NotImplemented  # Python raises TypeError, not returns NotImplemented
   ✓ CORRECT: assert a.__lt__(b) is NotImplemented  # Direct dunder method call
   
4. AVOID METHOD IDENTITY CHECKS - Bound methods create new objects:
   ✗ WRONG: assert obj.method is original_method  # Often fails due to binding
   ✓ CORRECT: Test the behavior, not object identity

5. IMPORT ALL NEEDED CLASSES - Import Event, Handler, exceptions from the source module.
   Don't create your own versions.

⚠️ CRITICAL - TEST STRUCTURE CONSISTENCY:
You MUST be consistent with your test structure throughout the ENTIRE file:

OPTION A - ALL STANDALONE FUNCTIONS (Recommended for simple modules):
   def test_function_basic():
       assert function(1) == 2
   
   def test_function_edge_case():
       assert function(0) == 0

OPTION B - ALL CLASS-BASED TESTS (Recommended for complex modules):
   class TestFunction:
       def test_basic(self):
           assert function(1) == 2
       
       def test_edge_case(self):
           assert function(0) == 0

⛔ NEVER MIX THESE IN THE SAME FILE:
   ✗ WRONG - This will cause pytest errors:
   class TestSomething:
       def test_a(self):  # Has self - inside class
           pass
   
   def test_b(self):  # ❌ BROKEN: Has self but NOT inside class!
       pass
   
   ✓ CORRECT - Pick ONE style and stick with it:
   def test_a():  # No self - standalone
       pass
   
   def test_b():  # No self - standalone
       pass

IMPORTANT GUIDELINES:
- Generate tests based on what the code SHOULD do (based on names/docs)
- Do NOT assume the current implementation is correct
- The tests you generate will be used to find and fix bugs
- Be thorough but realistic in your test cases

Return ONLY valid Python test code with pytest.
Do NOT include explanations outside of docstrings."""
    
    def execute(
        self,
        plan: RefactoringPlan = None,
        files: List[str] = None,
        force_regeneration: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the test generation process.
        
        Args:
            plan: RefactoringPlan from the Auditor (contains analysis info).
            files: Optional list of specific files to generate tests for.
            force_regeneration: If True, regenerate tests even if they already exist.
            
        Returns:
            Dictionary containing the test generation report.
        """
        self._update_state(status="working", task="Generating test files")
        self._log("Starting test generation...")
        
        try:
            # Determine which files to process
            if files:
                source_files = files
            elif plan:
                source_files = plan.files_analyzed
            else:
                # Discover source files (exclude test files)
                all_files = self.file_tools.list_python_files()
                source_files = [
                    f for f in all_files
                    if not os.path.basename(f).startswith('test_')
                    and not os.path.basename(f).endswith('_test.py')
                ]
            
            if not source_files:
                self._log("No source files found to generate tests for", "WARN")
                return {
                    "success": False,
                    "error": "No source files found",
                    "report": None
                }
            
            self._log(f"Processing {len(source_files)} source file(s)")
            
            results = []
            tests_generated = []
            tests_reused = 0
            total_functions = 0
            
            # Separate files that need generation from those that can reuse tests
            files_to_generate = []
            
            for filepath in source_files:
                test_filepath = self._get_test_filepath(filepath)
                existing_test_result = None
                
                if not force_regeneration:
                    existing_test_result = self._check_existing_test(filepath, test_filepath)
                
                if existing_test_result is not None:
                    # Existing test file is valid - reuse it
                    results.append(existing_test_result)
                    tests_generated.append(existing_test_result.test_file)
                    total_functions += len(existing_test_result.functions_tested)
                    tests_reused += 1
                    self._log(f"  ✅ {filepath}: Reusing existing test file ({len(existing_test_result.functions_tested)} tests)")
                else:
                    # Need to generate tests for this file
                    file_issues = []
                    if plan:
                        file_issues = plan.get_issues_for_file(filepath)
                    files_to_generate.append((filepath, file_issues))
            
            # PARALLEL PROCESSING for files that need generation
            if len(files_to_generate) > 1:
                self._log(f"  🚀 Generating tests for {len(files_to_generate)} files in parallel...")
                
                with ThreadPoolExecutor(max_workers=self.MAX_PARALLEL_FILES) as executor:
                    future_to_file = {
                        executor.submit(self._generate_tests_with_retries, filepath, file_issues): filepath
                        for filepath, file_issues in files_to_generate
                    }
                    
                    for future in as_completed(future_to_file):
                        filepath = future_to_file[future]
                        try:
                            result = future.result()
                            results.append(result)
                            
                            if result.success:
                                tests_generated.append(result.test_file)
                                total_functions += len(result.functions_tested)
                                self._log_safe(f"    ✅ Generated {len(result.functions_tested)} test(s) for {os.path.basename(filepath)}")
                            else:
                                self._log_safe(f"    ❌ Failed {os.path.basename(filepath)}: {result.error_message}")
                        except Exception as e:
                            self._log_safe(f"    ❌ Error on {os.path.basename(filepath)}: {e}")
                            results.append(GeneratedTest(
                                source_file=filepath,
                                test_file="",
                                test_code="",
                                functions_tested=[],
                                success=False,
                                error_message=str(e)
                            ))
            else:
                # Single file - no parallelization
                for filepath, file_issues in files_to_generate:
                    self._log(f"  Generating tests for {filepath}...")
                    result = self._generate_tests_with_retries(filepath, file_issues)
                    results.append(result)
                    
                    if result.success:
                        tests_generated.append(result.test_file)
                        total_functions += len(result.functions_tested)
                        self._log(f"    ✅ Generated {len(result.functions_tested)} test(s)")
                    else:
                        self._log(f"    ❌ Failed: {result.error_message}", "ERROR")
            
            report = TestGenerationReport(
                files_processed=source_files,
                tests_generated=tests_generated,
                total_functions_tested=total_functions,
                successful=sum(1 for r in results if r.success),
                failed=sum(1 for r in results if not r.success),
                tests_reused=tests_reused,
                results=results
            )
            
            self._update_state(status="completed", result=report.to_dict())
            
            # Log summary
            if tests_reused > 0:
                self._log(f"Test generation complete: {report.successful}/{len(source_files)} files ({tests_reused} reused)", "SUCCESS")
            else:
                self._log(f"Test generation complete: {report.successful}/{len(source_files)} files", "SUCCESS")
            
            return {
                "success": True,
                "report": report,
                "tests_generated": tests_generated
            }
            
        except Exception as e:
            self._update_state(status="failed", error=str(e))
            self._log(f"Test generation failed: {e}", "ERROR")
            return {
                "success": False,
                "error": str(e),
                "report": None
            }
    
    def _generate_tests_with_retries(
        self,
        filepath: str,
        file_issues: List[CodeIssue],
        max_retries: int = 3
    ) -> GeneratedTest:
        """
        Generate tests for a file with automatic retries on syntax errors.
        
        Thread-safe method for parallel processing.
        
        Args:
            filepath: Path to source file.
            file_issues: List of code issues for this file.
            max_retries: Maximum retry attempts.
            
        Returns:
            GeneratedTest result.
        """
        result = None
        last_error = None
        
        for attempt in range(max_retries):
            if attempt > 0:
                self._log_safe(f"    🔄 Retry {attempt}/{max_retries - 1} for {os.path.basename(filepath)}...")
            
            result = self._generate_tests_for_file(
                filepath, 
                file_issues,
                previous_error=last_error
            )
            
            if result.success:
                return result
            
            # Check if it's a syntax error - if so, retry
            if result.error_message and "syntax error" in result.error_message.lower():
                last_error = result.error_message
                continue
            else:
                # Non-syntax error, don't retry
                return result
        
        return result if result else GeneratedTest(
            source_file=filepath,
            test_file="",
            test_code="",
            functions_tested=[],
            success=False,
            error_message="Max retries exceeded"
        )
    
    def _get_test_filepath(self, source_filepath: str) -> str:
        """
        Get the expected test file path for a source file.
        
        Args:
            source_filepath: Path to the source file.
            
        Returns:
            Path where the test file should be located.
        """
        dirname = os.path.dirname(source_filepath)
        basename = os.path.basename(source_filepath)
        test_filename = f"test_{basename}"
        return os.path.join(dirname, test_filename) if dirname else test_filename
    
    def _check_existing_test(
        self,
        source_filepath: str,
        test_filepath: str
    ) -> Optional[GeneratedTest]:
        """
        Check if an existing test file is valid and can be reused.
        
        Validation criteria:
        1. Test file exists
        2. Test file has valid Python syntax
        3. Test file contains at least one test function (test_*)
        4. Test file imports from the source module
        
        Args:
            source_filepath: Path to the source file.
            test_filepath: Path to the expected test file.
            
        Returns:
            GeneratedTest if valid existing test found, None otherwise.
        """
        # Check if test file exists
        if not self.file_tools.file_exists(test_filepath):
            return None
        
        try:
            # Read the test file content
            test_code = self.file_tools.read_file(test_filepath)
            
            # Validate syntax by trying to compile
            try:
                compile(test_code, test_filepath, 'exec')
            except SyntaxError as e:
                self._log(f"  ⚠️ Existing test file has syntax error: {e}", "WARN")
                return None
            
            # Check for test functions
            test_functions = re.findall(r'def (test_\w+)\s*\(', test_code)
            if not test_functions:
                self._log(f"  ⚠️ Existing test file has no test functions", "WARN")
                return None
            
            # Check for proper imports from source module
            module_name = os.path.basename(source_filepath).replace('.py', '')
            import_patterns = [
                rf'from\s+{re.escape(module_name)}\s+import',
                rf'import\s+{re.escape(module_name)}',
            ]
            has_import = any(re.search(p, test_code) for p in import_patterns)
            
            if not has_import:
                self._log(f"  ⚠️ Existing test file doesn't import from {module_name}", "WARN")
                return None
            
            # All validations passed - return the existing test
            return GeneratedTest(
                source_file=source_filepath,
                test_file=test_filepath,
                test_code=test_code,
                functions_tested=test_functions,
                success=True,
                was_existing=True
            )
            
        except Exception as e:
            self._log(f"  ⚠️ Error checking existing test: {e}", "WARN")
            return None
    
    def _generate_tests_for_file(
        self,
        filepath: str,
        issues: List[CodeIssue] = None,
        previous_error: str = None
    ) -> GeneratedTest:
        """
        Generate a test file for a single source file.
        
        Args:
            filepath: Path to the source file.
            issues: List of issues identified by Auditor (for context).
            previous_error: Error message from a previous failed attempt (for retry context).
            
        Returns:
            GeneratedTest with the result.
        """
        try:
            # Read source file
            content = self.file_tools.read_file(filepath)
            
            # Extract module name for imports
            basename = os.path.basename(filepath)
            module_name = basename.replace('.py', '')
            
            # Build issue context if available
            issue_context = ""
            if issues:
                issue_hints = []
                for issue in issues[:10]:  # Limit to avoid token overflow
                    issue_hints.append(f"- {issue.description}")
                if issue_hints:
                    issue_context = f"""
KNOWN ISSUES (from static analysis):
The following potential issues were identified. Your tests should verify correct behavior:
{chr(10).join(issue_hints)}
"""
            
            # Build error context if retrying after a syntax error
            error_context = ""
            if previous_error:
                error_context = f"""
IMPORTANT - PREVIOUS ATTEMPT FAILED:
Your previous test generation had a syntax error:
{previous_error}

Please be extra careful to generate syntactically valid Python code.
Common issues to avoid:
- Unmatched parentheses, brackets, or braces
- Missing colons after def/class/if/for/while statements
- Incorrect indentation
- Unclosed string literals
- Invalid escape sequences
"""
            
            # Generate tests using LLM
            test_code = self._generate_test_code(filepath, content, module_name, issue_context + error_context)
            
            # Validate the generated code has valid Python syntax
            try:
                compile(test_code, f"test_{basename}", 'exec')
            except SyntaxError as e:
                return GeneratedTest(
                    source_file=filepath,
                    test_file="",
                    test_code="",
                    functions_tested=[],
                    success=False,
                    error_message=f"Generated tests have syntax error: {e}"
                )
            
            # Check for orphan 'self' parameters (functions with self outside class)
            # This is a critical structural issue that causes "fixture 'self' not found"
            orphan_self_issues = self._check_orphan_self_parameters(test_code)
            if orphan_self_issues:
                # Try to auto-fix by removing 'self' from orphan functions
                test_code = self._fix_orphan_self_parameters(test_code)
                self._log(f"    ⚠️ Fixed {len(orphan_self_issues)} orphan 'self' parameter(s)", "WARN")
            
            # Check for critical anti-patterns that would make tests unreliable
            critical_issues = self._check_critical_antipatterns(test_code, module_name, content)
            if critical_issues:
                return GeneratedTest(
                    source_file=filepath,
                    test_file="",
                    test_code="",
                    functions_tested=[],
                    success=False,
                    error_message=f"Generated tests have critical issues: {'; '.join(critical_issues)}"
                )
            
            # Determine test file path
            dirname = os.path.dirname(filepath)
            test_filename = f"test_{basename}"
            test_filepath = os.path.join(dirname, test_filename) if dirname else test_filename
            
            # Write test file
            self.file_tools.write_file(test_filepath, test_code)
            
            # Validate test collection with pytest - retry up to 2 times on failure
            max_retries = 2
            for attempt in range(max_retries + 1):
                collection_ok, collection_error = self.validate_test_collection(test_filepath)
                if collection_ok:
                    break
                
                if attempt < max_retries:
                    self._log(f"    ⚠️ Test collection failed (attempt {attempt + 1}/{max_retries + 1}), regenerating...", "WARN")
                    # Remove the invalid test file
                    try:
                        abs_test_path = os.path.join(self.sandbox_root, test_filepath)
                        if os.path.exists(abs_test_path):
                            os.remove(abs_test_path)
                    except Exception:
                        pass
                    
                    # Regenerate with explicit instruction to avoid the error
                    retry_context = f"\n\nPREVIOUS ATTEMPT FAILED WITH ERROR:\n{collection_error}\n\nFix this error in your new generation. Common issues:\n- Do NOT use 'self' parameter outside of a class\n- Ensure all test functions are either standalone or properly inside a TestClass"
                    test_code = self._generate_test_code(
                        filepath, content, module_name, 
                        issue_context=issue_context + retry_context
                    )
                    
                    # Apply fixes again
                    test_code = self._fix_orphan_self_parameters(test_code)
                    
                    # Write and try again
                    self.file_tools.write_file(test_filepath, test_code)
                else:
                    # Final attempt failed - give up
                    try:
                        abs_test_path = os.path.join(self.sandbox_root, test_filepath)
                        if os.path.exists(abs_test_path):
                            os.remove(abs_test_path)
                    except Exception:
                        pass
                    
                    return GeneratedTest(
                        source_file=filepath,
                        test_file="",
                        test_code="",
                        functions_tested=[],
                        success=False,
                        error_message=f"Test collection failed after {max_retries + 1} attempts: {collection_error}"
                    )
            
            # Extract function names that were tested
            import re
            test_functions = re.findall(r'def (test_\w+)\s*\(', test_code)
            
            return GeneratedTest(
                source_file=filepath,
                test_file=test_filepath,
                test_code=test_code,
                functions_tested=test_functions,
                success=True
            )
            
        except Exception as e:
            return GeneratedTest(
                source_file=filepath,
                test_file="",
                test_code="",
                functions_tested=[],
                success=False,
                error_message=str(e)
            )
    
    def _generate_test_code(
        self,
        filepath: str,
        content: str,
        module_name: str,
        issue_context: str = ""
    ) -> str:
        """
        Generate test code using the LLM.
        
        Args:
            filepath: Path to the source file.
            content: Source file contents.
            module_name: Module name for imports.
            issue_context: Context about known issues.
            
        Returns:
            Generated test code string.
        """
        prompt = f"""Generate a comprehensive pytest test file for the following Python module.

SOURCE FILE: {filepath}
MODULE NAME: {module_name}

```python
{content}
```
{issue_context}

REQUIREMENTS:
1. Import the module correctly: `from {module_name} import ...`
2. Test EACH public function and class method
3. For each function, include tests for:
   - Normal/typical inputs
   - Edge cases (empty strings, empty lists, zero, None if applicable)
   - Boundary conditions (negative numbers, large values)
   - Error cases (if the function should raise exceptions)
4. Use descriptive test function names: test_<function_name>_<scenario>
5. Add a brief docstring to each test explaining what it verifies
6. Group related tests in classes if there are many functions

⛔ LIBRARY RESTRICTIONS:
- Use ONLY pytest and standard library modules
- Do NOT use: freezegun, mock, unittest.mock, responses, httpretty, fakefs, or any third-party testing libraries
- For datetime testing, use simple comparisons or check that values are reasonable
- For mocking, use pytest's built-in monkeypatch fixture if absolutely needed

INFER EXPECTED BEHAVIOR FROM:
- Function names (e.g., `add` should add numbers, `is_valid` should return bool)
- Docstrings (describe what the function should do)
- Type hints (indicate expected input/output types)
- Parameter names (e.g., `items` suggests a collection)

EXAMPLE TEST STRUCTURE:
```python
\"\"\"Tests for {module_name} module.\"\"\"
import pytest
from {module_name} import function1, ClassName


def test_function1_basic():
    \"\"\"Test function1 with typical input.\"\"\"
    assert function1(5, 3) == 8


def test_function1_zero():
    \"\"\"Test function1 with zero.\"\"\"
    assert function1(0, 0) == 0


def test_function1_negative():
    \"\"\"Test function1 with negative numbers.\"\"\"
    assert function1(-1, -2) == -3


def test_function1_raises_on_invalid():
    \"\"\"Test that function1 raises TypeError for invalid input.\"\"\"
    with pytest.raises(TypeError):
        function1("invalid", 1)


class TestClassName:
    \"\"\"Tests for ClassName.\"\"\"
    
    def test_method_basic(self):
        \"\"\"Test method with typical input.\"\"\"
        obj = ClassName()
        assert obj.method(10) == 20
```

IMPORTANT:
- Use `pytest.raises(ExceptionType)` for testing exceptions
- Test edge cases: empty strings, empty lists, zero, None, negative numbers
- Each test should be independent and focused on one behavior

⛔ ANTI-PATTERNS TO AVOID:
- NEVER put manual validation logic inside `with pytest.raises()` blocks
- NEVER raise exceptions manually inside test code to "simulate" expected behavior
- Tests must ONLY call the function/method being tested, not replicate its logic
- If testing exception handling, just call the function that should raise
- Example of BAD test (DO NOT DO THIS):
  ```python
  with pytest.raises(ValueError):
      if some_condition:  # DON'T replicate logic
          raise ValueError()  # DON'T manually raise
      function_under_test()
  ```
- Example of GOOD test:
  ```python
  with pytest.raises(ValueError):
      function_under_test(invalid_input)  # ONLY call the function
  ```

Return ONLY the complete Python test file code.
Start with the imports, no markdown code blocks."""

        response = self._call_llm(
            prompt=prompt,
            action=ActionType.GENERATION,
            additional_details={"file_tested": filepath, "module": module_name}
        )
        
        if not response.success:
            raise Exception(f"LLM failed to generate tests: {response.error}")
        
        # Extract code from response (remove markdown if present)
        test_code = self._extract_code(response.content)
        
        # Validate test code for anti-patterns
        validation_issues = self._validate_test_code(test_code)
        if validation_issues:
            self._log(f"⚠️ Validation warnings for {module_name}:", "WARN")
            for issue in validation_issues:
                self._log(f"  - {issue}", "WARN")
        
        # Ensure the code ends with a newline
        if not test_code.endswith('\n'):
            test_code += '\n'
        
        return test_code
    
    def _extract_code(self, text: str) -> str:
        """
        Extract Python code from LLM response.
        
        Args:
            text: LLM response text.
            
        Returns:
            Extracted Python code.
        """
        import re
        
        # Try to extract from markdown code block
        patterns = [
            r"```python\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # If no code block, return as-is (hopefully it's just code)
        # Remove any leading text before imports
        lines = text.strip().split('\n')
        
        # Find the first line that looks like Python code
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith(('"""', "'''", '#', 'import ', 'from ', 'def ', 'class ', '@')):
                start_idx = i
                break
        
        return '\n'.join(lines[start_idx:]).strip()

    def _validate_test_code(self, test_code: str) -> List[str]:
        """
        Check for common test anti-patterns.
        
        Args:
            test_code: The generated test code.
            
        Returns:
            List of issue descriptions found.
        """
        import re
        issues = []
        
        # Check for manual raises inside pytest.raises
        # Pattern looks for: with pytest.raises(...): ... raise ...
        if re.search(r'with pytest\.raises.*:\s*\n.*raise\s+\w+', test_code, re.MULTILINE | re.DOTALL):
            issues.append("Manual raise inside pytest.raises block detected (potential anti-pattern)")
        
        # Check for mock class definitions that shadow source module classes
        # This catches cases like defining "class Event:" or "class EventBus:" in tests
        # when those should be imported from source
        class_defs = re.findall(r'^class\s+(\w+)\s*[:\(]', test_code, re.MULTILINE)
        for class_name in class_defs:
            # Skip test classes (classes that start with "Test")
            if class_name.startswith('Test'):
                continue
            # Skip fake/mock classes explicitly named as such
            if any(prefix in class_name for prefix in ['Fake', 'Mock', 'Stub', 'Dummy']):
                continue
            # Warn about other class definitions that might shadow source classes
            issues.append(f"Class '{class_name}' defined in test file - ensure it doesn't shadow source module classes")
        
        # Check for testing Python built-in behavior incorrectly
        # e.g., checking "x < y is NotImplemented" instead of "x.__lt__(y) is NotImplemented"
        if re.search(r'\([^)]+\s*[<>]=?\s*[^)]+\)\s*is\s+NotImplemented', test_code):
            issues.append("Testing comparison result 'is NotImplemented' - should use __lt__/__gt__ directly")
        
        # Check for assertions on internal Python mechanics that are often wrong
        if re.search(r'assert.*\bis\b.*\bemit\b', test_code) or re.search(r'assert.*\bemit\b.*\bis\b', test_code):
            issues.append("Identity check on method reference - may fail due to bound method semantics")
            
        return issues

    def _check_critical_antipatterns(
        self,
        test_code: str,
        module_name: str,
        source_code: str
    ) -> List[str]:
        """
        Check for CRITICAL anti-patterns that make tests unreliable.
        
        Unlike _validate_test_code which just warns, this returns issues
        that should cause test regeneration.
        
        Args:
            test_code: The generated test code.
            module_name: Name of the source module being tested.
            source_code: The source code being tested.
            
        Returns:
            List of critical issues that require regeneration.
        """
        import re
        import ast
        
        critical_issues = []
        
        # Extract class names from source code
        try:
            tree = ast.parse(source_code)
            source_classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        except SyntaxError:
            source_classes = set(re.findall(r'^class\s+(\w+)\s*[:\(]', source_code, re.MULTILINE))
        
        # Check if test file redefines any source classes (CRITICAL - causes shadowing)
        test_class_defs = re.findall(r'^class\s+(\w+)\s*[:\(]', test_code, re.MULTILINE)
        for class_name in test_class_defs:
            if class_name.startswith('Test'):
                continue
            if class_name in source_classes:
                critical_issues.append(
                    f"Test redefines source class '{class_name}' - this shadows the real class and breaks tests. "
                    f"Import from {module_name} instead."
                )
        
        # Check if test imports from the source module
        import_patterns = [
            rf'from\s+{re.escape(module_name)}\s+import',
            rf'import\s+{re.escape(module_name)}',
        ]
        has_import = any(re.search(p, test_code) for p in import_patterns)
        
        if not has_import and source_classes:
            critical_issues.append(
                f"Test file doesn't import from '{module_name}' - tests must import classes/functions from the source module."
            )
        
        return critical_issues

    def _check_orphan_self_parameters(self, test_code: str) -> List[Dict[str, Any]]:
        """
        Check for functions with 'self' parameter that are NOT inside a class.
        
        This catches the common LLM mistake of mixing class-based and standalone tests,
        which causes "fixture 'self' not found" errors in pytest.
        
        Args:
            test_code: The generated test code.
            
        Returns:
            List of dictionaries with info about orphan self parameters.
        """
        import ast
        
        orphan_functions = []
        
        try:
            tree = ast.parse(test_code)
            
            # Get all class names to track which functions are inside classes
            class_ranges = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Record the line range of this class
                    end_line = max(
                        getattr(child, 'end_lineno', node.lineno) 
                        for child in ast.walk(node)
                    )
                    class_ranges.append((node.lineno, end_line))
            
            # Check all function definitions
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Check if this function is inside any class
                    inside_class = any(
                        start <= node.lineno <= end 
                        for start, end in class_ranges
                    )
                    
                    # Check if first parameter is 'self'
                    if node.args.args and node.args.args[0].arg == 'self':
                        if not inside_class:
                            orphan_functions.append({
                                'name': node.name,
                                'lineno': node.lineno,
                                'col_offset': node.col_offset
                            })
        except SyntaxError:
            # If parsing fails, use regex fallback
            import re
            lines = test_code.split('\n')
            inside_class = False
            class_indent = 0
            
            for i, line in enumerate(lines, 1):
                # Track class definitions
                class_match = re.match(r'^(\s*)class\s+\w+', line)
                if class_match:
                    inside_class = True
                    class_indent = len(class_match.group(1))
                    continue
                
                # Check if we've left the class (based on indentation)
                if inside_class and line.strip() and not line.startswith(' ' * (class_indent + 1)):
                    if not line.strip().startswith('#'):
                        inside_class = False
                
                # Check for function with self parameter
                func_match = re.match(r'^(\s*)def\s+(test_\w+)\s*\(\s*self\s*[,)]', line)
                if func_match and not inside_class:
                    orphan_functions.append({
                        'name': func_match.group(2),
                        'lineno': i,
                        'col_offset': len(func_match.group(1))
                    })
        
        return orphan_functions

    def _fix_orphan_self_parameters(self, test_code: str) -> str:
        """
        Fix orphan 'self' parameters by removing them from standalone functions.
        
        Args:
            test_code: The test code with orphan self parameters.
            
        Returns:
            Fixed test code.
        """
        import re
        
        # Get orphan functions
        orphans = self._check_orphan_self_parameters(test_code)
        if not orphans:
            return test_code
        
        lines = test_code.split('\n')
        orphan_lines = {o['lineno'] for o in orphans}
        
        fixed_lines = []
        for i, line in enumerate(lines, 1):
            if i in orphan_lines:
                # Remove 'self' parameter
                # Handle: def test_something(self): -> def test_something():
                # Handle: def test_something(self, arg): -> def test_something(arg):
                line = re.sub(r'\(\s*self\s*,\s*', '(', line)
                line = re.sub(r'\(\s*self\s*\)', '()', line)
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)

    def validate_test_collection(self, test_filepath: str) -> Tuple[bool, str]:
        """
        Validate that pytest can collect the test file without errors.
        
        This catches structural issues like 'fixture self not found' before
        they cause problems in the Judge.
        
        Args:
            test_filepath: Path to the test file.
            
        Returns:
            Tuple of (success, error_message).
        """
        import subprocess
        import sys
        
        try:
            # Run pytest --collect-only to validate test collection
            abs_path = os.path.join(self.sandbox_root, test_filepath)
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', '--collect-only', '-q', abs_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.sandbox_root
            )
            
            # Check for collection errors - use returncode as primary signal
            # Note: Don't check for 'error' in stdout since test names often contain 'error'
            # (e.g., test_raises_value_error) which would cause false positives
            if result.returncode != 0:
                error_msg = result.stderr + result.stdout
                # Extract relevant error message
                if "fixture 'self' not found" in error_msg:
                    return False, "Functions have 'self' parameter but are not inside a class"
                if "ModuleNotFoundError" in error_msg or "ImportError" in error_msg:
                    return False, f"Import error: {error_msg[:500]}"
                if "SyntaxError" in error_msg:
                    return False, f"Syntax error: {error_msg[:500]}"
                return False, f"pytest collection failed: {error_msg[:500]}"
            
            return True, ""
            
        except subprocess.TimeoutExpired:
            return False, "Test collection timed out"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
