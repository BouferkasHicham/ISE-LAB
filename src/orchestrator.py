"""
Orchestrator - Multi-Agent Workflow Controller.

The Orchestrator manages the execution flow between Auditor, TestGenerator, 
Fixer, and Judge agents, implementing the self-healing loop for iterative 
code improvement.
"""

import os
from typing import Dict, Any, Optional, List, TypedDict, Annotated
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.agents.auditor import AuditorAgent, RefactoringPlan
from src.agents.test_generator import TestGeneratorAgent, TestGenerationReport
from src.agents.fixer import FixerAgent, FixerReport
from src.agents.judge import JudgeAgent, Judgment, JudgmentResult
from src.tools.file_tools import FileTools
from src.utils.logger import log_experiment, ActionType
from src.utils.output_formatter import OutputFormatter, get_formatter, Colors, Icons
from src.config.settings import get_settings


class WorkflowState(Enum):
    """States in the refactoring workflow."""
    START = "START"
    AUDITING = "AUDITING"
    GENERATING_TESTS = "GENERATING_TESTS"
    FIXING = "FIXING"
    JUDGING = "JUDGING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class AgentState(TypedDict):
    """State shared between agents in the workflow."""
    target_dir: str
    iteration: int
    max_iterations: int
    workflow_state: str
    
    # Audit results
    refactoring_plan: Optional[Dict[str, Any]]
    files_analyzed: List[str]
    initial_scores: Dict[str, float]
    
    # Test generation results
    test_generation_report: Optional[Dict[str, Any]]
    tests_generated: List[str]
    
    # Fix results
    fixer_report: Optional[Dict[str, Any]]
    files_fixed: List[str]
    
    # Judge results
    judgment: Optional[Dict[str, Any]]
    tests_passed: bool
    current_scores: Dict[str, float]
    
    # Error tracking
    errors: List[str]
    
    # History for analysis
    history: List[Dict[str, Any]]
    
    # Test history tracking for oscillation detection
    # Maps test name -> list of outcomes across iterations (True=passed, False=failed)
    test_history: Dict[str, List[bool]]
    # Tests that passed in the previous iteration (to warn LLM not to break them)
    previously_passing_tests: List[str]
    
    # Track persistent failures for self-healing (file -> consecutive failure count)
    persistent_failures: Dict[str, int]


@dataclass
class OrchestrationResult:
    """Final result of the orchestration process."""
    success: bool
    iterations_used: int
    initial_scores: Dict[str, float]
    final_scores: Dict[str, float]
    tests_passed: bool
    files_processed: List[str]
    total_issues_fixed: int
    improvement: float
    error_message: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "iterations_used": self.iterations_used,
            "initial_scores": self.initial_scores,
            "final_scores": self.final_scores,
            "tests_passed": self.tests_passed,
            "files_processed": self.files_processed,
            "total_issues_fixed": self.total_issues_fixed,
            "improvement": self.improvement,
            "error_message": self.error_message
        }
    
    def get_summary(self) -> str:
        """Get a formatted summary."""
        status = "✅ SUCCESS" if self.success else "❌ FAILED"
        
        initial_avg = sum(self.initial_scores.values()) / len(self.initial_scores) if self.initial_scores else 0
        final_avg = sum(self.final_scores.values()) / len(self.final_scores) if self.final_scores else 0
        
        return f"""
{'═' * 60}
🎯 REFACTORING SWARM - MISSION REPORT
{'═' * 60}

Status: {status}
Iterations: {self.iterations_used}
Tests: {'✅ Passed' if self.tests_passed else '❌ Failed'}

📊 Quality Improvement:
   Initial Average Score: {initial_avg:.2f}/10
   Final Average Score:   {final_avg:.2f}/10
   Improvement:           {self.improvement:+.2f}

📁 Files Processed: {len(self.files_processed)}
🔧 Issues Fixed: {self.total_issues_fixed}

{'═' * 60}
"""


class Orchestrator:
    """
    The Orchestrator coordinates the multi-agent refactoring workflow.
    
    It implements a state machine that:
    1. Runs the Auditor to analyze code and create a refactoring plan
    2. Runs the TestGenerator to create pytest test files
    3. Runs the Fixer to apply fixes based on the plan
    4. Runs the Judge to validate fixes through testing
    5. Loops back to Fixer if tests fail (self-healing loop)
    6. Exits when tests pass or max iterations reached
    """
    
    def __init__(
        self,
        target_dir: str,
        max_iterations: int = 5,
        pylint_threshold: float = 8.5,
        verbose: bool = True
    ):
        """
        Initialize the Orchestrator.
        
        Args:
            target_dir: Directory containing code to refactor.
            max_iterations: Maximum self-healing loop iterations.
            pylint_threshold: Minimum acceptable Pylint score.
            verbose: Enable verbose output.
        """
        self.target_dir = os.path.abspath(target_dir)
        self.max_iterations = max_iterations
        self.pylint_threshold = pylint_threshold
        self.verbose = verbose
        
        # Initialize output formatter
        self.formatter = get_formatter(verbose=verbose)
        
        # Initialize agents
        self.auditor = AuditorAgent(
            sandbox_root=self.target_dir,
            verbose=verbose
        )
        self.test_generator = TestGeneratorAgent(
            sandbox_root=self.target_dir,
            verbose=verbose
        )
        self.fixer = FixerAgent(
            sandbox_root=self.target_dir,
            verbose=verbose
        )
        self.judge = JudgeAgent(
            sandbox_root=self.target_dir,
            pylint_threshold=pylint_threshold,
            verbose=verbose
        )
        
        self.file_tools = FileTools(self.target_dir)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow for agent orchestration.
        
        Workflow: Auditor → CosmeticFixer → TestGenerator → Fixer → Judge → (loop to Fixer or end)
        
        The CosmeticFixer runs BEFORE test generation to rename functions/variables
        to PEP 8 style, so tests are generated with clean names.
        
        Returns:
            Configured StateGraph.
        """
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("auditor", self._run_auditor)
        workflow.add_node("cosmetic_fixer", self._run_cosmetic_fixer)
        workflow.add_node("test_generator", self._run_test_generator)
        workflow.add_node("fixer", self._run_fixer)
        workflow.add_node("judge", self._run_judge)
        
        # Set entry point
        workflow.set_entry_point("auditor")
        
        # Auditor → CosmeticFixer (rename identifiers first)
        workflow.add_edge("auditor", "cosmetic_fixer")
        
        # CosmeticFixer → TestGenerator (generate tests with clean names)
        workflow.add_edge("cosmetic_fixer", "test_generator")
        
        # TestGenerator always goes to Fixer
        workflow.add_edge("test_generator", "fixer")
        
        # Fixer always goes to Judge
        workflow.add_edge("fixer", "judge")
        
        # Conditional edge from judge (self-healing loop)
        # Note: Loop goes back to Fixer, NOT to TestGenerator
        # Tests are generated ONCE at the beginning
        workflow.add_conditional_edges(
            "judge",
            self._should_continue,
            {
                "continue": "fixer",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _run_cosmetic_fixer(self, state: AgentState) -> Dict[str, Any]:
        """
        Run cosmetic fixes BEFORE test generation.
        
        This renames functions, classes, variables to PEP 8 style
        WITHOUT changing any logic. This ensures tests are generated
        with clean names, avoiding the "locked-in bad names" problem.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated state dictionary.
        """
        self.formatter.print_phase_start("COSMETIC FIXES", "Renaming identifiers to PEP 8 style...")
        self.formatter.print_agent_action("fixer", "Applying cosmetic-only fixes (names, imports, formatting)")
        
        files_analyzed = state.get("files_analyzed", [])
        
        if not files_analyzed:
            self.formatter.print_status("warning", "No files to apply cosmetic fixes to")
            return {}
        
        # Run cosmetic fixes
        result = self.fixer.execute_cosmetic(files_analyzed)
        
        if result.get("success"):
            files_fixed = result.get("files_fixed", [])
            self.formatter.print_agent_action(
                "fixer",
                f"Cosmetic fixes applied to {len(files_fixed)}/{len(files_analyzed)} files"
            )
            for f in files_fixed:
                self.formatter.print_status("success", f"Renamed: {os.path.basename(f)}", indent=4)
        else:
            self.formatter.print_status("warning", "Cosmetic fixes partially failed")
        
        return {
            "history": state.get("history", []) + [{
                "agent": "CosmeticFixer",
                "iteration": 0,
                "timestamp": datetime.now().isoformat(),
                "files_fixed": result.get("files_fixed", [])
            }]
        }
    
    def _run_auditor(self, state: AgentState) -> Dict[str, Any]:
        """
        Run the Auditor agent.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated state dictionary.
        """
        self.formatter.print_phase_start("CODE ANALYSIS", "Scanning for bugs and style issues...")
        self.formatter.print_agent_action("auditor", "Starting comprehensive code analysis")
        
        result = self.auditor.execute()
        
        if not result["success"]:
            self.formatter.print_status("error", f"Auditor failed: {result.get('error')}")
            return {
                "workflow_state": WorkflowState.FAILED.value,
                "errors": state.get("errors", []) + [f"Auditor failed: {result.get('error')}"]
            }
        
        plan: RefactoringPlan = result["plan"]
        fix_type = result.get("fix_type", plan.fix_type if hasattr(plan, 'fix_type') else "BUGS")
        
        # Display findings
        self.formatter.print_agent_action(
            "auditor", 
            f"Analysis complete: Found {plan.total_issues} issues in {len(plan.files_analyzed)} files"
        )
        
        # Show file scores
        if plan.initial_scores:
            files_info = [
                {"name": os.path.basename(f), "score": s, "status": "analyzing"}
                for f, s in plan.initial_scores.items()
            ]
            self.formatter.print_file_table(files_info, "Initial Pylint Scores", threshold=self.pylint_threshold)
        
        return {
            "workflow_state": WorkflowState.GENERATING_TESTS.value,
            "refactoring_plan": plan.to_dict(),
            "files_analyzed": plan.files_analyzed,
            "initial_scores": plan.initial_scores,
            "history": state.get("history", []) + [{
                "agent": "Auditor",
                "iteration": state.get("iteration", 0),
                "timestamp": datetime.now().isoformat(),
                "issues_found": plan.total_issues,
                "files": plan.files_analyzed
            }]
        }
    
    def _run_test_generator(self, state: AgentState) -> Dict[str, Any]:
        """
        Run the TestGenerator agent.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated state dictionary.
        """
        self.formatter.print_phase_start("TEST GENERATION", "Creating pytest test files...")
        self.formatter.print_agent_action("test_generator", "Generating tests based on code analysis")
        
        # Get the refactoring plan from state
        plan_dict = state.get("refactoring_plan")
        plan = None
        if plan_dict:
            # Reconstruct RefactoringPlan from dict
            from src.agents.auditor import RefactoringPlan, CodeIssue
            issues = []
            for i in plan_dict.get("issues", []):
                issues.append(CodeIssue(
                    file_path=i["file_path"],
                    line_number=i["line_number"],
                    issue_type=i["issue_type"],
                    severity=i["severity"],
                    description=i["description"],
                    suggested_fix=i["suggested_fix"],
                    pylint_symbol=i.get("pylint_symbol")
                ))
            
            plan = RefactoringPlan(
                files_analyzed=plan_dict.get("files_analyzed", []),
                total_issues=plan_dict.get("total_issues", 0),
                issues=issues,
                priority_order=plan_dict.get("priority_order", []),
                initial_scores=plan_dict.get("initial_scores", {}),
                estimated_complexity=plan_dict.get("estimated_complexity", "medium"),
                summary=plan_dict.get("summary", ""),
                files_needing_style=plan_dict.get("files_needing_style", [])
            )
        
        result = self.test_generator.execute(plan=plan)
        
        if not result["success"]:
            self.formatter.print_status("error", f"TestGenerator failed: {result.get('error')}")
            return {
                "workflow_state": WorkflowState.FAILED.value,
                "errors": state.get("errors", []) + [f"TestGenerator failed: {result.get('error')}"]
            }
        
        report: TestGenerationReport = result["report"]
        tests_generated = result.get("tests_generated", [])
        
        # Track files that failed test generation (need to retry later)
        files_without_tests = []
        for gen_result in report.results:
            if not gen_result.success:
                files_without_tests.append(gen_result.source_file)
        
        if files_without_tests:
            self.formatter.print_status(
                "warning", 
                f"⚠️ {len(files_without_tests)} file(s) have no tests (will retry each iteration):"
            )
            for f in files_without_tests:
                self.formatter.print_status("warning", f"  • {os.path.basename(f)}", indent=4)
        
        self.formatter.print_agent_action(
            "test_generator",
            f"Generated {len(tests_generated)} test file(s) with {report.total_functions_tested} test functions"
        )
        
        # Show generated tests
        for test_file in tests_generated:
            self.formatter.print_status("success", f"Created: {test_file}", indent=4)
        
        return {
            "workflow_state": WorkflowState.FIXING.value,
            "test_generation_report": report.to_dict(),
            "tests_generated": tests_generated,
            "files_without_tests": files_without_tests,
            "history": state.get("history", []) + [{
                "agent": "TestGenerator",
                "iteration": state.get("iteration", 0),
                "timestamp": datetime.now().isoformat(),
                "tests_generated": tests_generated,
                "total_test_functions": report.total_functions_tested
            }]
        }
    
    def _run_fixer(self, state: AgentState) -> Dict[str, Any]:
        """
        Run the Fixer agent.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated state dictionary.
        """
        iteration = state.get("iteration", 0) + 1
        max_iter = state.get("max_iterations", 10)
        
        # Initialize persistent failures from state
        new_persistent_failures = state.get("persistent_failures", {}).copy()
        
        # Track files without tests (need retry)
        files_without_tests = list(state.get("files_without_tests", []))
        
        self.formatter.print_iteration_header(iteration, max_iter)
        self.formatter.print_phase_start("APPLYING FIXES", f"Iteration {iteration}/{max_iter}")
        self.formatter.print_agent_action("fixer", "Analyzing issues and applying fixes...")
        
        # Get the refactoring plan
        plan_dict = state.get("refactoring_plan")
        
        # Check for judgment first (self-healing loop)
        judgment_dict = state.get("judgment")
        if judgment_dict and not judgment_dict.get("passed", True):
            # Check if tests passed but only style fixes needed
            test_results = judgment_dict.get("test_results", {})
            tests_passed = test_results.get("all_passed", False) if test_results else False
            files_to_fix = judgment_dict.get("files_to_fix", [])
            feedback = judgment_dict.get("feedback", "")
            
            if tests_passed and files_to_fix:
                # STYLE_ONLY mode: Tests pass but Pylint is low
                self.formatter.print_status("info", f"Tests pass but {len(files_to_fix)} files need style fixes")
                self.formatter.print_agent_action("fixer", "Applying style improvements (STYLE_ONLY mode)")
                
                # Build a style-only plan for these files
                from src.agents.auditor import RefactoringPlan, CodeIssue
                
                # Get Pylint scores from judgment
                pylint_scores = judgment_dict.get("pylint_scores", {})
                
                # Create style issues for each file
                style_issues = []
                for filepath in files_to_fix:
                    score = pylint_scores.get(filepath, 0)
                    style_issues.append(CodeIssue(
                        file_path=filepath,
                        line_number=1,
                        issue_type="style",
                        severity="medium",
                        description=f"Pylint score {score:.2f} is below threshold",
                        suggested_fix="Improve code style, add docstrings, type hints"
                    ))
                
                style_plan = RefactoringPlan(
                    files_analyzed=files_to_fix,
                    total_issues=len(style_issues),
                    issues=style_issues,
                    priority_order=files_to_fix,
                    initial_scores=pylint_scores,
                    estimated_complexity="low",
                    summary="Style-only fixes needed after bug fixes",
                    fix_type="STYLE_ONLY",
                    files_needing_style=files_to_fix
                )
                
                result = self.fixer.execute(plan=style_plan, fix_type="STYLE_ONLY")
            else:
                # BUGS mode: Fix based on test failures (Test-Driven Repair)
                # ONLY fix files that actually have failing tests
                files_with_test_failures = judgment_dict.get("files_with_test_failures", [])
                
                # --- TEST-DRIVEN REPAIR: Retry test generation for files without tests ---
                files_without_tests = list(state.get("files_without_tests", []))
                if files_without_tests:
                    self.formatter.print_status(
                        "warning", 
                        f"🔄 {len(files_without_tests)} file(s) have no tests - retrying generation..."
                    )
                    
                    # Retry test generation for these files
                    regen_result = self.test_generator.execute(files=files_without_tests, force_regeneration=True)
                    
                    if regen_result.get("success"):
                        regen_report = regen_result.get("report")
                        newly_generated = []
                        still_without_tests = []
                        
                        for gen_result in regen_report.results:
                            if gen_result.success:
                                newly_generated.append(gen_result.source_file)
                                self.formatter.print_status("success", f"  ✅ Generated tests for {os.path.basename(gen_result.source_file)}")
                            else:
                                still_without_tests.append(gen_result.source_file)
                                self.formatter.print_status("warning", f"  ⚠️ Still no tests for {os.path.basename(gen_result.source_file)}")
                        
                        # Update files_without_tests
                        files_without_tests = still_without_tests
                
                # Block files without tests from BUGS fixing (Test-Driven Repair requires tests!)
                if files_without_tests:
                    self.formatter.print_status(
                        "info", 
                        f"⏭️ Skipping {len(files_without_tests)} file(s) in BUGS mode (no tests to guide fixes)"
                    )
                    # Remove files without tests from files_with_test_failures
                    files_with_test_failures = [f for f in files_with_test_failures if f not in files_without_tests]
                
                # --- SELF-HEALING: Check for persistent failures ---
                # Key: filename, Value: consecutive failures
                persistent_failures = state.get("persistent_failures", {})
                
                # Update failure counts
                new_persistent_failures = persistent_failures.copy()
                files_to_regenerate = []
                
                for filepath in files_with_test_failures:
                    count = persistent_failures.get(filepath, 0) + 1
                    new_persistent_failures[filepath] = count
                    
                    # If failed 3 times in a row, regenerate tests instead of fixing
                    if count >= 3:
                        files_to_regenerate.append(filepath)
                
                # Reset count for files that passed (aren't in files_with_test_failures)
                # We need all analyzed files to adhere to "files_analyzed"
                all_files = state.get("files_analyzed", [])
                for filepath in all_files:
                    if filepath not in files_with_test_failures:
                        if filepath in new_persistent_failures:
                            del new_persistent_failures[filepath]
                            
                # Update state with new counts
                # We can't update state directly here as return value is what matters for next node
                # But we can pass it along? LangGraph merges returns into state.
                
                # Handle test regeneration
                if files_to_regenerate:
                    self.formatter.print_status("warning", f"Files failing persistently: {', '.join(os.path.basename(f) for f in files_to_regenerate)}")
                    self.formatter.print_agent_action("test_generator", "Regenerating tests due to persistent failures")
                    
                    # Regenerate tests for these files
                    regen_report = self.test_generator.execute(files=files_to_regenerate, force_regeneration=True)
                    
                    # Reset failure count for these files
                    for f in files_to_regenerate:
                        new_persistent_failures[f] = 0
                        
                    return {
                        "workflow_state": WorkflowState.FIXING.value,
                        "persistent_failures": new_persistent_failures,
                        # Don't update other things, basically skip fixing this round to let tests refresh
                        "history": state.get("history", []) + [{
                            "agent": "Orchestrator",
                            "action": "REGENERATE_TESTS",
                            "timestamp": datetime.now().isoformat(),
                            "files": files_to_regenerate
                        }]
                    }

                # Do NOT fall back to all files - only fix files with actual test failures
                if not files_with_test_failures:
                    self.formatter.print_status("warning", "No specific files identified with test failures")
                    # Try to extract from feedback as last resort
                    feedback = judgment_dict.get("feedback", "")
                    test_results_dict = judgment_dict.get("test_results", {})
                    raw_output = test_results_dict.get("raw_output", "") if test_results_dict else ""
                    
                    # Check for collection errors (ImportError, SyntaxError, etc.)
                    # These often indicate which file has the problem
                    import re
                    for f in files_to_fix:
                        basename = os.path.basename(f)
                        # Check if file is mentioned in feedback or raw output
                        if basename in feedback or basename in raw_output:
                            files_with_test_failures.append(f)
                        # Also check for import errors mentioning the module
                        module_name = basename.replace('.py', '')
                        if f'from {module_name}' in raw_output or f'import {module_name}' in raw_output:
                            if f not in files_with_test_failures:
                                files_with_test_failures.append(f)
                
                if files_with_test_failures:
                    self.formatter.print_status("warning", f"Tests failed - fixing {len(files_with_test_failures)} file(s): {', '.join(os.path.basename(f) for f in files_with_test_failures)}")
                else:
                    self.formatter.print_status("error", "Cannot identify which files need fixing")
                    return {
                        "workflow_state": WorkflowState.JUDGING.value,
                        "iteration": iteration
                    }
                self.formatter.print_agent_action("fixer", "Applying bug fixes based on test failures")
                
                # Get raw pytest output - much better than parsed feedback!
                # The LLM can understand the full context directly
                test_results_dict = judgment_dict.get("test_results", {})
                raw_test_output = test_results_dict.get("raw_output", "") if test_results_dict else ""
                
                # Fall back to feedback if raw output not available
                if not raw_test_output:
                    raw_test_output = judgment_dict.get("feedback", "")
                
                fix_results = []
                files_fixed = []
                
                # Get previously passing tests to avoid breaking them
                previously_passing_tests = state.get("previously_passing_tests", [])
                
                # Track test results before fixes for regression detection
                prev_test_results = judgment_dict.get("test_results", {})
                prev_failed_count = prev_test_results.get("failed", 0) if prev_test_results else 0
                prev_error_count = prev_test_results.get("errors", 0) if prev_test_results else 0
                
                for filepath in files_with_test_failures:
                    filename = os.path.basename(filepath)
                    fix_result = self.fixer.fix_based_on_test_failure(
                        filepath=filepath,
                        test_output=raw_test_output,
                        traceback="",  # Raw output already contains tracebacks
                        previously_passing_tests=previously_passing_tests
                    )
                    
                    fix_results.append(fix_result)
                    if fix_result.success:
                        # Quick regression check: run tests to see if we made things worse
                        quick_test = self.judge.test_tools.run_tests(verbose=False)
                        new_failed_count = quick_test.failed if quick_test else 0
                        new_error_count = quick_test.errors if quick_test else 0
                        
                        # Check for regression, but account for collection errors:
                        # - If previous state had collection errors (imports broken, etc.)
                        #   and the fix resolves them, that's an improvement even if
                        #   we now see actual test failures
                        # - Only consider it a regression if errors didn't decrease AND failures increased
                        errors_improved = new_error_count < prev_error_count
                        failures_increased = new_failed_count > prev_failed_count
                        
                        # It's a regression only if failures increased WITHOUT fixing errors
                        is_regression = failures_increased and not errors_improved
                        
                        if is_regression:
                            # Regression detected! Roll back this file
                            self.formatter.print_status(
                                "warning", 
                                f"⚠️ Regression: {filename} fix increased failures ({prev_failed_count} → {new_failed_count}), rolling back",
                                indent=4
                            )
                            self.fixer.file_tools.restore_backup(filepath)
                            fix_result.success = False
                            fix_result.error_message = f"Rolled back: caused regression ({prev_failed_count} → {new_failed_count} failures)"
                        else:
                            files_fixed.append(filepath)
                            if errors_improved:
                                self.formatter.print_status("success", f"Fixed {filename} (resolved {prev_error_count - new_error_count} collection error(s))", indent=4)
                            else:
                                self.formatter.print_status("success", f"Fixed {filename}", indent=4)
                            # Update baseline for next file
                            prev_failed_count = new_failed_count
                            prev_error_count = new_error_count
                    else:
                        self.formatter.print_status("error", f"Failed to fix {filename}", indent=4)
                
                # Create a synthetic report for consistency
                report = FixerReport(
                    files_fixed=files_fixed,
                    total_fixes=len(files_to_fix),
                    successful_fixes=len(files_fixed),
                    failed_fixes=len(files_to_fix) - len(files_fixed),
                    fix_results=fix_results,
                    overall_improvement=0.0
                )
                
                result = {
                    "success": True, 
                    "report": report,
                    "files_fixed": files_fixed
                }
            
        elif plan_dict and iteration <= 1:
            # First pass ONLY: Fix based on Auditor plan
            # Subsequent iterations should use test-failure based fixing
            # Reconstruct RefactoringPlan from dict
            from src.agents.auditor import RefactoringPlan, CodeIssue
            
            # --- TEST-DRIVEN REPAIR: Skip files without tests ---
            if files_without_tests:
                self.formatter.print_status(
                    "warning", 
                    f"⏭️ {len(files_without_tests)} file(s) skipped (no tests): {', '.join(os.path.basename(f) for f in files_without_tests)}"
                )
            
            issues = []
            for i in plan_dict.get("issues", []):
                # Skip issues for files without tests
                if i["file_path"] in files_without_tests:
                    continue
                issues.append(CodeIssue(
                    file_path=i["file_path"],
                    line_number=i["line_number"],
                    issue_type=i["issue_type"],
                    severity=i["severity"],
                    description=i["description"],
                    suggested_fix=i["suggested_fix"],
                    pylint_symbol=i.get("pylint_symbol")
                ))
            
            # Filter files_analyzed to exclude files without tests
            files_analyzed = [f for f in plan_dict.get("files_analyzed", []) if f not in files_without_tests]
            
            plan = RefactoringPlan(
                files_analyzed=files_analyzed,
                total_issues=len(issues),
                issues=issues,
                priority_order=[f for f in plan_dict.get("priority_order", []) if f not in files_without_tests],
                initial_scores=plan_dict.get("initial_scores", {}),
                estimated_complexity=plan_dict.get("estimated_complexity", "medium"),
                summary=plan_dict.get("summary", ""),
                fix_type=plan_dict.get("fix_type", "BUGS"),
                files_needing_style=plan_dict.get("files_needing_style", [])
            )
            
            # Pass fix_type to fixer
            fix_type = plan_dict.get("fix_type", "BUGS")
            result = self.fixer.execute(plan=plan, fix_type=fix_type)
        else:
            # Subsequent iterations: Fix based on previous judgment or re-analyze
            judgment_dict = state.get("judgment")
            if judgment_dict and (judgment_dict.get("files_with_test_failures") or judgment_dict.get("files_to_fix")):
                # ONLY fix files with test failures - do not fall back to all files
                files_with_test_failures = judgment_dict.get("files_with_test_failures", [])
                files_to_fix = judgment_dict.get("files_to_fix", [])
                feedback = judgment_dict.get("feedback", "")
                
                # Prioritize test failures, but DON'T fall back to all files
                target_files = files_with_test_failures
                
                # Only use files_to_fix if there are NO test failures (style-only mode)
                if not target_files and files_to_fix:
                    # This should only happen if tests pass but pylint needs fixes
                    target_files = files_to_fix
                    self._log(f"🔄 Style fixes for {len(target_files)} files...")
                elif target_files:
                    self._log(f"🔄 Fixing {len(target_files)} file(s) with test failures: {', '.join(os.path.basename(f) for f in target_files)}")
                
                fix_results = []
                files_fixed = []
                
                for filepath in target_files:
                    fix_result = self.fixer.fix_based_on_test_failure(
                        filepath=filepath,
                        test_output=feedback,
                        traceback=""  # Feedback already contains tracebacks
                    )
                    fix_results.append(fix_result)
                    if fix_result.success:
                        files_fixed.append(filepath)
                
                # Create report
                report = FixerReport(
                    files_fixed=files_fixed,
                    total_fixes=len(target_files),
                    successful_fixes=len(files_fixed),
                    failed_fixes=len(target_files) - len(files_fixed),
                    fix_results=fix_results,
                    overall_improvement=0.0
                )
                
                result = {"success": True, "report": report, "files_fixed": files_fixed}
            else:
                result = {"success": False, "error": "No judgment feedback available for iteration"}
        
        if not result.get("success"):
            self.formatter.print_status("warning", f"Fixer warning: {result.get('error', 'Unknown')}")
            return {
                "workflow_state": WorkflowState.JUDGING.value,
                "iteration": iteration,
                "errors": state.get("errors", []) + [f"Fixer warning: {result.get('error', 'Unknown')}"],
                "history": state.get("history", []) + [{
                    "agent": "Fixer",
                    "iteration": iteration,
                    "timestamp": datetime.now().isoformat(),
                    "status": "warning",
                    "message": result.get("error", "Unknown error")
                }]
            }
        
        report = result.get("report")
        files_fixed = result.get("files_fixed", [])
        
        self.formatter.print_status("success", f"Fixed {len(files_fixed)} file(s)")
        
        # Show fixed files
        if files_fixed:
            for f in files_fixed:
                self.formatter.print_status("success", os.path.basename(f), indent=4)
        
        return {
            "workflow_state": WorkflowState.JUDGING.value,
            "iteration": iteration,
            "fixer_report": report.to_dict() if report else None,
            "files_fixed": files_fixed,
            "files_without_tests": files_without_tests,
            "persistent_failures": new_persistent_failures,
            "history": state.get("history", []) + [{
                "agent": "Fixer",
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "files_fixed": files_fixed,
                "improvement": result.get("improvement", 0)
            }]
        }
    
    def _run_judge(self, state: AgentState) -> Dict[str, Any]:
        """
        Run the Judge agent.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated state dictionary.
        """
        self.formatter.print_phase_start("VALIDATION", "Running tests and quality checks...")
        self.formatter.print_agent_action("judge", "Executing tests and analyzing code quality")
        
        result = self.judge.execute()
        
        if not result["success"]:
            self.formatter.print_status("error", f"Judge failed: {result.get('error')}")
            return {
                "workflow_state": WorkflowState.FAILED.value,
                "errors": state.get("errors", []) + [f"Judge failed: {result.get('error')}"]
            }
        
        judgment: Judgment = result["judgment"]
        
        # Show test results
        if judgment.test_results:
            # Get failed test names with their source files
            failed_tests = judgment.test_results.get_failed_tests() if hasattr(judgment.test_results, 'get_failed_tests') else []
            
            # Build failure display with source file mapping
            failure_display = []
            for t in failed_tests:  # Show all failures
                test_file = os.path.basename(t.file_path) if t.file_path else "unknown"
                # Map test file to source file
                if test_file.startswith("test_"):
                    source_file = test_file[5:]  # Remove "test_" prefix
                elif test_file.endswith("_test.py"):
                    source_file = test_file.replace("_test.py", ".py")
                else:
                    source_file = test_file
                failure_display.append(f"{t.name} → {source_file}")
            
            self.formatter.print_test_results(
                passed=judgment.test_results.passed,
                failed=judgment.test_results.failed,
                total=judgment.test_results.total,
                failures=failure_display if failure_display else None
            )
        
        # Show pylint scores
        if judgment.pylint_scores:
            files_info = [
                {"name": os.path.basename(f), "score": s, "status": "fixed" if s >= self.pylint_threshold else "pending"}
                for f, s in judgment.pylint_scores.items()
            ]
            self.formatter.print_file_table(files_info, "Pylint Scores", threshold=self.pylint_threshold)
        
        # Overall result
        if judgment.passed:
            self.formatter.print_status("success", "All checks passed!")
        else:
            self.formatter.print_status("warning", "Some checks failed - will retry")
            if judgment.feedback:
                feedback_preview = judgment.feedback[:150] + "..." if len(judgment.feedback) > 150 else judgment.feedback
                self.formatter.print_status("info", f"Feedback: {feedback_preview}", indent=2)
        
        # Track test history for oscillation detection
        test_history = dict(state.get("test_history", {}))
        previously_passing_tests = []
        current_passing_tests = []
        current_failing_tests = []
        
        if judgment.test_results and hasattr(judgment.test_results, 'test_results'):
            for test_result in judgment.test_results.test_results:
                test_name = test_result.name
                passed = test_result.passed
                
                # Track current state
                if passed:
                    current_passing_tests.append(test_name)
                else:
                    current_failing_tests.append(test_name)
                
                # Update history
                if test_name not in test_history:
                    test_history[test_name] = []
                test_history[test_name].append(passed)
        
        # Detect oscillating tests (tests that flip between pass/fail)
        oscillating_tests = []
        for test_name, history in test_history.items():
            if len(history) >= 3:
                # Check for oscillation pattern: pass->fail->pass or fail->pass->fail
                recent = history[-3:]
                if recent[0] != recent[1] and recent[1] != recent[2]:
                    oscillating_tests.append(test_name)
        
        if oscillating_tests:
            self.formatter.print_status(
                "warning", 
                f"⚠️ Detected {len(oscillating_tests)} oscillating test(s) - same tests keep passing/failing",
                indent=2
            )
        
        # Get tests that were passing before this iteration (to warn fixer not to break them)
        # These are tests that passed in the previous iteration
        prev_history = state.get("test_history", {})
        for test_name, history in prev_history.items():
            if history and history[-1]:  # Last result was passing
                previously_passing_tests.append(test_name)
        
        return {
            "workflow_state": WorkflowState.COMPLETE.value if judgment.passed else WorkflowState.FIXING.value,
            "judgment": judgment.to_dict(),
            "tests_passed": judgment.passed,
            "current_scores": judgment.pylint_scores,
            "test_history": test_history,
            "previously_passing_tests": current_passing_tests,  # Tests passing NOW become "previously passing" for next iteration
            "history": state.get("history", []) + [{
                "agent": "Judge",
                "iteration": state.get("iteration", 0),
                "timestamp": datetime.now().isoformat(),
                "result": judgment.result.value,
                "tests_passed": judgment.test_results.all_passed if judgment.test_results else None,
                "test_failures": judgment.test_results.failed if judgment.test_results else 0,
                "avg_pylint": judgment.average_pylint_score,
                "oscillating_tests": oscillating_tests
            }]
        }
    
    def _should_continue(self, state: AgentState) -> str:
        """
        Determine if the self-healing loop should continue.
        
        Args:
            state: Current workflow state.
            
        Returns:
            "continue" to loop back, "end" to finish.
        """
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", self.max_iterations)
        tests_passed = state.get("tests_passed", False)
        workflow_state = state.get("workflow_state", "")
        
        # End if tests passed
        if tests_passed or workflow_state == WorkflowState.COMPLETE.value:
            self.formatter.print_status("success", "Mission Complete - All checks passed!")
            return "end"
        
        # CRITICAL: End immediately if test file errors detected
        # These are errors IN the generated test files (not source code bugs)
        # The Fixer cannot fix these - test regeneration is required
        judgment = state.get("judgment")
        if judgment:
            # Handle both Judgment object and dict
            test_file_errors = None
            if hasattr(judgment, 'test_file_errors'):
                test_file_errors = judgment.test_file_errors
            elif isinstance(judgment, dict):
                test_file_errors = judgment.get('test_file_errors', [])
            
            if test_file_errors:
                self.formatter.print_status(
                    "error", 
                    f"🚨 CRITICAL: {len(test_file_errors)} error(s) detected IN TEST FILES"
                )
                self.formatter.print_status(
                    "error",
                    "These cannot be fixed by modifying source code. Stopping workflow."
                )
                for err in test_file_errors:
                    err_type = err.get('type', 'unknown')
                    err_file = err.get('test_file', 'unknown')
                    err_msg = err.get('message', 'No details')
                    self.formatter.print_status("error", f"  → [{err_type}] {err_file}: {err_msg}")
                
                self.formatter.print_status(
                    "warning",
                    "ACTION REQUIRED: Delete test files and re-run. The TestGenerator will create new valid tests."
                )
                return "end"
        
        # End if max iterations reached
        if iteration >= max_iterations:
            self.formatter.print_status("warning", f"Max iterations ({max_iterations}) reached - Stopping")
            return "end"
        
        # End if workflow failed
        if workflow_state == WorkflowState.FAILED.value:
            self.formatter.print_status("error", "Workflow failed - Stopping")
            return "end"
        
        # No early stopping - let max_iterations be the only limit
        # Continue the loop
        self.formatter.print_status("working", f"Self-healing loop: Starting iteration {iteration + 1}")
        return "continue"
    
    def run(self) -> OrchestrationResult:
        """
        Run the full orchestration workflow.
        
        Returns:
            OrchestrationResult with the final outcome.
        """
        self.formatter.print_header("REFACTORING SWARM INITIALIZED", Icons.START)
        self.formatter.log(f"Target: {self.target_dir}", "Orchestrator")
        self.formatter.log(f"Max Iterations: {self.max_iterations}", "Orchestrator")
        self.formatter.log(f"Pylint Threshold: {self.pylint_threshold}", "Orchestrator")
        
        # Log startup
        log_experiment(
            agent_name="Orchestrator",
            model_used="system",
            action=ActionType.ANALYSIS,
            details={
                "input_prompt": f"Start refactoring workflow for {self.target_dir}",
                "output_response": "Workflow initialized",
                "target_dir": self.target_dir,
                "max_iterations": self.max_iterations
            },
            status="SUCCESS"
        )
        
        # Initialize state
        initial_state: AgentState = {
            "target_dir": self.target_dir,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "workflow_state": WorkflowState.START.value,
            "refactoring_plan": None,
            "files_analyzed": [],
            "initial_scores": {},
            "test_generation_report": None,
            "tests_generated": [],
            "fixer_report": None,
            "files_fixed": [],
            "judgment": None,
            "tests_passed": False,
            "current_scores": {},
            "errors": [],
            "history": [],
            "test_history": {},
            "previously_passing_tests": [],
            "persistent_failures": {}
        }
        
        try:
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Extract results
            tests_passed = final_state.get("tests_passed", False)
            initial_scores = final_state.get("initial_scores", {})
            final_scores = final_state.get("current_scores", {})
            iterations = final_state.get("iteration", 0)
            files_fixed = final_state.get("files_fixed", [])
            errors = final_state.get("errors", [])
            history = final_state.get("history", [])
            
            # Calculate improvement
            initial_avg = sum(initial_scores.values()) / len(initial_scores) if initial_scores else 0
            final_avg = sum(final_scores.values()) / len(final_scores) if final_scores else 0
            improvement = final_avg - initial_avg
            
            # Count issues fixed
            fixer_report = final_state.get("fixer_report", {})
            total_issues_fixed = fixer_report.get("successful_fixes", 0) if fixer_report else 0
            
            # Success requires: tests passed AND each file must meet Pylint threshold
            # Per lab requirements, tests are mandatory for success
            all_files_pass_pylint = all(score >= self.pylint_threshold for score in final_scores.values()) if final_scores else False
            success = tests_passed and all_files_pass_pylint
            
            result = OrchestrationResult(
                success=success,
                iterations_used=iterations,
                initial_scores=initial_scores,
                final_scores=final_scores,
                tests_passed=tests_passed,
                files_processed=list(set(final_state.get("files_analyzed", []) + files_fixed)),
                total_issues_fixed=total_issues_fixed,
                improvement=improvement,
                error_message="; ".join(errors) if errors else None,
                history=history
            )
            
            # Log completion
            log_experiment(
                agent_name="Orchestrator",
                model_used="system",
                action=ActionType.ANALYSIS,
                details={
                    "input_prompt": "Complete refactoring workflow",
                    "output_response": result.get_summary(),
                    "success": success,
                    "iterations": iterations,
                    "improvement": improvement
                },
                status="SUCCESS" if success else "FAILURE"
            )
            
            # Clean up backup files after successful run
            if success:
                from src.tools.file_tools import FileTools
                file_tools = FileTools(self.target_dir)
                backups_removed = file_tools.cleanup_backups()
                if backups_removed > 0:
                    self.formatter.print_status("info", f"Cleaned up {backups_removed} backup file(s)")
            
            # Print the final report using the formatter
            self.formatter.print_final_report(result)
            
            return result
            
        except Exception as e:
            self.formatter.print_status("error", f"Orchestration failed: {e}")
            
            log_experiment(
                agent_name="Orchestrator",
                model_used="system",
                action=ActionType.DEBUG,
                details={
                    "input_prompt": "Orchestration error",
                    "output_response": str(e),
                    "error": str(e)
                },
                status="FAILURE"
            )
            
            return OrchestrationResult(
                success=False,
                iterations_used=0,
                initial_scores={},
                final_scores={},
                tests_passed=False,
                files_processed=[],
                total_issues_fixed=0,
                improvement=0,
                error_message=str(e)
            )
    
    def _log(self, message: str, level: str = "INFO") -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            self.formatter.log(message, "Orchestrator", level)


def run_refactoring_swarm(
    target_dir: str,
    max_iterations: int = 10,
    pylint_threshold: float = 8.5,
    verbose: bool = True
) -> OrchestrationResult:
    """
    Convenience function to run the refactoring swarm.
    
    Args:
        target_dir: Directory containing code to refactor.
        max_iterations: Maximum self-healing iterations.
        pylint_threshold: Minimum acceptable Pylint score (default 8.5).
        verbose: Enable verbose output.
        
    Returns:
        OrchestrationResult with the outcome.
    """
    orchestrator = Orchestrator(
        target_dir=target_dir,
        max_iterations=max_iterations,
        pylint_threshold=pylint_threshold,
        verbose=verbose
    )
    
    return orchestrator.run()
