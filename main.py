#!/usr/bin/env python3
"""
The Refactoring Swarm - Main Entry Point

A multi-agent LLM system for automated Python code refactoring.
This is the mandatory entry point for the AutoCorrect Bot.

Usage:
    python main.py --target_dir "./sandbox/dataset_inconnu"

Authors: Refactoring Swarm Team
Course: IGL Lab 2025-2026
"""

import argparse
import sys
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from src.utils.logger import log_experiment, ActionType
from src.utils.output_formatter import get_formatter, reset_formatter, Colors, Icons
from src.orchestrator import Orchestrator, OrchestrationResult
from src.config.settings import get_settings


def validate_environment() -> bool:
    """
    Validate that the environment is properly configured.
    
    Returns:
        True if environment is valid, False otherwise.
    """
    errors = []
    warnings = []
    
    # Check Python version
    version = sys.version_info
    if version.major == 3 and version.minor in [10, 11]:
        pass  # Ideal version
    elif version.major == 3 and version.minor >= 12:
        warnings.append(f"Python {version.major}.{version.minor} (recommended: 3.10 or 3.11)")
    else:
        errors.append(f"Python version {version.major}.{version.minor} not supported (need 3.10+)")
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        errors.append("GOOGLE_API_KEY not found in environment")
    
    # Check required directories
    if not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True)
    
    # Print warnings
    for warning in warnings:
        print(f"⚠️  {warning}")
    
    if errors:
        for error in errors:
            print(f"❌ {error}")
        return False
    
    return True


def setup_logging(target_dir: str) -> None:
    """
    Initialize logging for the session.
    
    Args:
        target_dir: Target directory for refactoring.
    """
    log_experiment(
        agent_name="System",
        model_used="startup",
        action=ActionType.ANALYSIS,
        details={
            "input_prompt": f"Initialize refactoring session for {target_dir}",
            "output_response": "Session initialized successfully",
            "target_dir": target_dir,
            "timestamp": datetime.now().isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        },
        status="SUCCESS"
    )


def print_banner() -> None:
    """Print the startup banner using the formatter."""
    formatter = get_formatter()
    formatter.print_banner()


MAX_ITERATIONS = 10 


def run_refactoring(target_dir: str) -> OrchestrationResult:
    """
    Run the refactoring process on the target directory.
    
    Args:
        target_dir: Directory containing code to refactor.
        
    Returns:
        OrchestrationResult with the outcome.
    """
    # Get settings
    try:
        settings = get_settings()
        pylint_threshold = settings.analysis.pylint_threshold
    except Exception:
        pylint_threshold = 8.5  # Must match settings.py AnalysisConfig.pylint_threshold
    
    # Create and run orchestrator
    orchestrator = Orchestrator(
        target_dir=target_dir,
        max_iterations=MAX_ITERATIONS,
        pylint_threshold=pylint_threshold,
        verbose=True
    )
    
    return orchestrator.run()


def main():
    """Main entry point for the Refactoring Swarm."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="The Refactoring Swarm - Automated Python Code Refactoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --target_dir "./sandbox/buggy_code"
    python main.py --target_dir "./sandbox/dataset_test"
        """
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Directory containing Python code to refactor"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Reset and get formatter
    reset_formatter()
    formatter = get_formatter(verbose=not args.quiet)
    
    # Print banner
    if not args.quiet:
        print_banner()
    
    # Validate target directory
    if not os.path.exists(args.target_dir):
        formatter.print_status("error", f"Target directory not found: {args.target_dir}")
        sys.exit(1)
    
    if not os.path.isdir(args.target_dir):
        formatter.print_status("error", f"Target path is not a directory: {args.target_dir}")
        sys.exit(1)
    
    # Check for Python files
    python_files = [f for f in os.listdir(args.target_dir) if f.endswith('.py')]
    if not python_files:
        # Check subdirectories
        for root, dirs, files in os.walk(args.target_dir):
            python_files.extend([f for f in files if f.endswith('.py')])
    
    if not python_files:
        formatter.print_status("error", f"No Python files found in: {args.target_dir}")
        sys.exit(1)
    
    # Validate environment
    formatter.print_header("ENVIRONMENT CHECK", Icons.AUDITOR)
    if not validate_environment():
        formatter.print_status("error", "Environment validation failed. Please fix the issues above.")
        sys.exit(1)
    formatter.print_status("success", "Environment validated")
    
    # Setup logging
    setup_logging(args.target_dir)
    
    # Display target info
    formatter.print_header("TARGET INFORMATION", Icons.FOLDER)
    formatter.print_status("info", f"Directory: {args.target_dir}")
    formatter.print_status("info", f"Python files: {len(python_files)}")
    formatter.print_status("info", f"Max iterations: {MAX_ITERATIONS}")
    
    # List files
    formatter.print_subheader("Files to process:")
    for f in python_files[:10]:  # Show max 10 files
        formatter.print_status("info", f, indent=2)
    if len(python_files) > 10:
        formatter.print_status("info", f"... and {len(python_files) - 10} more", indent=2)
    
    formatter.print_divider()
    
    try:
        result = run_refactoring(target_dir=args.target_dir)
        
        # Log final result
        log_experiment(
            agent_name="System",
            model_used="completion",
            action=ActionType.ANALYSIS,
            details={
                "input_prompt": "Complete refactoring session",
                "output_response": result.get_summary(),
                "success": result.success,
                "iterations": result.iterations_used,
                "tests_passed": result.tests_passed,
                "improvement": result.improvement
            },
            status="SUCCESS" if result.success else "FAILURE"
        )
        
        # Final status - the detailed report is already printed by the orchestrator
        if result.success:
            print(f"\n{Colors.GREEN}{Colors.BOLD}{Icons.COMPLETE} MISSION_COMPLETE{Colors.RESET}")
            sys.exit(0)
        else:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}{Icons.WARNING} MISSION_INCOMPLETE{Colors.RESET}")
            if result.error_message:
                formatter.print_status("warning", f"Reason: {result.error_message}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}{Icons.STOP} Process interrupted by user{Colors.RESET}")
        log_experiment(
            agent_name="System",
            model_used="interrupt",
            action=ActionType.DEBUG,
            details={
                "input_prompt": "User interrupt",
                "output_response": "Process terminated by user"
            },
            status="FAILURE"
        )
        sys.exit(130)
        
    except Exception as e:
        formatter.print_status("error", f"FATAL ERROR: {e}")
        log_experiment(
            agent_name="System",
            model_used="error",
            action=ActionType.DEBUG,
            details={
                "input_prompt": "Fatal error",
                "output_response": str(e),
                "error_type": type(e).__name__
            },
            status="FAILURE"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
