"""
Output Formatter - Beautiful terminal output for the Refactoring Swarm.

Provides rich, colorful, and informative output formatting for tracking
progress, displaying results, and showing status updates.
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class Colors:
    """ANSI color codes for terminal output."""
    # Basic colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    
    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    
    # Reset
    RESET = "\033[0m"
    
    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    
    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output)."""
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith('_'):
                setattr(cls, attr, "")


# Disable colors if not a TTY
if not sys.stdout.isatty():
    Colors.disable()


class Icons:
    """Unicode icons for visual feedback."""
    # Status icons
    SUCCESS = "✅"
    FAILURE = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"
    PENDING = "⏳"
    WORKING = "🔄"
    
    # Agent icons
    AUDITOR = "🔍"
    FIXER = "🔧"
    JUDGE = "⚖️"
    ORCHESTRATOR = "🎯"
    TEST_GEN = "🧪"
    
    # Action icons
    FILE = "📄"
    FOLDER = "📁"
    BUG = "🐛"
    FIX = "🩹"
    TEST = "🧪"
    STYLE = "🎨"
    
    # Progress icons
    START = "🚀"
    COMPLETE = "🎉"
    LOOP = "🔁"
    STOP = "🛑"
    
    # Metrics icons
    CHART = "📊"
    UP = "📈"
    DOWN = "📉"
    SCORE = "⭐"


class OutputFormatter:
    """
    Beautiful terminal output formatter for the Refactoring Swarm.
    
    Provides consistent, colorful, and informative output formatting.
    """
    
    # Box drawing characters
    BOX_TOP_LEFT = "╔"
    BOX_TOP_RIGHT = "╗"
    BOX_BOTTOM_LEFT = "╚"
    BOX_BOTTOM_RIGHT = "╝"
    BOX_HORIZONTAL = "═"
    BOX_VERTICAL = "║"
    BOX_T_RIGHT = "╠"
    BOX_T_LEFT = "╣"
    
    # Light box for sections
    LIGHT_HORIZONTAL = "─"
    LIGHT_VERTICAL = "│"
    LIGHT_TOP_LEFT = "┌"
    LIGHT_TOP_RIGHT = "┐"
    LIGHT_BOTTOM_LEFT = "└"
    LIGHT_BOTTOM_RIGHT = "┘"
    
    def __init__(self, width: int = 70, verbose: bool = True):
        """
        Initialize the formatter.
        
        Args:
            width: Terminal width for formatting.
            verbose: Enable verbose output.
        """
        self.width = width
        self.verbose = verbose
        self.start_time = datetime.now()
        self._iteration = 0
        self._phase = "Initializing"
    
    def print_banner(self) -> None:
        """Print the main application banner."""
        banner = f"""
{Colors.CYAN}{Colors.BOLD}
{self.BOX_TOP_LEFT}{self.BOX_HORIZONTAL * (self.width - 2)}{self.BOX_TOP_RIGHT}
{self.BOX_VERTICAL}{'🐝 THE REFACTORING SWARM 🐝'.center(self.width - 2)}{self.BOX_VERTICAL}
{self.BOX_VERTICAL}{' '.center(self.width - 2)}{self.BOX_VERTICAL}
{self.BOX_VERTICAL}{'Automated Code Analysis, Repair & Style Enhancement'.center(self.width - 2)}{self.BOX_VERTICAL}
{self.BOX_VERTICAL}{'Multi-Agent AI System with Self-Healing Loop'.center(self.width - 2)}{self.BOX_VERTICAL}
{self.BOX_VERTICAL}{' '.center(self.width - 2)}{self.BOX_VERTICAL}
{self.BOX_VERTICAL}{'IGL Lab 2025-2026'.center(self.width - 2)}{self.BOX_VERTICAL}
{self.BOX_BOTTOM_LEFT}{self.BOX_HORIZONTAL * (self.width - 2)}{self.BOX_BOTTOM_RIGHT}
{Colors.RESET}"""
        print(banner)
    
    def print_header(self, title: str, icon: str = "") -> None:
        """Print a section header."""
        if icon:
            title = f"{icon} {title}"
        
        line = self.LIGHT_HORIZONTAL * (self.width - 2)
        print(f"\n{Colors.CYAN}{self.LIGHT_TOP_LEFT}{line}{self.LIGHT_TOP_RIGHT}")
        print(f"{self.LIGHT_VERTICAL} {Colors.BOLD}{title.ljust(self.width - 4)}{Colors.RESET}{Colors.CYAN} {self.LIGHT_VERTICAL}")
        print(f"{self.LIGHT_BOTTOM_LEFT}{line}{self.LIGHT_BOTTOM_RIGHT}{Colors.RESET}")
    
    def print_subheader(self, text: str) -> None:
        """Print a subheader."""
        print(f"\n{Colors.YELLOW}{Colors.BOLD}▸ {text}{Colors.RESET}")
    
    def print_divider(self, char: str = "─", color: str = Colors.GRAY) -> None:
        """Print a horizontal divider."""
        print(f"{color}{char * self.width}{Colors.RESET}")
    
    def print_status(self, status: str, message: str, indent: int = 0) -> None:
        """
        Print a status message with icon.
        
        Args:
            status: One of 'success', 'error', 'warning', 'info', 'working'
            message: The message to display
            indent: Number of spaces to indent
        """
        icons = {
            'success': (Icons.SUCCESS, Colors.GREEN),
            'error': (Icons.FAILURE, Colors.RED),
            'warning': (Icons.WARNING, Colors.YELLOW),
            'info': (Icons.INFO, Colors.BLUE),
            'working': (Icons.WORKING, Colors.CYAN),
            'pending': (Icons.PENDING, Colors.GRAY),
        }
        
        icon, color = icons.get(status, (Icons.INFO, Colors.WHITE))
        prefix = " " * indent
        print(f"{prefix}{icon} {color}{message}{Colors.RESET}")
    
    def print_phase_start(self, phase: str, details: str = "") -> None:
        """Print the start of a phase."""
        self._phase = phase
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n{Colors.BOLD}{Colors.BLUE}┌{'─' * 50}┐{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}│{Colors.RESET} {Icons.WORKING} {Colors.CYAN}{Colors.BOLD}{phase}{Colors.RESET}")
        if details:
            print(f"{Colors.BOLD}{Colors.BLUE}│{Colors.RESET} {Colors.GRAY}{details}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}│{Colors.RESET} {Colors.DIM}Started at {timestamp}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}└{'─' * 50}┘{Colors.RESET}")
    
    def print_phase_end(self, success: bool, summary: str = "") -> None:
        """Print the end of a phase."""
        status_icon = Icons.SUCCESS if success else Icons.FAILURE
        status_color = Colors.GREEN if success else Colors.RED
        status_text = "COMPLETE" if success else "FAILED"
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        print(f"\n{status_color}  {status_icon} {self._phase}: {status_text}{Colors.RESET}")
        if summary:
            print(f"{Colors.GRAY}     {summary}{Colors.RESET}")
    
    def print_iteration_header(self, iteration: int, max_iterations: int) -> None:
        """Print iteration header for the self-healing loop."""
        self._iteration = iteration
        progress = iteration / max_iterations
        bar_width = 30
        filled = int(bar_width * progress)
        
        bar = f"{'█' * filled}{'░' * (bar_width - filled)}"
        
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}")
        print(f"{'═' * self.width}")
        print(f"  {Icons.LOOP} ITERATION {iteration}/{max_iterations}")
        print(f"  [{bar}] {progress * 100:.0f}%")
        print(f"{'═' * self.width}")
        print(f"{Colors.RESET}")
    
    def print_agent_action(self, agent: str, action: str, details: str = "") -> None:
        """
        Print an agent action.
        
        Args:
            agent: Agent name ('auditor', 'fixer', 'judge')
            action: What the agent is doing
            details: Additional details
        """
        agent_icons = {
            'auditor': (Icons.AUDITOR, Colors.BLUE, "AUDITOR"),
            'fixer': (Icons.FIXER, Colors.GREEN, "FIXER"),
            'judge': (Icons.JUDGE, Colors.MAGENTA, "JUDGE"),
            'orchestrator': (Icons.ORCHESTRATOR, Colors.CYAN, "ORCHESTRATOR"),
            'test_generator': (Icons.TEST_GEN, Colors.YELLOW, "TEST_GENERATOR"),
        }
        
        icon, color, name = agent_icons.get(agent.lower(), (Icons.INFO, Colors.WHITE, agent.upper()))
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"{Colors.GRAY}[{timestamp}]{Colors.RESET} {icon} {color}{Colors.BOLD}{name}{Colors.RESET}: {action}")
        if details:
            # Wrap long details
            for line in details.split('\n'):
                print(f"         {Colors.GRAY}{line}{Colors.RESET}")
    
    def print_file_table(self, files: List[Dict[str, Any]], title: str = "Files",
                         threshold: float = 8.5) -> None:
        """
        Print a table of files with their status.
        
        Args:
            files: List of file info dicts with 'name', 'score', 'status' keys
            title: Table title
            threshold: Pylint threshold for green color (default 8.5)
        """
        if not files:
            return
        
        print(f"\n{Colors.BOLD}  {title}:{Colors.RESET}")
        print(f"  {self.LIGHT_HORIZONTAL * 60}")
        print(f"  {Colors.BOLD}{'File':<40} {'Score':<10} {'Status':<10}{Colors.RESET}")
        print(f"  {self.LIGHT_HORIZONTAL * 60}")
        
        for f in files:
            name = f.get('name', 'unknown')[:38]
            score = f.get('score', 0)
            status = f.get('status', 'unknown')
            
            # Color based on threshold
            if score >= threshold:
                score_color = Colors.GREEN
            elif score >= threshold - 1.5:
                score_color = Colors.YELLOW
            else:
                score_color = Colors.RED
            
            # Status icon
            status_icons = {
                'fixed': f"{Colors.GREEN}{Icons.SUCCESS}",
                'failed': f"{Colors.RED}{Icons.FAILURE}",
                'pending': f"{Colors.GRAY}{Icons.PENDING}",
                'analyzing': f"{Colors.BLUE}{Icons.WORKING}",
            }
            status_display = status_icons.get(status, status)
            
            print(f"  {Colors.WHITE}{name:<40}{Colors.RESET} {score_color}{score:<10.2f}{Colors.RESET} {status_display}{Colors.RESET}")
        
        print(f"  {self.LIGHT_HORIZONTAL * 60}")
    
    def print_test_results(self, passed: int, failed: int, total: int, 
                          failures: List[str] = None) -> None:
        """
        Print test results summary.
        
        Args:
            passed: Number of tests passed
            failed: Number of tests failed
            total: Total number of tests
            failures: List of failure messages
        """
        print(f"\n{Colors.BOLD}  {Icons.TEST} Test Results:{Colors.RESET}")
        
        # Progress bar
        if total > 0:
            pass_ratio = passed / total
            bar_width = 40
            filled = int(bar_width * pass_ratio)
            bar_color = Colors.GREEN if pass_ratio == 1 else Colors.YELLOW if pass_ratio >= 0.7 else Colors.RED
            bar = f"{bar_color}{'█' * filled}{Colors.GRAY}{'░' * (bar_width - filled)}{Colors.RESET}"
            
            print(f"  [{bar}] {passed}/{total}")
        
        # Stats
        print(f"\n  {Colors.GREEN}✓ Passed: {passed}{Colors.RESET}")
        if failed > 0:
            print(f"  {Colors.RED}✗ Failed: {failed}{Colors.RESET}")
            
            if failures:
                print(f"\n  {Colors.YELLOW}Failed tests (test → source file):{Colors.RESET}")
                for i, f in enumerate(failures[:10]):  # Show up to 10 failures
                    print(f"    {Colors.RED}• {f}{Colors.RESET}")
                if len(failures) > 10:
                    print(f"    {Colors.GRAY}... and {len(failures) - 10} more failures{Colors.RESET}")
    
    def print_metrics(self, before: Dict[str, float], after: Dict[str, float],
                     tests_before: int = 0, tests_after: int = 0) -> None:
        """
        Print before/after metrics comparison.
        
        Args:
            before: Initial scores by file
            after: Final scores by file
            tests_before: Tests passing before
            tests_after: Tests passing after
        """
        before_avg = sum(before.values()) / len(before) if before else 0
        after_avg = sum(after.values()) / len(after) if after else 0
        improvement = after_avg - before_avg
        
        print(f"\n{Colors.BOLD}  {Icons.CHART} Quality Metrics:{Colors.RESET}")
        print(f"  {self.LIGHT_HORIZONTAL * 50}")
        
        # Score comparison
        print(f"\n  {'Metric':<25} {'Before':<12} {'After':<12} {'Change':<12}")
        print(f"  {self.LIGHT_HORIZONTAL * 50}")
        
        # Pylint scores
        imp_color = Colors.GREEN if improvement > 0 else Colors.RED if improvement < 0 else Colors.GRAY
        imp_icon = Icons.UP if improvement > 0 else Icons.DOWN if improvement < 0 else ""
        print(f"  {'Pylint Score':<25} {before_avg:<12.2f} {after_avg:<12.2f} {imp_color}{improvement:+.2f} {imp_icon}{Colors.RESET}")
        
        # Test improvement
        if tests_before is not None and tests_after is not None:
            test_imp = tests_after - tests_before
            test_color = Colors.GREEN if test_imp > 0 else Colors.RED if test_imp < 0 else Colors.GRAY
            print(f"  {'Tests Passing':<25} {tests_before:<12} {tests_after:<12} {test_color}{test_imp:+d}{Colors.RESET}")
        
        print(f"  {self.LIGHT_HORIZONTAL * 50}")
        
        # Per-file breakdown
        if before and after:
            print(f"\n  {Colors.BOLD}Per-file Breakdown:{Colors.RESET}")
            for filepath in sorted(set(list(before.keys()) + list(after.keys()))):
                filename = os.path.basename(filepath)
                b_score = before.get(filepath, 0)
                a_score = after.get(filepath, 0)
                file_imp = a_score - b_score
                
                imp_color = Colors.GREEN if file_imp > 0 else Colors.RED if file_imp < 0 else Colors.GRAY
                print(f"    {filename:<35} {b_score:.2f} → {a_score:.2f} {imp_color}({file_imp:+.2f}){Colors.RESET}")
    
    def print_final_report(self, result: Any) -> None:
        """
        Print the final mission report.
        
        Args:
            result: OrchestrationResult object
        """
        success = result.success
        
        # Final banner
        status_color = Colors.GREEN if success else Colors.RED
        status_text = "MISSION COMPLETE" if success else "MISSION INCOMPLETE"
        status_icon = Icons.COMPLETE if success else Icons.STOP
        
        print(f"\n{status_color}{Colors.BOLD}")
        print(f"{'═' * self.width}")
        print(f"  {status_icon} {status_text} {status_icon}")
        print(f"{'═' * self.width}{Colors.RESET}")
        
        # Summary stats
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        print(f"\n{Colors.BOLD}  📋 Summary:{Colors.RESET}")
        print(f"  {self.LIGHT_HORIZONTAL * 50}")
        
        items = [
            ("Status", f"{Colors.GREEN}SUCCESS{Colors.RESET}" if success else f"{Colors.RED}FAILED{Colors.RESET}"),
            ("Iterations Used", f"{result.iterations_used}"),
            ("Files Processed", f"{len(result.files_processed)}"),
            ("Issues Fixed", f"{result.total_issues_fixed}"),
            ("Tests", f"{'✅ Passed' if result.tests_passed else '❌ Failed'}"),
            ("Time Elapsed", f"{elapsed:.1f}s"),
        ]
        
        for label, value in items:
            print(f"  {Colors.GRAY}{label}:{Colors.RESET} {value}")
        
        # Quality improvement
        if result.initial_scores and result.final_scores:
            print(f"\n{Colors.BOLD}  📈 Quality Improvement:{Colors.RESET}")
            initial_avg = sum(result.initial_scores.values()) / len(result.initial_scores)
            final_avg = sum(result.final_scores.values()) / len(result.final_scores)
            improvement = result.improvement
            
            imp_color = Colors.GREEN if improvement > 0 else Colors.GRAY
            print(f"    Initial Score: {initial_avg:.2f}/10")
            print(f"    Final Score:   {final_avg:.2f}/10")
            print(f"    {imp_color}Improvement:   {improvement:+.2f}{Colors.RESET}")
        
        # Files breakdown
        if result.files_processed:
            print(f"\n{Colors.BOLD}  📁 Files:{Colors.RESET}")
            for filepath in result.files_processed:
                filename = os.path.basename(filepath)
                final_score = result.final_scores.get(filepath, 0)
                score_color = Colors.GREEN if final_score >= 9 else Colors.YELLOW if final_score >= 7 else Colors.RED
                print(f"    {Icons.FILE} {filename}: {score_color}{final_score:.2f}/10{Colors.RESET}")
        
        # Error messages
        if result.error_message:
            print(f"\n{Colors.YELLOW}  ⚠️ Notes:{Colors.RESET}")
            print(f"    {Colors.GRAY}{result.error_message}{Colors.RESET}")
        
        print(f"\n{status_color}{'═' * self.width}{Colors.RESET}\n")
    
    def print_progress_bar(self, current: int, total: int, label: str = "",
                          bar_width: int = 40) -> None:
        """Print a progress bar."""
        if total == 0:
            return
        
        progress = current / total
        filled = int(bar_width * progress)
        bar = f"{'█' * filled}{'░' * (bar_width - filled)}"
        
        color = Colors.GREEN if progress == 1 else Colors.CYAN
        print(f"\r  {label} [{color}{bar}{Colors.RESET}] {current}/{total} ({progress * 100:.0f}%)", end="")
        
        if current >= total:
            print()
    
    def log(self, message: str, agent: str = "", level: str = "INFO") -> None:
        """
        Log a message with timestamp.
        
        Args:
            message: Message to log
            agent: Agent name (optional)
            level: Log level
        """
        if not self.verbose:
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        level_colors = {
            'INFO': Colors.WHITE,
            'SUCCESS': Colors.GREEN,
            'WARNING': Colors.YELLOW,
            'ERROR': Colors.RED,
            'DEBUG': Colors.GRAY,
        }
        
        color = level_colors.get(level.upper(), Colors.WHITE)
        
        agent_str = f"{Colors.CYAN}[{agent}]{Colors.RESET} " if agent else ""
        print(f"{Colors.GRAY}[{timestamp}]{Colors.RESET} {agent_str}{color}{message}{Colors.RESET}")


# Global formatter instance
_formatter: Optional[OutputFormatter] = None


def get_formatter(width: int = 70, verbose: bool = True) -> OutputFormatter:
    """Get or create the global formatter instance."""
    global _formatter
    if _formatter is None:
        _formatter = OutputFormatter(width=width, verbose=verbose)
    return _formatter


def reset_formatter():
    """Reset the global formatter."""
    global _formatter
    _formatter = None
