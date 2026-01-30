"""
Settings module for the Refactoring Swarm system.
Centralized configuration management with environment variable support.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default constants - use these as fallbacks to ensure consistency
DEFAULT_PYLINT_THRESHOLD = 8.25
DEFAULT_MAX_ITERATIONS = 5


@dataclass
class LLMConfig:
    """Configuration for LLM (Large Language Model) settings."""
    
    model_name: str = "gemini-2.5-pro"
    temperature: float = 0.1  # Low temperature for deterministic code fixes
    max_tokens: int = 8192
    top_p: float = 0.95
    api_key: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "❌ GOOGLE_API_KEY not found! "
                "Please set it in your .env file or environment variables."
            )


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""
    
    max_iterations: int = 5  # Maximum self-healing loop iterations
    max_fix_attempts: int = 5  # Maximum fix attempts per issue
    timeout_seconds: int = 120  # Timeout for each agent operation
    verbose: bool = True  # Enable detailed logging


@dataclass
class AnalysisConfig:
    """Configuration for code analysis tools."""
    
    pylint_threshold: float = DEFAULT_PYLINT_THRESHOLD  # Minimum acceptable pylint score (0-10)
    enable_type_checking: bool = True
    check_docstrings: bool = True
    max_line_length: int = 100
    

@dataclass
class PathConfig:
    """Configuration for file paths."""
    
    sandbox_dir: str = "./sandbox"
    logs_dir: str = "./logs"
    experiment_log: str = "./logs/experiment_data.json"
    
    def get_sandbox_path(self, target_dir: str) -> str:
        """Get absolute sandbox path for a target directory."""
        return os.path.abspath(target_dir)
    
    def is_within_sandbox(self, filepath: str, sandbox_root: str) -> bool:
        """Check if a file path is within the allowed sandbox directory."""
        abs_filepath = os.path.abspath(filepath)
        abs_sandbox = os.path.abspath(sandbox_root)
        return abs_filepath.startswith(abs_sandbox)


@dataclass
class Settings:
    """
    Main settings class that aggregates all configuration.
    
    Usage:
        settings = Settings()
        api_key = settings.llm.api_key
        max_iterations = settings.agent.max_iterations
    """
    
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables with defaults."""
        return cls(
            llm=LLMConfig(
                model_name=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "8192")),
            ),
            agent=AgentConfig(
                max_iterations=int(os.getenv("MAX_ITERATIONS", "5")),
                verbose=os.getenv("VERBOSE", "true").lower() == "true",
            ),
            analysis=AnalysisConfig(
                pylint_threshold=float(os.getenv("PYLINT_THRESHOLD", "8.5")),
            ),
            paths=PathConfig(
                sandbox_dir=os.getenv("SANDBOX_DIR", "./sandbox"),
                logs_dir=os.getenv("LOGS_DIR", "./logs"),
            ),
        )


# Global settings instance (lazy initialization)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get or create the global settings instance.
    
    Returns:
        Settings: The global settings instance.
    """
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reset_settings() -> None:
    """Reset the global settings instance (useful for testing)."""
    global _settings
    _settings = None
