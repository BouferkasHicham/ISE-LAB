"""
Base Agent Module - Foundation for all specialized agents.

Provides common functionality for logging, LLM interaction, and state management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from src.llm import GeminiLLM, get_llm, LLMResponse
from src.utils.logger import log_experiment, ActionType


@dataclass
class AgentInternalState:
    """Represents the internal state of an agent's work."""
    agent_name: str
    status: str = "idle"  # idle, working, completed, failed
    current_task: Optional[str] = None
    iteration: int = 0
    last_result: Optional[Dict[str, Any]] = None
    errors: list = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "agent_name": self.agent_name,
            "status": self.status,
            "current_task": self.current_task,
            "iteration": self.iteration,
            "last_result": self.last_result,
            "errors": self.errors,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class BaseAgent(ABC):
    """
    Abstract base class for all refactoring agents.
    
    Provides common functionality:
    - LLM interaction with automatic logging
    - State management
    - Error handling
    - Prompt templating
    """
    
    def __init__(
        self,
        name: str,
        llm: Optional[GeminiLLM] = None,
        model_name: str = "gemini-2.5-pro",
        verbose: bool = True
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Agent identifier (e.g., "Auditor", "Fixer").
            llm: Optional pre-configured LLM instance.
            model_name: Model to use if creating new LLM.
            verbose: Enable verbose output.
        """
        self.name = name
        self.llm = llm or get_llm(model_name=model_name)
        self.model_name = model_name
        self.verbose = verbose
        self.state = AgentInternalState(agent_name=name)
        
        # System prompt (to be overridden by subclasses)
        self.system_prompt = self._get_system_prompt()
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.
        Must be implemented by subclasses.
        
        Returns:
            System prompt string.
        """
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's main task.
        Must be implemented by subclasses.
        
        Returns:
            Dictionary with execution results.
        """
        pass
    
    def _call_llm(
        self,
        prompt: str,
        action: ActionType,
        additional_details: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """
        Call the LLM with automatic logging.
        
        Args:
            prompt: The prompt to send.
            action: The action type for logging.
            additional_details: Extra details to log.
            
        Returns:
            LLMResponse from the LLM.
        """
        if self.verbose:
            print(f"🤖 [{self.name}] Calling LLM ({action.value})...")
        
        # Call LLM
        response = self.llm.generate(
            prompt=prompt,
            system_instruction=self.system_prompt
        )
        
        # Prepare logging details
        details = {
            "input_prompt": prompt,
            "output_response": response.content,
            "tokens_used": response.total_tokens,
            "duration_seconds": response.duration
        }
        
        if additional_details:
            details.update(additional_details)
        
        # Log the interaction
        log_experiment(
            agent_name=self.name,
            model_used=self.model_name,
            action=action,
            details=details,
            status="SUCCESS" if response.success else "FAILURE"
        )
        
        if self.verbose:
            status = "✅" if response.success else "❌"
            print(f"{status} [{self.name}] LLM response received ({response.duration:.2f}s)")
        
        return response
    
    def _call_llm_json(
        self,
        prompt: str,
        action: ActionType,
        additional_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call the LLM and parse JSON response with logging.
        
        Args:
            prompt: The prompt to send (should request JSON).
            action: The action type for logging.
            additional_details: Extra details to log.
            
        Returns:
            Parsed JSON dictionary.
        """
        response = self._call_llm(prompt, action, additional_details)
        
        if not response.success:
            return {"error": response.error, "success": False}
        
        return self.llm._extract_json(response.content)
    
    def _log(self, message: str, level: str = "INFO") -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            icon = {"INFO": "ℹ️", "WARN": "⚠️", "ERROR": "❌", "SUCCESS": "✅"}.get(level, "")
            print(f"{icon} [{self.name}] {message}")
    
    def _update_state(
        self,
        status: Optional[str] = None,
        task: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Update the agent's state."""
        if status:
            self.state.status = status
            if status == "working" and self.state.started_at is None:
                self.state.started_at = datetime.now()
            elif status in ("completed", "failed"):
                self.state.completed_at = datetime.now()
        
        if task:
            self.state.current_task = task
        
        if result:
            self.state.last_result = result
        
        if error:
            self.state.errors.append(error)
    
    def reset(self) -> None:
        """Reset the agent's state."""
        self.state = AgentInternalState(agent_name=self.name)
