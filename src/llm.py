"""
LLM Module - Google Gemini Integration.

This module provides a clean interface for interacting with Google's Gemini LLM,
with support for structured outputs, retry logic, and token management.
"""

import os
import time
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from google import genai
from google.genai import types


@dataclass
class LLMResponse:
    """Represents a response from the LLM."""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration: float
    success: bool
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "content": self.content,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "duration_seconds": self.duration,
            "success": self.success,
            "error": self.error
        }


class GeminiLLM:
    """
    Google Gemini LLM wrapper with retry logic and structured output support.
    
    Provides a clean interface for:
    - Sending prompts to Gemini
    - Handling retries on failure
    - Managing safety settings
    - Extracting structured data from responses
    """
    
    # Class-level rate limiting to stay under quota
    _last_call_time: float = 0
    _rate_limit_delay: float = 0.0  # No delay - 150 RPM is plenty
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 65536,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize the Gemini LLM client.
        
        Args:
            model_name: Gemini model to use.
            api_key: Google API key (or from env GOOGLE_API_KEY).
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum output tokens.
            max_retries: Number of retry attempts on failure.
            retry_delay: Delay between retries in seconds.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Get API key
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "❌ GOOGLE_API_KEY not found! "
                "Set it in .env file or pass as parameter."
            )
        
        # Initialize the new client
        self.client = genai.Client(api_key=self.api_key)
    
    def _wait_for_rate_limit(self):
        """Wait if needed to respect rate limits."""
        elapsed = time.time() - GeminiLLM._last_call_time
        if elapsed < GeminiLLM._rate_limit_delay:
            wait_time = GeminiLLM._rate_limit_delay - elapsed
            time.sleep(wait_time)
    
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt/query.
            system_instruction: Optional system instruction to prepend.
            
        Returns:
            LLMResponse with the generated content.
        """
        # Rate limiting - wait if needed
        self._wait_for_rate_limit()
        
        start_time = time.time()
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Build contents
                contents = prompt
                
                # Build config
                config = types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    system_instruction=system_instruction if system_instruction else None,
                )
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )
                
                # Update rate limit timestamp
                GeminiLLM._last_call_time = time.time()
                
                # Extract content safely
                content = ""
                block_reason = None
                finish_reason_str = ""
                try:
                    # FIRST: Always check candidates for finish_reason and content
                    # This is more reliable than response.text which can throw
                    if response and hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        
                        # Get finish reason
                        if hasattr(candidate, 'finish_reason'):
                            finish_reason_str = str(candidate.finish_reason)
                            if 'SAFETY' in finish_reason_str or 'BLOCK' in finish_reason_str:
                                block_reason = f"Blocked by safety filter: {finish_reason_str}"
                        
                        # Extract content from parts
                        if hasattr(candidate, 'content') and candidate.content:
                            parts = candidate.content.parts
                            if parts:
                                all_text = []
                                for part in parts:
                                    if hasattr(part, 'text') and part.text:
                                        all_text.append(part.text)
                                content = ''.join(all_text)
                        
                        # Debug: if MAX_TOKENS but no content, log it
                        if 'MAX_TOKENS' in finish_reason_str and not content:
                            print(f"  🔍 DEBUG: MAX_TOKENS but no content. Has content attr: {hasattr(candidate, 'content')}")
                            if hasattr(candidate, 'content') and candidate.content:
                                print(f"  🔍 DEBUG: parts count: {len(candidate.content.parts) if candidate.content.parts else 0}")
                    
                    # Fallback: try response.text (might throw on MAX_TOKENS)
                    if not content:
                        try:
                            if response and response.text:
                                content = response.text
                        except Exception:
                            pass  # Ignore - we'll use candidate content if available
                    
                    # Check prompt feedback for blocking
                    if not content and response and hasattr(response, 'prompt_feedback'):
                        feedback = response.prompt_feedback
                        if hasattr(feedback, 'block_reason') and feedback.block_reason:
                            block_reason = f"Prompt blocked: {feedback.block_reason}"
                except Exception as extract_err:
                    # If text extraction fails, try to get any available content
                    content = str(response) if response else ""
                    block_reason = f"Extract error: {extract_err}"
                    print(f"  🔍 DEBUG: extraction error = {extract_err}")
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Extract actual token counts from usage_metadata (accurate)
                # Fall back to estimation only if metadata is unavailable
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                
                if response and hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = response.usage_metadata
                    prompt_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                    completion_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                    total_tokens = getattr(usage, 'total_token_count', 0) or 0
                    # If total not provided, calculate it
                    if total_tokens == 0:
                        total_tokens = prompt_tokens + completion_tokens
                
                # Fallback to estimation only if API didn't provide token counts
                if total_tokens == 0:
                    # Rough estimate: ~1.3 tokens per word for English text
                    prompt_tokens = int(len(prompt.split()) * 1.3)
                    completion_tokens = int(len(content.split()) * 1.3) if content else 0
                    total_tokens = prompt_tokens + completion_tokens
                
                # Success only if we got actual content
                is_success = bool(content and content.strip())
                
                # Determine error message - include finish_reason for debugging
                error_msg = None
                if not is_success:
                    if block_reason:
                        error_msg = block_reason
                    elif finish_reason_str:
                        error_msg = f"Empty response (finish_reason: {finish_reason_str})"
                    else:
                        error_msg = "Empty response from API (no candidates)"
                
                return LLMResponse(
                    content=content,
                    model=self.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    duration=duration,
                    success=is_success,
                    error=error_msg
                )
                
            except Exception as e:
                last_error = str(e)
                error_lower = last_error.lower()
                
                # Determine retry strategy based on error type
                should_retry = True
                retry_wait = self.retry_delay * (attempt + 1)  # Default linear backoff
                
                # NON-RETRYABLE ERRORS - don't waste API calls
                if any(phrase in error_lower for phrase in [
                    'safety', 'blocked', 'content filter', 'harmful',
                    'invalid api key', 'authentication', 'unauthorized',
                    'permission denied', 'forbidden', 'not found'
                ]):
                    # Safety/auth errors won't succeed on retry
                    should_retry = False
                    print(f"  ⚠️ Non-retryable error: {last_error[:100]}")
                
                # RATE LIMIT ERRORS - use exponential backoff
                elif any(phrase in error_lower for phrase in [
                    'rate limit', 'quota', '429', 'too many requests',
                    'resource exhausted', 'overloaded'
                ]):
                    # Exponential backoff for rate limits: 2^attempt * base_delay
                    retry_wait = (2 ** attempt) * self.retry_delay
                    # Cap at 60 seconds
                    retry_wait = min(retry_wait, 60.0)
                    print(f"  ⏳ Rate limit hit, waiting {retry_wait:.1f}s before retry...")
                
                # TRANSIENT ERRORS - standard retry with linear backoff
                elif any(phrase in error_lower for phrase in [
                    'timeout', 'connection', 'network', 'temporary',
                    'service unavailable', '503', '500', 'internal'
                ]):
                    # These are transient, use default linear backoff
                    print(f"  🔄 Transient error, retrying in {retry_wait:.1f}s...")
                
                # Only retry if appropriate and attempts remain
                if should_retry and attempt < self.max_retries - 1:
                    time.sleep(retry_wait)
                elif not should_retry:
                    # Break immediately for non-retryable errors
                    break
                
                continue
        
        # All retries failed
        return LLMResponse(
            content="",
            model=self.model_name,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            duration=time.time() - start_time,
            success=False,
            error=last_error
        )
    
    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a JSON response from the LLM.
        
        Extracts JSON from the response, handling markdown code blocks.
        
        Args:
            prompt: The user prompt (should request JSON output).
            system_instruction: Optional system instruction.
            
        Returns:
            Parsed JSON dictionary, or empty dict on failure.
        """
        response = self.generate(prompt, system_instruction)
        
        if not response.success:
            return {"error": response.error}
        
        return self._extract_json(response.content)
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from text, handling various formats.
        
        Args:
            text: Text that may contain JSON.
            
        Returns:
            Parsed JSON dictionary.
        """
        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code block
        import re
        json_patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
            r"\{[\s\S]*\}"
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    json_str = match.group(1) if "```" in pattern else match.group(0)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        return {"raw_content": text, "parse_error": "Could not extract JSON"}
    
    def generate_code(
        self,
        prompt: str,
        language: str = "python",
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Generate code with automatic extraction from markdown blocks.
        
        Args:
            prompt: The code generation prompt.
            language: Programming language.
            system_instruction: Optional system instruction.
            
        Returns:
            Extracted code string.
        """
        response = self.generate(prompt, system_instruction)
        
        if not response.success:
            return f"# Error: {response.error}"
        
        return self._extract_code(response.content, language)
    
    def _extract_code(self, text: str, language: str = "python") -> str:
        """
        Extract code from text, handling markdown code blocks.
        
        Args:
            text: Text that may contain code.
            language: Expected programming language.
            
        Returns:
            Extracted code string.
        """
        import re
        
        # Try language-specific code block
        pattern = rf"```{language}\s*([\s\S]*?)\s*```"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try generic code block
        pattern = r"```\s*([\s\S]*?)\s*```"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        
        # Return as-is if no code block found
        return text.strip()
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        system_instruction: Optional[str] = None
    ) -> LLMResponse:
        """
        Multi-turn conversation support.
        
        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."} dicts.
            system_instruction: Optional system instruction.
            
        Returns:
            LLMResponse with the generated content.
        """
        # Build conversation as a single prompt for simplicity
        conversation_text = ""
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n\n"
        
        conversation_text += "Assistant:"
        
        return self.generate(conversation_text, system_instruction)


# Cache of LLM instances keyed by (model_name, temperature)
_llm_instances: Dict[tuple, GeminiLLM] = {}


def get_llm(
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.2
) -> GeminiLLM:
    """
    Get or create an LLM instance for the given configuration.
    
    Instances are cached by (model_name, temperature) to avoid creating
    duplicate instances for the same configuration.
    
    Args:
        model_name: Gemini model to use.
        temperature: Sampling temperature.
        
    Returns:
        GeminiLLM instance for the specified configuration.
    """
    global _llm_instances
    cache_key = (model_name, temperature)
    
    if cache_key not in _llm_instances:
        _llm_instances[cache_key] = GeminiLLM(model_name=model_name, temperature=temperature)
    
    return _llm_instances[cache_key]


def reset_llm() -> None:
    """Reset all cached LLM instances."""
    global _llm_instances
    _llm_instances = {}
