import json
import os
import uuid
import threading
from datetime import datetime
from enum import Enum

# Path to the log file
LOG_FILE = os.path.join("logs", "experiment_data.json")

# Thread-safe lock for file operations
_log_lock = threading.Lock()

class ActionType(str, Enum):
    """
    Enumeration of possible action types for standardized analysis.
    """
    ANALYSIS = "CODE_ANALYSIS"  # Audit, reading, bug searching
    GENERATION = "CODE_GEN"     # Creating new code/tests/docs
    DEBUG = "DEBUG"             # Analyzing runtime errors
    FIX = "FIX"                 # Applying fixes

def log_experiment(agent_name: str, model_used: str, action: ActionType, details: dict, status: str):
    """
    Record an agent interaction for scientific analysis.
    
    Thread-safe: Uses a lock to prevent race conditions during parallel operations.

    Args:
        agent_name (str): Name of the agent (e.g., "Auditor", "Fixer").
        model_used (str): LLM model used (e.g., "gemini-1.5-flash").
        action (ActionType): The type of action performed (use the ActionType Enum).
        details (dict): Dictionary containing details. MUST contain 'input_prompt' and 'output_response'.
        status (str): "SUCCESS" or "FAILURE".

    Raises:
        ValueError: If required fields are missing in 'details' or if the action is invalid.
    """
    
    # --- 1. VALIDATE ACTION TYPE ---
    # Accept either the Enum object or the corresponding string
    valid_actions = [a.value for a in ActionType]
    if isinstance(action, ActionType):
        action_str = action.value
    elif action in valid_actions:
        action_str = action
    else:
        raise ValueError(f"❌ Invalid action: '{action}'. Use the ActionType class (e.g., ActionType.FIX).")

    # --- 2. STRICT DATA VALIDATION (Prompts) ---
    # For scientific analysis, we absolutely need the prompt and response
    # for actions involving major code interaction.
    if action_str in [ActionType.ANALYSIS.value, ActionType.GENERATION.value, ActionType.DEBUG.value, ActionType.FIX.value]:
        required_keys = ["input_prompt", "output_response"]
        missing_keys = [key for key in required_keys if key not in details]
        
        if missing_keys:
            raise ValueError(
                f"❌ Logging Error (Agent: {agent_name}): "
                f"Fields {missing_keys} are missing in the 'details' dictionary. "
                f"They are REQUIRED to validate the lab work."
            )

    # --- 3. PREPARE THE ENTRY ---
    # Create the logs folder if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    entry = {
        "id": str(uuid.uuid4()),  # Unique ID to avoid duplicates when merging data
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "model": model_used,
        "action": action_str,
        "details": details,
        "status": status
    }

    # --- 4. THREAD-SAFE READ & WRITE ---
    with _log_lock:
        data = []
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Check that the file is not just empty
                        data = json.loads(content)
            except json.JSONDecodeError:
                # If the file is corrupted, start fresh (or could save a backup)
                print(f"⚠️ Warning: The log file {LOG_FILE} was corrupted. A new list has been created.")
                data = []

        data.append(entry)
        
        # Write to file
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)