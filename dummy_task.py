import re

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

def get_dummy_prompt() -> str:
    """
    Returns the constant prompt for the number guessing task.
    """
    return "Output a single integer between 1 and 100 with no other text."

def get_dummy_reward(completion: str, target: int) -> float:
    """
    Calculates a reward based on the absolute difference between the predicted number
    and the target number. The reward is scaled to be between -1 and 0.

    - A perfect guess (80) gets a reward of 0.
    - The furthest possible guess (e.g., 100 or 1) gets a reward of -1.
    """
    try:
        # Clean up EOS tokens before parsing
        clean_completion = completion.replace('<|im_end|>', '').strip()
        # Use regex to find the first number in the completion string
        match = re.search(r'\d+', clean_completion)
        if match:
            predicted_number = int(match.group(0))
            
            # Ensure the number is within the expected range
            if 1 <= predicted_number <= 100:
                # We scale the reward to be between -1 (max diff) and 0 (no diff)
                max_diff = float(max(100 - target, target - 1))
                
                # Use linear absolute error for a consistent signal.
                abs_error = abs(predicted_number - target)
                
                # Normalize the reward
                reward = - (abs_error / max_diff)
                return reward
    except (ValueError, TypeError):
        # Could not parse an int from the completion
        pass
        
    # Return the worst possible reward if parsing fails or number is out of range
    return -1.0

if __name__ == "__main__":
    # --- Test Cases ---
    print(f"Prompt: {get_dummy_prompt()}")

    # Perfect guess
    completion_perfect = "The number is 80."
    print(f"Completion: '{completion_perfect}' -> Reward: {get_dummy_reward(completion_perfect, target=80)}")

    # Close guess
    completion_close = "I think it's 25."
    print(f"Completion: '{completion_close}' -> Reward: {get_dummy_reward(completion_close, target=80)}")

    # Far guess
    completion_far = "Maybe 100?"
    print(f"Completion: '{completion_far}' -> Reward: {get_dummy_reward(completion_far, target=80)}")

    # Far guess
    completion_far_2 = "1"
    print(f"Completion: '{completion_far_2}' -> Reward: {get_dummy_reward(completion_far_2, target=80)}")
    
    # Invalid completion
    completion_invalid = "I have no idea."
    print(f"Completion: '{completion_invalid}' -> Reward: {get_dummy_reward(completion_invalid, target=80)}")

    # Out of range
    completion_oor = "The number is 200."
    print(f"Completion: '{completion_oor}' -> Reward: {get_dummy_reward(completion_oor, target=80)}") 