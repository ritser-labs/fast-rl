import os
import torch
from typing import List, cast
import asyncio
import random

from model import SPOModel
from dummy_task import SYSTEM_PROMPT
from transformers.tokenization_utils import PreTrainedTokenizer

# --- Off-Policy Sampling ---

async def sample_off_policy_parallel(num_samples: int) -> list[str]:
    """
    Generates truly uniform random number samples as strings.
    This is a fair test of SPO's ability to learn from uniform random data.
    """
    print(f"Generating {num_samples} uniform off-policy samples (1-100)...")
    
    # Simulate async behavior
    await asyncio.sleep(0)
    
    completions = []
    for _ in range(num_samples):
        # Truly uniform sampling from 1-100
        number = str(random.randint(1, 100))
        # Just return the raw number - no EOS token manipulation
        completion = number
        completions.append(completion)
    
    return completions

# --- On-Policy Sampling from the local SPOModel ---

def sample_on_policy(
    model: SPOModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    num_samples: int,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    use_reference_model: bool = False,
) -> List[str]:
    """
    Samples completions from the on-policy model (pi_theta).
    If `use_reference_model` is True, samples from the reference model (pi_ref).
    """
    model.eval()
    
    # Choose which model to use for generation
    policy_model = model.pi_ref if use_reference_model else model.pi_theta
    
    device = policy_model.device
    
    # Use the chat template to format the prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    # Note: We set add_generation_prompt=True to ensure the model knows it's time to generate.
    input_ids = cast(torch.Tensor, tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )).to(device)

    # Generate samples
    with torch.no_grad():
        outputs = policy_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_samples,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode the generated sequences - keep EOS tokens for training consistency
    completions = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=False)
    
    return completions
