import torch
import torch.optim as optim
import math
import os
import asyncio
from torch.optim import AdamW
import json
from concurrent.futures import ThreadPoolExecutor

from model import SPOModel
from trainer import SPOTrainer
from sampler import sample_on_policy
from dummy_task import get_dummy_prompt, get_dummy_reward


async def main():
    """
    Main function to run the SPO dummy experiment.
    """
    # --- Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" # Local model
    BETA = 0.01  # Much smaller beta for stability
    LEARNING_RATE = 1e-4  # More conservative learning rate  
    EPOCHS = 3  # Sweet spot - enough training but not too much
    SAMPLES_PER_EPOCH = 200  # Slightly fewer samples per epoch
    LOSS_TYPE = "squared"  # Use squared loss (paper shows this works best)
    TARGET_NUMBER = 1 # The number the model should learn to guess
    EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement for 3 epochs
    GRADIENT_CLIP_NORM = 1.0  # Gradient clipping

    print("--- Starting SPO Dummy Experiment ---")
    print(f"Device: {DEVICE}")
    print(f"Local Model: {MODEL_NAME}")
    print(f"Off-policy Sampler: 100% off-policy (uniform 1-100)")

    # --- Initialize Core Components ---
    spo_model = SPOModel(model_name=MODEL_NAME, device=DEVICE)

    # --- Sample from untrained model ---
    print("\n--- Sampling from untrained model pre-training ---")
    prompt = get_dummy_prompt()
    with ThreadPoolExecutor(max_workers=1) as executor:
        loop = asyncio.get_running_loop()
        initial_completions = await loop.run_in_executor(
            executor,
            lambda: sample_on_policy(
                model=spo_model,
                tokenizer=spo_model.tokenizer,
                prompt=prompt,
                num_samples=10,
                temperature=0.7,
                max_new_tokens=50
            )
        )
    for i, completion in enumerate(initial_completions):
        # Clean up ALL tokens for display - show only the number
        clean_comp = completion.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()
        # Extract just the first number if there are multiple tokens
        import re
        match = re.search(r'\d+', clean_comp)
        if match:
            clean_comp = match.group(0)
        print(f"  Initial Sample {i + 1}: {clean_comp}")
    # --- End sampling ---

    # --- Optimizer ---
    optimizer = AdamW(spo_model.pi_theta.parameters(), lr=LEARNING_RATE)

    # --- Trainer ---
    trainer = SPOTrainer(
        spo_model=spo_model,
        optimizer=optimizer,
        beta=BETA,
        device=DEVICE,
        reward_target=TARGET_NUMBER,
        loss_type=LOSS_TYPE,
        gradient_clip_norm=GRADIENT_CLIP_NORM,
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )

    # --- Estimate Q0 using the reference model ---
    # This is done once before training, as per the paper.
    q0_estimate = await trainer._estimate_q0(num_samples=256, use_reference_model=True)
    print(f"--- Initial Q0 estimate (from pi_ref): {q0_estimate.item():.4f} ---")

    # --- Main Training Loop ---
    await trainer.train(
        epochs=EPOCHS,
        samples_per_epoch=SAMPLES_PER_EPOCH,
        q0_estimate=q0_estimate
    )

    # --- Post-Training Evaluation ---
    print("\n--- Sampling from trained model post-training ---")
    prompt = get_dummy_prompt()
    num_evaluation_samples = 10
    spo_model.eval()  # Set the model to evaluation mode

    completions = sample_on_policy(
        model=spo_model,
        tokenizer=spo_model.tokenizer,
        prompt=prompt,
        num_samples=num_evaluation_samples,
        temperature=0.7,
        max_new_tokens=50
    )

    print(f"Generated {len(completions)} samples:")
    for i, comp in enumerate(completions):
        # Clean up ALL tokens for display - show only the number
        clean_comp = comp.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()
        # Extract just the first number if there are multiple tokens
        import re
        match = re.search(r'\d+', clean_comp)
        if match:
            clean_comp = match.group(0)
        print(f"  Sample {i+1}: {clean_comp}")

    print("\n--- Dummy Experiment Finished ---")

if __name__ == "__main__":
    asyncio.run(main()) 