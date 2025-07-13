import torch
import torch.nn as nn
import torch.optim as optim
import math
import re
from tqdm import tqdm
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import typing
import random

from model import SPOModel
from sampler import sample_off_policy_parallel, sample_on_policy
from dummy_task import get_dummy_reward, get_dummy_prompt, SYSTEM_PROMPT

class SPOTrainer:
    def __init__(self,
                 spo_model: SPOModel,
                 optimizer: optim.Optimizer,
                 beta: float,
                 device: str,
                 reward_target: int,
                 loss_type: str = "squared",  # "squared" or "cross_entropy"
                 gradient_clip_norm: float = 1.0,
                 early_stopping_patience: int = 3):
        self.model = spo_model
        self.pi_theta = spo_model.pi_theta
        self.pi_ref = spo_model.pi_ref
        self.tokenizer = spo_model.tokenizer
        self.optimizer = optimizer
        self.beta = beta
        self.device = device
        self.reward_target = reward_target
        self.loss_type = loss_type
        self.gradient_clip_norm = gradient_clip_norm
        self.early_stopping_patience = early_stopping_patience
        
        # Early stopping tracking
        self.best_avg_prediction = float('inf')
        self.patience_counter = 0
        self.should_stop = False
        
        # Thread executor for async operations
        self.executor = ThreadPoolExecutor(max_workers=1)

    def _calculate_loss(self,
                        pi_log_probs: torch.Tensor,
                        ref_log_probs: torch.Tensor,
                        reward: torch.Tensor,
                        q0_estimate: torch.Tensor) -> torch.Tensor:
        """
        Calculates the SPO loss according to the paper.
        Implements Terminal Q-regression with either squared loss or cross-entropy loss.
        """
        # Cumulative Q-parameterization from the paper:
        # A_t^θ = β(log π_θ(a_t | a_<t, x) - log π_ref(a_t | a_<t, x))
        advantages = self.beta * (pi_log_probs - ref_log_probs)
        
        # Q_T^θ = Q̂_0 + Σ(t=1 to T) A_t^θ
        # Sum advantages over the sequence (dim=-1 sums over tokens)
        cumulative_advantage = torch.sum(advantages, dim=-1)
        
        # Add numerical stability - clamp all values to reasonable ranges
        q0_clamped = torch.clamp(q0_estimate, min=-10.0, max=10.0)
        cumulative_advantage_clamped = torch.clamp(cumulative_advantage, min=-10.0, max=10.0)
        q_terminal = q0_clamped + cumulative_advantage_clamped
        
        # Debug: occasionally print values to understand what's happening
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 100 == 0:  # Print every 100 steps (more frequent)
            print(f"DEBUG - Q0: {q0_estimate:.4f}, Cumulative Advantage: {cumulative_advantage.mean():.4f}, Q_terminal: {q_terminal.mean():.4f}, Reward: {reward.mean():.4f}")
        
        if self.loss_type == "squared":
            # Terminal Q-regression with squared loss: L(Q_T^θ, r) = (Q_T^θ - r)²
            loss = (q_terminal - reward).pow(2)
        elif self.loss_type == "cross_entropy":
            # Terminal Q-regression with cross-entropy loss
            # Since Q_t has interpretation as log p(o=1 | a_≤t, x), use BCE loss
            # Clipping is needed since Q_t can exceed 0
            q_terminal_clipped = torch.clamp(q_terminal, max=0.0)
            reward_prob = torch.exp(reward)  # Convert reward to probability
            q_terminal_prob = torch.exp(q_terminal_clipped)
            
            # Binary cross-entropy: -[target * log(pred) + (1-target) * log(1-pred)]
            loss = -(reward_prob * q_terminal_clipped + (1 - reward_prob) * torch.log(1 - q_terminal_prob + 1e-8))
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        return loss.mean()

    async def _estimate_q0(self, num_samples: int = 64, use_reference_model: bool = False) -> torch.Tensor:
        """
        Estimates Q_0 for the dummy prompt via Monte-Carlo simulation.
        If `use_reference_model` is True, it uses the reference policy (pi_ref),
        which should be done once before training. Otherwise, it uses the current
        on-policy model (pi_theta).
        """
        prompt = get_dummy_prompt()
        
        target_model_name = "pi_ref" if use_reference_model else "pi_theta"
        print(f"Estimating Q0 using {target_model_name} with {num_samples} samples...")

        with torch.no_grad():
            # The executor is used to run the synchronous `sample_on_policy` in a separate thread
            # to avoid blocking the asyncio event loop.
            loop = asyncio.get_running_loop()
            completions = await loop.run_in_executor(
                self.executor,
                sample_on_policy,
                self.model,
                self.tokenizer,
                prompt,
                num_samples,
                1024, # max_new_tokens
                1.0, # temperature
                use_reference_model,
            )
        
        rewards = [get_dummy_reward(comp, target=self.reward_target) for comp in completions]
        # Filter out any None rewards if parsing fails
        valid_rewards = [r for r in rewards if r is not None]

        if not valid_rewards:
            # If no valid rewards, assume worst case but avoid -inf
            print("WARNING: No valid rewards found during Q0 estimation. Using fallback value.")
            return torch.tensor(-1.0, device=self.device)

        rewards_tensor = torch.tensor(valid_rewards, dtype=self.model.pi_theta.dtype, device=self.device)
        
        # Count success rate for debugging
        success_rate = (rewards_tensor > -0.5).float().mean().item()
        print(f"Q0 estimation: {len(valid_rewards)} valid samples, {success_rate:.1%} success rate")
        
        # If success rate is too low, use a reasonable fallback estimate
        if success_rate < 0.01:  # Less than 1% success
            print(f"WARNING: Very low success rate ({success_rate:.1%}). Using fallback Q0 estimate.")
            # For uniform random policy on 1-100 with target 1, expected reward ≈ -0.5
            return torch.tensor(-0.5, device=self.device)
        
        # As per the paper (Sec 3.1.4), Q_t = beta * log(E[exp(r/beta)])
        # Add numerical stability - clamp exp values to avoid overflow/underflow
        exp_rewards = torch.exp(torch.clamp(rewards_tensor / self.beta, min=-20, max=0))
        mean_exp_reward = torch.mean(exp_rewards)
        
        # Avoid log(0) by adding small epsilon
        q0_estimate = self.beta * torch.log(torch.clamp(mean_exp_reward, min=1e-8))
        
        return q0_estimate

    def train_on_completion(self, prompt: str, completion: str, q0_estimate: torch.Tensor):
        """
        Performs a single training step on a given completion.
        """
        self.optimizer.zero_grad()
        
        reward = get_dummy_reward(completion, target=self.reward_target)
        if reward is None:
            return None, None, None
        reward_tensor = torch.tensor(reward, dtype=self.model.pi_theta.dtype, device=self.device)
        
        # Format the prompt and completion using the chat template
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        
        # Tokenize the full sequence. add_generation_prompt=False is correct here.
        sequence_ids = typing.cast(torch.Tensor, self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            return_tensors="pt"
        ))
        
        # To correctly find the start of the completion, we tokenize the prompt *with* the generation prompt.
        # This gives us the exact token sequence that precedes the assistant's response.
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        prompt_ids = typing.cast(torch.Tensor, self.tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ))
        prompt_len = prompt_ids.shape[1]

        # Move tensors to the correct device
        sequence_ids = sequence_ids.to(self.device)

        if sequence_ids.shape[1] > self.model.pi_theta.config.max_position_embeddings:
            return None, None, None # Skip sequences that are too long

        # Create labels and attention mask
        attention_mask = torch.ones_like(sequence_ids)
        labels = sequence_ids.clone()
        labels[:, :prompt_len] = -100 # Mask out prompt tokens
        
        pi_log_probs, ref_log_probs = self.model(sequence_ids, attention_mask, labels)
        
        # The model returns log_probs per token, we pass them to the loss function.
        loss = self._calculate_loss(pi_log_probs, ref_log_probs, reward_tensor, q0_estimate)
        loss.backward()
        
        # Clip gradients to prevent explosion (conservative clipping)
        torch.nn.utils.clip_grad_norm_(self.model.pi_theta.parameters(), max_norm=self.gradient_clip_norm)
        
        self.optimizer.step()
        
        num = -1
        match = re.search(r'\d+', completion)
        if match:
            try:
                num = int(match.group(0))
            except ValueError:
                num = -1

        return loss.item(), reward, num

    async def train(self, epochs: int, samples_per_epoch: int, q0_estimate: torch.Tensor):
        self.model.train()
        prompt = get_dummy_prompt()
        
        for epoch in range(epochs):
            if self.should_stop:
                print(f"\n--- Early stopping at epoch {epoch + 1} due to no improvement ---")
                break
                
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            
            # Re-estimate Q0 every epoch for better stability
            if epoch > 0:
                print("Re-estimating Q0 for stability...")
                old_q0 = q0_estimate.item()
                q0_estimate = await self._estimate_q0(num_samples=256, use_reference_model=False)  # Use current model
                print(f"Q0 estimate: {old_q0:.4f} → {q0_estimate.item():.4f} (change: {q0_estimate.item() - old_q0:+.4f})")

            # Generate ONLY off-policy samples as requested
            print(f"Generating {samples_per_epoch} off-policy samples...")
            
            # Off-policy samples (uniform random sampling)
            completions = await sample_off_policy_parallel(samples_per_epoch)
            
            if epoch == 0:
                # --- Print some training examples from first epoch ---
                print("\n--- First 5 training samples ---")
                for i, completion in enumerate(completions[:5]):
                    reward = get_dummy_reward(completion, target=self.reward_target)
                    if reward is not None:
                        sample_type = "off-policy"  # All samples are off-policy now
                        # Clean up ALL tokens for display - show only the number
                        clean_comp = completion.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()
                        # Extract just the first number if there are multiple tokens
                        import re
                        match = re.search(r'\d+', clean_comp)
                        if match:
                            clean_comp = match.group(0)
                        print(f"  Sample {i + 1} ({sample_type}): (Reward: {reward:.3f})")
                        print(f"    Completion: \"{clean_comp}\"")
                print("-------------------------------------------")
                # --- End printing ---

            # Q0 is now estimated once before training and passed in.
            print(f"Using pre-computed Q0 estimate for this epoch: {q0_estimate.item():.4f}")
            
            # Training phase
            total_loss, total_reward, num_valid_samples = 0, 0, 0
            print("Training on generated samples...")
            progress_bar = tqdm(completions, desc="Training on samples")
            
            for completion in progress_bar:
                if not completion:
                    continue

                loss, reward, guess = self.train_on_completion(prompt, completion, q0_estimate)
                
                if loss is not None and reward is not None:
                    total_loss += loss
                    total_reward += reward
                    num_valid_samples += 1
                    progress_bar.set_postfix({
                        "guess": f"{guess}",
                        "reward": f"{reward:.3f}",
                        "loss": f"{loss:.4f}",
                    })
            
            if num_valid_samples > 0:
                avg_loss = total_loss / num_valid_samples
                avg_reward = total_reward / num_valid_samples
                print(f"Epoch {epoch + 1} Summary: Avg Loss: {avg_loss:.4f} | Avg Reward: {avg_reward:.3f}")
                
                # Quick model sampling every epoch to monitor progress
                if (epoch + 1) % 1 == 0:  # Every epoch
                    with torch.no_grad():
                        loop = asyncio.get_running_loop()
                        quick_samples = await loop.run_in_executor(
                            self.executor,
                            sample_on_policy,
                            self.model,
                            self.tokenizer,
                            prompt,
                            5,  # Just 5 samples
                            20,  # max_new_tokens
                            0.7,  # temperature
                            False,  # use current policy
                        )
                        numbers = []
                        for sample in quick_samples:
                            match = re.search(r'\d+', sample)
                            if match:
                                numbers.append(int(match.group(0)))
                        if numbers:
                            avg_prediction = sum(numbers) / len(numbers)
                            print(f"Current model predictions: {numbers} (avg: {avg_prediction:.1f}, target: {self.reward_target})")
                            
                            # Early stopping logic based on how close we are to the target
                            current_distance = abs(avg_prediction - self.reward_target)
                            if current_distance < self.best_avg_prediction:
                                self.best_avg_prediction = current_distance
                                self.patience_counter = 0
                                print(f"✓ New best distance to target: {current_distance:.1f}")
                            else:
                                self.patience_counter += 1
                                print(f"✗ No improvement ({self.patience_counter}/{self.early_stopping_patience})")
                                
                            if self.patience_counter >= self.early_stopping_patience:
                                print(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                                self.should_stop = True
            else:
                print("No valid samples were generated in this epoch.")
        
        print("\n--- Training finished ---")