import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
import copy

class SPOModel(nn.Module):
    """
    A class that encapsulates the policy model (pi_theta) and the reference model (pi_ref).
    """
    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model_name = model_name

        # 1. Load the policy model (pi_theta)
        self.pi_theta: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32
        ).to(self.device)
        
        # 2. Load the tokenizer
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 3. Create the reference model (pi_ref) and freeze it
        pi_ref_model = copy.deepcopy(self.pi_theta)
        self.pi_ref: PreTrainedModel = pi_ref_model.to(self.device)
        self.pi_ref.eval()
        for param in self.pi_ref.parameters():
            param.requires_grad = False
        
        print("SPOModel initialized:")
        print(f"  - Policy model (pi_theta): {self.model_name}")
        print(f"  - Reference model (pi_ref): {self.model_name} (frozen)")
        print(f"  - Device: {self.device}")

    def forward(self,
                sequence_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: torch.Tensor):
        """
        Calculates the log probabilities of a sequence under both the policy and reference models.
        
        Args:
            sequence_ids: Tensor of the full sequence (prompt + completion).
            attention_mask: Attention mask for the full sequence.
            labels: Tensor of the same shape as sequence_ids, with prompt tokens masked.

        Returns:
            A tuple of (pi_log_probs, ref_log_probs).
        """
        # Get logits from the policy model
        pi_outputs = self.pi_theta(input_ids=sequence_ids, attention_mask=attention_mask)
        pi_logits = pi_outputs.logits

        # Get logits from the reference model (no gradients)
        with torch.no_grad():
            ref_outputs = self.pi_ref(input_ids=sequence_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits

        # Calculate log probabilities using the provided labels to identify completion tokens
        pi_log_probs = self._get_log_probs(pi_logits, labels)
        ref_log_probs = self._get_log_probs(ref_logits, labels)
        
        return pi_log_probs, ref_log_probs

    def _get_log_probs(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Helper to compute log probabilities of target tokens from logits.
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Shift logits and labels to align them for calculating loss
        # The logit at position i is used to predict the token at position i+1
        shifted_logits = log_probs[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        # To prevent an index out-of-bounds error in torch.gather, we clamp the ignore index (-100) to a valid
        # index (0). The gathered log probability will be masked out later anyway.
        gather_labels = shifted_labels.clone()
        gather_labels[shifted_labels == -100] = 0

        # Gather the logprobs of the specific tokens that were generated
        # The shape of log_probs will be (batch_size, seq_len-1)
        log_probs = torch.gather(shifted_logits, 2, gather_labels.unsqueeze(-1)).squeeze(-1)
        
        # Mask out the ignored tokens (prompt tokens) using the original labels
        log_probs[shifted_labels == -100] = 0
        
        return log_probs


if __name__ == '__main__':
    # --- Test Script ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running test on device: {device}")
    
    # 1. Initialize model
    spo_model = SPOModel(device=device)
    tokenizer = spo_model.tokenizer

    # 2. Prepare sample data
    prompt = "Write a python function to add two numbers."
    completion = "def add(a, b): return a+b"
    
    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    sequence_tokens = tokenizer(prompt + completion, return_tensors="pt")

    prompt_ids = prompt_tokens.input_ids.to(device)
    sequence_ids = sequence_tokens.input_ids.to(device)
    attention_mask = sequence_tokens.attention_mask.to(device)
    
    # Create labels: mask out prompt tokens
    prompt_len = prompt_ids.shape[1]
    labels = sequence_ids.clone()
    labels[:, :prompt_len] = -100

    print("\n--- Input shapes ---")
    print(f"Prompt IDs shape: {prompt_ids.shape}")
    print(f"Sequence IDs shape: {sequence_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Labels shape: {labels.shape}")

    # 3. Get log probabilities
    spo_model.train() # Ensure model is in training mode for the test
    pi_log_probs, ref_log_probs = spo_model(sequence_ids, attention_mask, labels)

    print("\n--- Output shapes ---")
    print(f"pi_log_probs shape: {pi_log_probs.shape}")
    print(f"ref_log_probs shape: {ref_log_probs.shape}")

    # 4. Verify outputs
    completion_ids = sequence_ids[:, prompt_len:]
    # The output log_probs will be for the whole sequence, but prompt toks will be 0.
    assert pi_log_probs.shape == (sequence_ids.shape[0], sequence_ids.shape[1] - 1)
    assert ref_log_probs.shape == (sequence_ids.shape[0], sequence_ids.shape[1] - 1)
    
    print("\n--- Verification ---")
    print("Log prob shapes are correct.")
    
    # 5. Check reference model gradients
    # The sum of pi_log_probs should have a gradient, ref_log_probs should not.
    pi_log_probs.sum().backward()
    
    grad_found = False
    for param in spo_model.pi_theta.parameters():
        if param.grad is not None:
            grad_found = True
            break
    assert grad_found, "No gradients found in pi_theta after backward pass."
    print("Gradients were successfully computed for pi_theta.")

    grad_found_ref = False
    for param in spo_model.pi_ref.parameters():
        if param.grad is not None:
            grad_found_ref = True
            break
    assert not grad_found_ref, "Gradients found in pi_ref. It should be frozen."
    print("No gradients were found for pi_ref, as expected.")

    print("\nSPOModel test passed!")
