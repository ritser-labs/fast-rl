# Soft Policy Optimization (SPO)

This project implements Soft Policy Optimization (SPO), a method for fast off-policy reinforcement learning for Large Language Models (LLMs). The implementation demonstrates SPO on a simple number-guessing task where a model learns to predict a target number through reinforcement learning.

[Blog post](https://ritser.com/posts/fast-rl)

## Dependencies

- Python 3.8+
- PyTorch
- Transformers
- TQDM
- Accelerate

## Installation

1. Clone the repository
1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the experiment with:
```bash
python dummy_experiment.py
```

The script will:
1. Initialize a local model and frozen reference model
2. Sample from the untrained model to show initial behavior
3. Train the model using SPO with uniform random off-policy data
4. Sample from the trained model to show learned behavior

## Configuration

You can modify the experiment parameters directly in `dummy_experiment.py`:

- `MODEL_NAME`: The HuggingFace model to train (default: "Qwen/Qwen2.5-1.5B-Instruct")
- `BETA`: Softness parameter for the Q-function (default: 0.01)
- `LEARNING_RATE`: Learning rate for the optimizer (default: 1e-4)
- `EPOCHS`: Number of training epochs (default: 3)
- `SAMPLES_PER_EPOCH`: Number of off-policy samples per epoch (default: 200)
- `TARGET_NUMBER`: Target number for the guessing task (default: 1)
- `LOSS_TYPE`: Loss function type - "squared" or "cross_entropy" (default: "squared")
- `EARLY_STOPPING_PATIENCE`: Early stopping patience (default: 3)
- `GRADIENT_CLIP_NORM`: Gradient clipping norm (default: 1.0)

## How It Works

1. **Initialization**: Creates a trainable policy model (π_θ) and frozen reference model (π_ref)
2. **Q0 Estimation**: Estimates baseline Q-value using the reference model
3. **Training Loop**: For each epoch:
   - Generates uniform random off-policy samples (numbers 1-100)
   - Computes rewards based on distance from target number
   - Updates policy using SPO loss function
4. **Evaluation**: Samples from the trained model to demonstrate learned behavior

The reward function gives higher scores for numbers closer to the target, with perfect guesses receiving reward 0 and maximum distance receiving reward -1.

## Example Output

```
root@c20fc8f683fd:/workspace# python -m spo.dummy_experiment
--- Starting SPO Dummy Experiment ---
Device: cuda
Local Model: Qwen/Qwen2.5-1.5B-Instruct
Off-policy Sampler: 100% off-policy (uniform 1-100)
Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered.
SPOModel initialized:
  - Policy model (pi_theta): Qwen/Qwen2.5-1.5B-Instruct
  - Reference model (pi_ref): Qwen/Qwen2.5-1.5B-Instruct (frozen)
  - Device: cuda

--- Sampling from untrained model pre-training ---
  Initial Sample 1: 74
  Initial Sample 2: 74
  Initial Sample 3: 74
  Initial Sample 4: 74
  Initial Sample 5: 74
  Initial Sample 6: 74
  Initial Sample 7: 74
  Initial Sample 8: 74
  Initial Sample 9: 74
  Initial Sample 10: 74
Estimating Q0 using pi_ref with 256 samples...
Q0 estimation: 256 valid samples, 0.0% success rate
WARNING: Very low success rate (0.0%). Using fallback Q0 estimate.
--- Initial Q0 estimate (from pi_ref): -0.5000 ---

--- Epoch 1/3 ---
Generating 200 off-policy samples...
Generating 200 uniform off-policy samples (1-100)...

--- First 5 training samples ---
  Sample 1 (off-policy): (Reward: -0.606)
    Completion: "61"
  Sample 2 (off-policy): (Reward: -0.919)
    Completion: "92"
  Sample 3 (off-policy): (Reward: -0.384)
    Completion: "39"
  Sample 4 (off-policy): (Reward: -0.020)
    Completion: "3"
  Sample 5 (off-policy): (Reward: -0.475)
    Completion: "48"
-------------------------------------------
Using pre-computed Q0 estimate for this epoch: -0.5000
Training on generated samples...
Training on samples:   0%|                                                                                                                                                                            | 0/200 [00:00<?, ?it/s]DEBUG - Q0: -0.5000, Cumulative Advantage: 0.0000, Q_terminal: -0.5000, Reward: -0.6055
Training on samples:  50%|██████████████████████████████████████████████████████████████                                                              | 100/200 [00:12<00:12,  7.91it/s, guess=12, reward=-0.111, loss=0.2178]DEBUG - Q0: -0.5000, Cumulative Advantage: -0.3711, Q_terminal: -0.8711, Reward: -0.7578
Training on samples: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:25<00:00,  7.86it/s, guess=74, reward=-0.737, loss=0.0055]
Epoch 1 Summary: Avg Loss: 0.0739 | Avg Reward: -0.496
Current model predictions: [25, 23222525, 521, 235, 5221] (avg: 4645705.4, target: 1)
✓ New best distance to target: 4645704.4

--- Epoch 2/3 ---
Re-estimating Q0 for stability...
Estimating Q0 using pi_theta with 256 samples...
Q0 estimation: 256 valid samples, 7.4% success rate
Q0 estimate: -0.5000 → -0.1836 (change: +0.3164)
Generating 200 off-policy samples...
Generating 200 uniform off-policy samples (1-100)...
Using pre-computed Q0 estimate for this epoch: -0.1836
Training on generated samples...
Training on samples:   0%|                                                                                                                                                                            | 0/200 [00:00<?, ?it/s]DEBUG - Q0: -0.1836, Cumulative Advantage: 0.0610, Q_terminal: -0.1226, Reward: -0.0403
Training on samples:  50%|██████████████████████████████████████████████████████████████                                                              | 100/200 [00:11<00:11,  8.45it/s, guess=12, reward=-0.111, loss=0.0067]DEBUG - Q0: -0.1836, Cumulative Advantage: -0.6680, Q_terminal: -0.8516, Reward: -0.9297
Training on samples: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:23<00:00,  8.44it/s, guess=12, reward=-0.111, loss=0.0019]
Epoch 2 Summary: Avg Loss: 0.0183 | Avg Reward: -0.487
Current model predictions: [2, 2, 2, 2, 2] (avg: 2.0, target: 1)
✓ New best distance to target: 1.0

--- Epoch 3/3 ---
Re-estimating Q0 for stability...
Estimating Q0 using pi_theta with 256 samples...
Q0 estimation: 256 valid samples, 100.0% success rate
Q0 estimate: -0.1836 → -0.0064 (change: +0.1772)
Generating 200 off-policy samples...
Generating 200 uniform off-policy samples (1-100)...
Using pre-computed Q0 estimate for this epoch: -0.0064
Training on generated samples...
Training on samples:   0%|                                                                                                                                                                            | 0/200 [00:00<?, ?it/s]DEBUG - Q0: -0.0064, Cumulative Advantage: 0.0001, Q_terminal: -0.0063, Reward: -0.1211
Training on samples:  50%|██████████████████████████████████████████████████████████████▌                                                              | 100/200 [00:12<00:12,  7.70it/s, guess=3, reward=-0.020, loss=0.0038]DEBUG - Q0: -0.0064, Cumulative Advantage: -0.6992, Q_terminal: -0.7070, Reward: -0.7695
Training on samples: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:25<00:00,  7.91it/s, guess=39, reward=-0.384, loss=0.0018]
Epoch 3 Summary: Avg Loss: 0.0039 | Avg Reward: -0.473
Current model predictions: [2, 1, 2, 2, 2] (avg: 1.8, target: 1)
✓ New best distance to target: 0.8

--- Training finished ---

--- Sampling from trained model post-training ---
Generated 10 samples:
  Sample 1: 2
  Sample 2: 2
  Sample 3: 1
  Sample 4: 2
  Sample 5: 2
  Sample 6: 2
  Sample 7: 2
  Sample 8: 2
  Sample 9: 2
  Sample 10: 1

--- Dummy Experiment Finished ---
```