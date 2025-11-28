# FEN Generator Updates

This document tracks the implementation details and improvements made to the FEN Generator.

## Core Implementation
- **Directory**: `FEN_generator/` created to house all new code, keeping the original codebase clean.
- **Tokenizer (`tokenizer.py`)**: Implemented a character-level tokenizer for FEN strings (board placement only).
- **Dataset (`dataset.py`)**: Created `FENGenerationDataset` to handle image loading and FEN tokenization.
- **Model (`model.py`)**: Implemented `ChessFENGenerator` using a pre-trained CLIP Encoder and a Transformer Decoder.

## Training Improvements (`train.py`)

### 1. Two-Stage Training Strategy
- **Stage 1**: Freeze Encoder, train Decoder only. (Learns FEN syntax).
- **Stage 2**: Unfreeze Encoder, fine-tune everything. (Learns spatial precision).

### 2. Performance Optimizations
- **Mixed Precision (AMP)**: Added `torch.cuda.amp` support (Autocast + GradScaler) to reduce memory usage and speed up training.
- **Debug Mode**: Fixed debug flag to correctly limit the number of batches for fast dry runs.

### 3. Stability Fixes (Addressing NaN Loss)
- **Gradient Clipping**: Added `torch.nn.utils.clip_grad_norm_` (max norm 1.0) to prevent gradient explosion during Stage 2.
- **Learning Rate Scheduler**: Implemented a **Warmup + Cosine Decay** scheduler for Stage 2 to smoothly adapt the pre-trained weights.
- **Lower Encoder LR**: Reduced `lr_encoder` to `1e-6` (from `1e-5`) for safer fine-tuning.
- **NaN Checks**: Added safeguards to skip training steps if the loss becomes NaN.

### 4. Logging
- **Log Location**: Logs are now saved to `runs/fen_generator/training.log` (alongside checkpoints) instead of the root directory.
- **Configuration**: Forced logging reconfiguration to ensure proper file writing.

## Critical Architecture Fix (v2)

### Problem: 0% Accuracy with Pooled Features
Initial training (v1) achieved low validation loss (0.68) but **0% exact match accuracy**. Investigation revealed:
- The model was using **pooled CLIP output** (single 512-dim vector)
- This lacked spatial information needed to distinguish individual chess squares
- Generated FENs were incomplete (e.g., `r3k2r/1N1P02/Q02/1PK1` instead of full 8 ranks)

### Solution: Spatial Patch Embeddings
**Model Architecture Changes (`model.py`)**:
- Changed encoder to use `forward_intermediates()` instead of standard `forward()`
- Extract spatial features: `[B, 768, 7, 7]` → reshape to `[B, 49, 768]` (49 patch tokens)
- Project from 768-dim to 512-dim to match decoder `d_model`
- Now decoder attends to **49 spatial tokens** instead of 1 global token

**v2 Training Results** (`runs/fen_generator_v2/`):
- Stage 1 (Frozen Encoder): **Val Loss 0.0225** (97% improvement!)
- Stage 2 diverged (as expected with small dataset)
- Used `best_model_stage1.pt` for evaluation

## Decoding Experiments

### 1. Greedy Decoding (Baseline)
- **Accuracy**: 0.00% (0/12500)
- **Average CER**: 0.6663
- **Issue**: Generated sequences were too short (~15-20 tokens vs expected 42-44)
- **Example**: GT `r3k2r/ppb2p1p/2nqpp2/1B1p3b/Q2N4/7P/PP1N1PP1/R1B2RK1` → GEN `r3k2r/1N1P02/Q02/1PK1`

### 2. Beam Search (beam_size=5)
- **Accuracy**: 0.00% (0/12500)
- **Average CER**: 0.7120 (slightly worse)
- **Finding**: Beam search didn't help because the issue was premature EOS *learning*, not *selection*
- **Conclusion**: Model learned to predict EOS early during training

### 3. Minimum Length Constraint (min_length=35)
- **Accuracy**: 0.00% (0/12500)
- **Average CER**: 0.7204 (worse)
- **Issue**: Forcing longer generation caused repetitive garbage: `8/8/8/8/8PrrPPPrPr8/8/8`
- **Root Cause**: Model was never trained to generate beyond ~20 tokens, so it hallucinates

## Training Data Analysis

**Findings** (`analyze_training_data.py`):
- ✓ All FENs are **complete** (7 slashes, 8 ranks)
- ✓ Average FEN length: **42.37 characters** (token length 44.37 with SOS/EOS)
- ✓ No truncated or corrupted data
- ✓ Proper board placement format

**Conclusion**: Data is perfect. The problem is the **training objective**.

## Root Cause: Padding Strategy

The model was trained with `max_len=80` padding:
```python
# In dataset.py
if len(token_ids) > self.max_len:
    token_ids = token_ids[:self.max_len]
else:
    token_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(token_ids))
```

**Why this causes early stopping**:
1. CrossEntropyLoss ignores padding (`ignore_index=pad_token_id`)
2. Model learns it can minimize loss by predicting EOS early, then padding
3. No penalty for short sequences in the loss function

## Recommended Solutions (未实施)

### Option 1: Length-Aware Loss (Preferred)
Add a length penalty term to the loss function:
```python
# Penalize sequences shorter than target length
target_len = (tgt_tokens != pad_token_id).sum(dim=1)
generated_len = ... # track in forward pass
length_penalty = F.relu(target_len - generated_len).mean()
total_loss = ce_loss + 0.1 * length_penalty
```

### Option 2: Sequence-Level Training
Use policy gradient or RL to optimize for exact match:
- Reward = 1.0 if exact match, else negative CER
- Requires more complex training loop

### Option 3: Constrained Beam Search (Implemented but Insufficient)
- ✓ Added `min_length` parameter to force longer generation
- ✗ Doesn't fix the underlying issue (model doesn't know how to continue)

### Option 4: Retrain with Better Curriculum
- Start with short FENs (endgames ~25 chars)
- Gradually increase to full positions (~45 chars)
- Progressive difficulty to teach continuation

## Current Status

**v3 Results - Length Penalty Overcorrected**
- Training: Val Loss 0.0084 (excellent)
- Evaluation: **CER 1.16** (73% worse than v2!)
- Issue: Model generates excessive length with repetitions
- Example: `8/8/8/8/8///////////` (excessive slashes and padding)

**Analysis:**
The length penalty forced the model to generate longer sequences, but:
1. It learned to pad with garbage tokens (`///`, repeated chars)
2. No understanding of valid FEN structure
3. CER degraded significantly from v2's 0.70

**Root Cause (Revised):**
The fundamental problem is **teacher forcing** during training:
- Model sees ground truth at each step
- Never learns to condition on its own predictions
- At inference, prediction errors compound ("exposure bias")
- After ~10-15 correct tokens, model enters garbage generation mode

## Recommended Next Approach

### Option A: Lower Length Penalty Weight (Quick Fix)
Reduce penalty from `0.1` to `0.01` or `0.02`:
- Less aggressive push toward longer sequences
- May find balance between v2 and v3

### Option B: Scheduled Sampling (Preferred)
Gradually replace teacher forcing with model predictions:
```python
# During training, use model's own predictions with probability p
if random.random() < sampling_prob:
    next_input = model_prediction
else:
    next_input = ground_truth
```
Start with `p=0.0`, increase to `p=0.5` over epochs.

### Option C: Minimum EOS Logit Masking (Simplest)
Mask out EOS token in logits until step > 35:
```python
if step < min_steps:
    logits[:, :, eos_token_id] = -float('inf')
```
Forces generation without changing loss function.

### Option D: Use v2 Model (Pragmatic)
Accept v2's 0% exact match but:
- CER of 0.66 means ~34% char accuracy
- Partial FENs might be useful for VLM context
- Document as limitation in paper

**Recommendation:** Try **Option C** (EOS masking) next - easiest to implement and doesn't require retraining loss changes.

## How to Run

### Training
```powershell
python FEN_generator/train.py --data_dir data/hf_chess_puzzles --checkpoint_path runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt --epochs_stage1 3 --epochs_stage2 0 --batch_size 64 --out_dir runs/fen_generator_v2
```

### Evaluation
```powershell
python FEN_generator/evaluate.py --data_dir data/hf_chess_puzzles --checkpoint_path runs/fen_generator_v2/best_model_stage1.pt --batch_size 32
```
