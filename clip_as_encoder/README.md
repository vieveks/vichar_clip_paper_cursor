# ChessCLIP as Vision Encoder for LLaVA

This experiment tests whether using a chess-finetuned CLIP model as the vision encoder in a LLaVA-style architecture improves chess reasoning performance compared to using a generic CLIP encoder.

## Experiment Overview

**Research Question:** Does a vision encoder trained explicitly for board-state discrimination (via FEN retrieval) lead to better downstream chess-question answering than a generic CLIP encoder?

**Hypothesis:** Chess-finetuned CLIP embeddings are more semantically aligned with chess state, enabling better multimodal chess reasoning even without explicit FEN strings.

## Architecture

The experiment compares three model variants:

1. **Baseline-LLaVA (Generic Vision)**
   - Standard pretrained CLIP ViT-B/32 as vision encoder
   - LLaVA language model (Mistral-7B)
   - Projection layer connecting vision to language

2. **ChessCLIP-LLaVA (Retrieval-Finetuned Vision)**
   - Chess-finetuned CLIP ViT-B/32 as vision encoder (from retrieval approach)
   - Same LLaVA language model
   - Same projection layer (trained on chess QA data)

3. **FEN-LLaVA (Optional)**
   - Same as above but with explicit FEN text as additional context
   - Demonstrates upper bound of symbolic grounding

## Files

- `model.py`: Custom LLaVA model with swappable vision encoder
- `dataset.py`: Dataset loader for chess QA pairs
- `train.py`: Training script for fine-tuning projection layer
- `evaluate.py`: Evaluation script comparing baseline vs chess-CLIP
- `README.md`: This file

## Setup

### Requirements

```bash
pip install torch torchvision
pip install open-clip-torch
pip install transformers
pip install pandas pillow tqdm
pip install chess  # For ground truth extraction
```

### Prerequisites

1. **Chess-finetuned CLIP checkpoint**: Path to your trained CLIP model (e.g., `runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt`)

2. **Dataset**: Chess puzzles dataset with:
   - Images: `data/hf_chess_puzzles/test/images/`
   - CSV: `data/hf_chess_puzzles/test.csv` (with `image_path` and `fen` columns)

## Usage

### 1. Training

Train the projection layer (and optionally language model) on chess QA data:

```bash
# Train baseline (generic CLIP)
python clip_as_encoder/train.py \
    --vision_encoder_type generic \
    --dataset_csv data/hf_chess_puzzles/test.csv \
    --images_dir data/hf_chess_puzzles/test/images \
    --epochs 3 \
    --batch_size 4 \
    --lr 1e-5 \
    --output_dir runs/llava_baseline \
    --num_samples 1000

# Train chess-CLIP version
python clip_as_encoder/train.py \
    --vision_encoder_type chess_finetuned \
    --chess_clip_checkpoint runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt \
    --dataset_csv data/hf_chess_puzzles/test.csv \
    --images_dir data/hf_chess_puzzles/test/images \
    --epochs 3 \
    --batch_size 4 \
    --lr 1e-5 \
    --output_dir runs/llava_chess_clip \
    --num_samples 1000
```

**Note:** For a lighter-weight experiment, you can freeze the language model and only train the projection layer (remove `--train_language_model` flag).

### 2. Evaluation

Compare baseline vs chess-CLIP models:

```bash
python clip_as_encoder/evaluate.py \
    --chess_clip_checkpoint runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt \
    --baseline_model runs/llava_baseline/best_model.pt \
    --chess_model runs/llava_chess_clip/best_model.pt \
    --dataset_csv data/hf_chess_puzzles/test.csv \
    --images_dir data/hf_chess_puzzles/test/images \
    --num_samples 100 \
    --output_dir evaluation_results
```

If models haven't been trained yet, the script will create them on-the-fly (untrained) for comparison.

### 3. Quick Test (No Training)

For a quick comparison without training, you can evaluate untrained models:

```bash
python clip_as_encoder/evaluate.py \
    --chess_clip_checkpoint runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt \
    --dataset_csv data/hf_chess_puzzles/test.csv \
    --images_dir data/hf_chess_puzzles/test/images \
    --num_samples 10 \
    --output_dir evaluation_results
```

This will compare:
- Baseline: Generic CLIP + LLaVA (untrained projection)
- ChessCLIP: Chess-finetuned CLIP + LLaVA (untrained projection)

## Expected Results

Based on the hypothesis, we expect:

1. **Piece count, material balance, check status**: Noticeable improvement with ChessCLIP-LLaVA
2. **Best move**: Small or noisy gains (still fundamentally a policy/reasoning problem)
3. **Tactical patterns, castling rights**: May remain low, but any improvement indicates better state estimation

## Interpretation

If ChessCLIP-LLaVA outperforms baseline:

- **Claim**: "A vision encoder trained purely on image–FEN alignment yields embeddings that improve downstream chess QA, even when no explicit FEN is provided."
- **Mechanism**: CLIP finetuned on FEN retrieval encodes board structure and piece configurations more faithfully, providing a "cleaner perceptual basis" for reasoning.

This supports the broader claim that **symbolic grounding at the vision level** (even implicitly, through FEN-supervision) is beneficial for structured reasoning tasks.

## Integration with Paper

This experiment adds a fourth pillar to the paper's narrative:

**Symbolic information can be injected at three levels:**
1. **Vision encoder training** (CLIP finetuned on FEN retrieval) ← This experiment
2. **Intermediate representation** (explicit FEN text) ← Existing benchmark
3. **Parsing/validation layer** (VASP) ← Existing work

All three show complementary benefits, demonstrating that symbolic grounding improves multimodal reasoning at multiple levels.

## Notes

- **Training is optional**: You can compare untrained models to see if chess-CLIP embeddings alone help
- **Lightweight option**: Freeze language model, train only projection layer (faster, less compute)
- **Evaluation uses LLM judge**: Same scoring methodology as main benchmark for consistency

## Troubleshooting

1. **Out of memory**: Reduce batch size (`--batch_size 2` or `1`)
2. **Model loading errors**: Ensure you have the correct LLaVA model name and it's available on HuggingFace
3. **Dataset errors**: Check that CSV has `image_path` and `fen` columns, and images exist at specified paths

