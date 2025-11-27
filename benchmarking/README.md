# Chess VLM Benchmarking

This directory contains tools for benchmarking Vision Language Models (VLMs) on chess-related questions, comparing performance with and without FEN (Forsyth-Edwards Notation) context from a finetuned CLIP model.

## Overview

The benchmark evaluates whether providing FEN representation (extracted from chess board images using a finetuned CLIP model) as context improves VLM accuracy in answering chess-related questions.

**Important:** We use **CLIP-predicted FEN** (not ground truth FEN from dataset) for ground truth extraction. This ensures we test the full pipeline: image → CLIP → FEN → ground truth → VLM answers.

## Components

### Core Modules

- **`benchmark.py`**: Main benchmarking script that orchestrates the entire evaluation process
- **`questions.py`**: Question definitions and templates
- **`clip_fen_extractor.py`**: Utility to extract FEN from chess board images using the finetuned CLIP model
- **`ground_truth.py`**: Ground truth extraction using pychess and Lichess API
- **`vlm_integration.py`**: VLM (LLaVA) integration for answering questions
- **`scoring.py`**: Scoring utilities for evaluating VLM responses

## Questions

The benchmark includes 10 questions:

1. **Piece Location** (weight: 1.0): "Describe which piece is where on this chess board."
2. **Best Move** (weight: 1.0): "What is the best move in this position?"
3. **Winning Assessment** (weight: 0.0): "Who seems to be winning in this position?"
4. **Position Strength** (weight: 0.5): "How strong is the position for white?"
5. **Previous Move Quality** (weight: 1.0): "How good was the previous move?"
6. **Knight Attacks** (weight: 1.0): "Which piece is the knight attacking?"
7. **Material Count** (weight: 1.0): "What is the material count for both sides?"
8. **Check Status** (weight: 1.0): "Is either king in check?"
9. **Castling Rights** (weight: 1.0): "What are the castling rights for both sides?"
10. **Threats** (weight: 1.0): "What are the main threats in this position?"

## Dataset for Benchmarking

**Recommended Dataset:** The test split from the Hugging Face chess puzzles dataset:
- **Path:** `data/hf_chess_puzzles/test/`
- **Images:** `data/hf_chess_puzzles/test/images/` (12,500 images)
- **FEN CSV:** `data/hf_chess_puzzles/test.csv` (contains FEN strings for each image)

This is the held-out test set from the CLIP training, ensuring the benchmark evaluates on unseen data.

## Usage

### Recommended: Using Test Dataset

```bash
python benchmarking/benchmark.py \
    --clip_checkpoint runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt \
    --images_dir data/hf_chess_puzzles/test/images \
    --fen_candidates data/hf_chess_puzzles/test.csv \
    --output_dir benchmark_results
```

### With Subset of Test Images (for faster testing)

```bash
# Test on first 10 images
python benchmarking/benchmark.py \
    --clip_checkpoint runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt \
    --images $(ls data/hf_chess_puzzles/test/images/*.png | head -10) \
    --fen_candidates data/hf_chess_puzzles/test.csv \
    --output_dir benchmark_results
```

### Basic Usage (Custom Images)

```bash
python benchmarking/benchmark.py \
    --clip_checkpoint runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt \
    --images path/to/image1.png path/to/image2.png \
    --output_dir benchmark_results
```

### Using Mock VLM (for testing)

```bash
python benchmarking/benchmark.py \
    --clip_checkpoint runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt \
    --images path/to/image.png \
    --use_mock_vlm \
    --output_dir benchmark_results
```

## Requirements

### Python Packages

```bash
pip install torch torchvision open-clip-torch
pip install transformers  # For LLaVA
pip install chess  # For pychess
pip install requests  # For Lichess API
pip install pandas pillow tqdm
```

### Model Requirements

- **CLIP Model**: Trained CLIP checkpoint (e.g., from `runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt`)
- **VLM Model**: LLaVA model will be downloaded automatically from HuggingFace (requires internet connection)

## Output

The benchmark generates:

1. **`detailed_results.json`**: Complete results for each test case
2. **`results.csv`**: Results in CSV format for easy analysis
3. **`summary.json`**: Summary statistics including:
   - Average scores with/without FEN
   - Improvement metrics
   - Per-question performance breakdown

## How It Works

1. **FEN Extraction**: Uses the finetuned CLIP model to extract **PREDICTED FEN** from chess board images
   - Dataset CSV is used only for FEN candidates (options for CLIP to choose from)
   - We do NOT use ground truth FEN from dataset (that would be cheating!)
2. **Ground Truth Extraction**: Uses **PREDICTED FEN** (not ground truth) with pychess and Lichess API to get ground truth answers
   - If CLIP predicts wrong FEN, ground truth will be wrong (this is intentional - testing full pipeline)
3. **VLM Testing**: Tests VLM with two scenarios:
   - **Without FEN**: Pure image + question prompt
   - **With FEN**: Image + question prompt + **PREDICTED FEN** context
4. **Scoring**: Evaluates responses against ground truth using question-specific scoring methods
5. **Analysis**: Generates comprehensive reports comparing performance

## Scoring

Each question type has a specific scoring method:

- **Piece Location**: Checks if mentioned pieces and squares match ground truth
- **Best Move**: Validates move notation and correctness
- **Evaluation**: Compares numerical evaluation or qualitative assessment
- **Material Count**: Validates material count accuracy
- **Check Status**: Verifies check detection
- **Castling Rights**: Checks castling rights identification
- **Knight Attacks**: Validates attack square identification

Scores range from 0.0 to 1.0, with 1.0 being a perfect match.

## Notes

- The Lichess API has rate limits; the benchmark includes delays between API calls
- Some questions (like "Previous Move Quality") require additional context (previous FEN)
- Mock VLM mode is useful for testing the pipeline without loading large models
- FEN candidates CSV should have a 'fen' column with candidate FEN strings

## Example Results Structure

```json
{
  "total_images": 10,
  "total_questions": 6,
  "average_score_without_fen": 0.45,
  "average_score_with_fen": 0.72,
  "average_improvement": 0.27,
  "improvement_percentage": 60.0
}
```

