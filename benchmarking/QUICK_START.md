# Quick Start Guide

## Overview

This benchmarking system evaluates whether providing FEN (Forsyth-Edwards Notation) context improves VLM accuracy on chess questions.

## Installation

1. Install additional dependencies:
```bash
pip install transformers accelerate chess requests
```

2. Ensure you have a trained CLIP model checkpoint (e.g., `runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt`)

## Dataset for Benchmarking

**Recommended:** Use the test split from the training dataset:
- **Images:** `data/hf_chess_puzzles/test/images/` (12,500 test images)
- **FEN CSV:** `data/hf_chess_puzzles/test.csv` (FEN strings for matching)

This ensures evaluation on held-out test data that wasn't seen during CLIP training.

## Quick Test

### Using Test Dataset with Mock VLM (No model download required)

```bash
python benchmarking/benchmark.py \
    --clip_checkpoint runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt \
    --images_dir data/hf_chess_puzzles/test/images \
    --fen_candidates data/hf_chess_puzzles/test.csv \
    --use_mock_vlm \
    --output_dir benchmark_results
```

### Using Test Dataset with Real LLaVA Model

```bash
python benchmarking/benchmark.py \
    --clip_checkpoint runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt \
    --images_dir data/hf_chess_puzzles/test/images \
    --fen_candidates data/hf_chess_puzzles/test.csv \
    --output_dir benchmark_results
```

Note: First run will download LLaVA model (~13GB), which may take time.

### Quick Test on Single Image

```bash
python benchmarking/benchmark.py \
    --clip_checkpoint runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt \
    --images data/hf_chess_puzzles/test/images/test_000000.png \
    --fen_candidates data/hf_chess_puzzles/test.csv \
    --use_mock_vlm \
    --output_dir benchmark_results
```

## Questions Tested

The benchmark tests 10 questions:

1. **Piece Location** - "Describe which piece is where on this chess board."
2. **Best Move** - "What is the best move in this position?"
3. **Winning Assessment** - "Who seems to be winning in this position?"
4. **Position Strength** - "How strong is the position for white?"
5. **Previous Move Quality** - "How good was the previous move?"
6. **Knight Attacks** - "Which piece is the knight attacking?"
7. **Material Count** - "What is the material count for both sides?"
8. **Check Status** - "Is either king in check?"
9. **Castling Rights** - "What are the castling rights for both sides?"
10. **Threats** - "What are the main threats in this position?"

## Output

Results are saved in the output directory:
- `detailed_results.json` - Complete results for each test
- `results.csv` - CSV format for analysis
- `summary.json` - Summary statistics

## Understanding Results

The benchmark compares two scenarios:
- **Without FEN**: VLM answers using only the image
- **With FEN**: VLM answers using image + FEN context

Key metrics:
- `average_score_without_fen`: Average accuracy without FEN
- `average_score_with_fen`: Average accuracy with FEN
- `average_improvement`: Difference (should be positive if FEN helps)
- `improvement_percentage`: Percentage improvement

## Troubleshooting

### Lichess API Rate Limits
The benchmark includes rate limiting, but if you encounter errors, you can:
- Increase delays in `ground_truth.py`
- Use cached ground truth data
- Test with fewer images

### Model Loading Issues
- Ensure sufficient GPU memory for LLaVA (requires ~13GB VRAM)
- Use `--use_mock_vlm` for testing without loading models
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

### FEN Extraction Issues
- Provide `--fen_candidates` CSV for better accuracy
- Ensure CLIP checkpoint path is correct
- Check image format (PNG/JPG supported)

