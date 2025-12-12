# Experiment A: Stockfish CP Loss Validation

## Objective

Validate that predicted FEN errors have minimal strategic impact by comparing Stockfish evaluations between predicted and ground truth FENs.

**Target**: Mean CP (centipawn) loss < 150 (expected ~127 ± 89)

## Implementation

- `stockfish_evaluator.py` - Stockfish evaluation module
- `evaluate_cp_loss.py` - Main evaluation script

## Usage

### Basic Evaluation

```bash
cd neurosymbolic_pipeline/experiment_a

python evaluate_cp_loss.py \
    --predictions ../../Improved_representations/results/predictions_clip_exp1b.jsonl \
    --stockfish_path /path/to/stockfish \
    --depth 15 \
    --max_samples 1000
```

### Without Stockfish Binary

If Stockfish binary is not available, the script will use python-chess built-in evaluation (less accurate but still useful):

```bash
python evaluate_cp_loss.py \
    --predictions ../../Improved_representations/results/predictions_clip_exp1b.jsonl \
    --max_samples 1000
```

### Parameters

- `--predictions`: Path to predictions JSONL file (from Exp 1B)
- `--use_lichess_api`: Use Lichess Cloud Evaluation API (default: True, recommended)
- `--no_lichess_api`: Disable Lichess API and use python-chess simple evaluation
- `--depth`: Search depth for Stockfish evaluation (default: 15, used by Lichess API)
- `--max_samples`: Maximum samples to evaluate (default: all)
- `--output`: Output path for results JSON (default: `../results/exp_a/cp_loss_results.json`)

## Expected Results

- **Mean CP Loss**: ~127 ± 89 (target: < 150)
- **Interpretation**: Errors are "benign" - strategic evaluation preserved despite low exact match

## Output

Results are saved to `neurosymbolic_pipeline/results/exp_a/cp_loss_results.json` with:
- Mean, std, median CP loss
- Min/max CP loss
- Percentiles (25th, 50th, 75th, 90th, 95th, 99th)
- Success/failure counts

## Notes

- Uses read-only access to existing predictions
- Uses Lichess Cloud Evaluation API (free, accurate Stockfish evaluation)
- Falls back to python-chess simple evaluation if API unavailable
- Rate limiting handled automatically by GroundTruthExtractor
- All results stored in isolated `results/` directory

