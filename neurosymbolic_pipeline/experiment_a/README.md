# Experiment A: Stockfish CP Loss Validation

## Objective

Validate that predicted FEN errors have minimal strategic impact by comparing Stockfish evaluations between predicted and ground truth FENs.

**Target**: Mean CP (centipawn) loss < 150 (expected ~127 ± 89)

## Implementation

- `stockfish_evaluator.py` - Stockfish evaluation module with Lichess API support
- `evaluate_cp_loss.py` - Main evaluation script

## Evaluation Priority

1. **Local Stockfish binary** (most accurate, if available)
2. **Lichess Cloud Eval API** (free, high depth 40-70, works great!)
3. **Simple material evaluation** (fallback)

## Usage

### Default (Uses Lichess API)

No installation required! The script uses Lichess Cloud Eval API by default:

```bash
cd neurosymbolic_pipeline/experiment_a

python evaluate_cp_loss.py \
    --predictions ../../Improved_representations/results/predictions_clip_exp1b.jsonl \
    --max_samples 100 \
    --output ../results/exp_a/cp_loss_results.json
```

### With Local Stockfish (Most Accurate)

For best accuracy or offline use:

```bash
# If Stockfish is in your PATH
python evaluate_cp_loss.py \
    --predictions ../../Improved_representations/results/predictions_clip_exp1b.jsonl \
    --depth 15 \
    --output ../results/exp_a/cp_loss_results.json

# If Stockfish is at a specific path
python evaluate_cp_loss.py \
    --predictions ../../Improved_representations/results/predictions_clip_exp1b.jsonl \
    --stockfish_path /path/to/stockfish \
    --depth 15 \
    --output ../results/exp_a/cp_loss_results.json
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--predictions` | Path to predictions JSONL file (from Exp 1B) | Required |
| `--stockfish_path` | Path to Stockfish executable | Auto-detect |
| `--depth` | Search depth for local Stockfish | 15 |
| `--max_samples` | Maximum samples to evaluate | All |
| `--output` | Output path for results JSON | `../results/exp_a/cp_loss_results.json` |

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
- Lichess API returns evaluations at depth 40-70 for common positions
- Positions not in Lichess cloud database will use simple material evaluation
- For best accuracy, install Stockfish locally: https://stockfishchess.org/download/
- All results stored in isolated `results/` directory

## API Details

### Lichess Cloud Eval API

- **Endpoint**: `https://lichess.org/api/cloud-eval?fen={encoded_fen}`
- **Response format**: `{"pvs": [{"cp": 18, "moves": "e2e4 e7e5"}], "depth": 40}`
- **Rate limit**: ~1 request/second recommended
- **Coverage**: Common positions have cloud evaluations; rare positions may return 404
