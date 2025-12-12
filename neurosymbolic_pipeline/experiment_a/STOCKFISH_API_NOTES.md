# Stockfish API Integration Notes

## Date: December 12, 2025

## Current Status: WORKING

The code now uses **Lichess Cloud Eval API** which works reliably!

### Evaluation Priority

1. **Local Stockfish binary** (if available in PATH or specified)
2. **Lichess Cloud Eval API** (free, depth 40-70, works great!)
3. **Simple material evaluation** (fallback for uncached positions)

## Lichess Cloud Eval API

- **Endpoint**: `https://lichess.org/api/cloud-eval?fen={encoded_fen}`
- **Response format**: `{"pvs": [{"cp": 18, "moves": "e2e4 e7e5"}], "depth": 40}`
- **Rate limit**: ~1 request/second recommended
- **Coverage**: Common positions have cloud evaluations; rare positions return 404

### Example Response

```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
  "knodes": 13683,
  "depth": 46,
  "pvs": [
    {"moves": "c7c5 g1f3 d7d6 d2d4 c5d4 f3d4", "cp": 27}
  ]
}
```

## Test Results (December 12, 2025)

| Samples | Mean CP Loss | Std | Success Rate |
|---------|--------------|-----|--------------|
| 10 | 661.10 | 511.56 | 100% |
| 50 | 517.06 | 496.45 | 100% |

Note: CP loss > 150 indicates significant strategic differences between predicted and ground truth FENs.

## Usage

```bash
# Default (uses Lichess API)
python evaluate_cp_loss.py \
    --predictions predictions.jsonl \
    --test_data test.jsonl \
    --max_samples 100

# With local Stockfish
python evaluate_cp_loss.py \
    --predictions predictions.jsonl \
    --test_data test.jsonl \
    --stockfish_path /path/to/stockfish
```

## Previous Issues (Resolved)

### Stockfish Online API (BROKEN)

The `https://stockfish.online/api/s/v2.php` endpoint is currently returning errors:
```json
{"success":false,"data":"Internal server error. If the issue persists, contact webmaster@stockfish.online with error code 200."}
```

This has been replaced with Lichess API in the codebase.
