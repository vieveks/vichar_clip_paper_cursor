# Lichess API Status and Alternatives

## Date: December 12, 2025

## Current Status

The Lichess Cloud Evaluation API endpoint (`https://lichess.org/api/cloud-eval`) appears to be **deprecated or removed**. All API calls return **404 Not Found** errors.

## Research Findings

Based on web search and testing:

1. **Lichess Cloud Evaluation API**: The `/api/cloud-eval` endpoint is no longer available (returns 404)
2. **No Direct Public API**: Lichess does not currently offer a public HTTP API endpoint for Stockfish position evaluation
3. **Alternative Services**: Lichess offers:
   - **Fishnet**: Distributed Stockfish analysis system (requires setup, not a simple API)
   - **Stockfish.js**: WebAssembly version for browser use
   - **Analysis Board Embedding**: For web applications, but not programmatic API access

## Recommended Solution

**Use Local Stockfish Binary** (via python-chess)

This is the most reliable and accurate method:

1. **Install Stockfish**: Download from https://stockfishchess.org/download/
2. **Add to PATH** or specify path in code
3. **Use python-chess**: The `chess.engine` module can interface with Stockfish directly

### Advantages:
- ✅ Most accurate evaluation
- ✅ No rate limits
- ✅ No network dependency
- ✅ Full control over depth and analysis parameters
- ✅ Works offline

### Code Example:
```python
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("/path/to/stockfish")
board = chess.Board(fen)
info = engine.analyse(board, chess.engine.Limit(depth=15))
score = info['score'].score()  # Centipawns
```

## Current Implementation

The `stockfish_evaluator.py` now uses a **priority-based approach**:

1. **Priority 1**: Local Stockfish binary (auto-detected from PATH or specified path)
2. **Priority 2**: Lichess API (if enabled, but currently unavailable)
3. **Priority 3**: Python-chess simple material evaluation (fallback)

## Usage

### With Local Stockfish (Recommended)
```bash
python evaluate_cp_loss.py \
    --predictions predictions.jsonl \
    --stockfish_path /path/to/stockfish \
    --depth 15
```

### Auto-detect Stockfish
```bash
# If Stockfish is in PATH
python evaluate_cp_loss.py \
    --predictions predictions.jsonl \
    --depth 15
```

### Fallback (Simple Evaluation)
```bash
# If Stockfish not available, uses simple material evaluation
python evaluate_cp_loss.py \
    --predictions predictions.jsonl \
    --depth 15
```

## Notes

- The simple material evaluation is less accurate but still useful for testing
- For accurate CP loss measurements, local Stockfish is required
- Lichess API integration remains in code but is disabled by default due to endpoint unavailability

## Future Considerations

If Lichess reintroduces a public evaluation API:
- Update `benchmarking/ground_truth.py` with new endpoint
- Re-enable Lichess API option in `stockfish_evaluator.py`
- Test and validate new endpoint

## References

- Stockfish Download: https://stockfishchess.org/download/
- Python-Chess Documentation: https://python-chess.readthedocs.io/
- Lichess API Docs: https://lichess.org/api (no evaluation endpoint listed)

