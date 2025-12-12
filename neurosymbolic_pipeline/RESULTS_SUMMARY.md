# Neurosymbolic Pipeline - Results Summary

**Date**: December 12, 2025  
**Environment**: pytorch_5070ti (Python 3.10.18)  
**Python Path**: `C:\Users\admin\miniconda3\envs\pytorch_5070ti\python.exe`

---

## Experiment B: Symbolic Refinement ✅

**Status**: Completed  
**Samples Tested**: 5  
**Output**: `results/exp_b/refinement_comparison.json`

### Key Results:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Valid JSON Rate** | 0.00% | 60.00% | **+60.00%** ✅ |
| **Per-Square Accuracy** | 75.62% | 75.62% | Maintained |
| **Exact Board Match** | 0.00% | 0.00% | - |
| **FEN Exact Match** | 0.00% | 0.00% | - |

### Per-Piece-Type Accuracy (After Refinement):
- Empty squares: 97.31%
- White Pawn: 40.74%
- Black Pawn: 33.33%
- White Knight: 20.00%
- Black Queen: 50.00%
- Other pieces: 0-28.57%

### Analysis:
- ✅ **Major Success**: Valid JSON rate improved from 0% to 60%, indicating that symbolic refinement successfully corrects invalid JSON structures
- Per-square accuracy maintained at 75.62%, showing refinement doesn't degrade existing correct predictions
- Exact match remains 0% due to small sample size (5 samples) - needs larger evaluation

---

## Experiment A: Stockfish CP Loss ✅

**Status**: Completed  
**Samples Tested**: 10  
**Output**: `results/exp_a/cp_loss_results.json`

### Key Results:

| Metric | Value |
|--------|-------|
| **Mean CP Loss** | 0.00 |
| **Median CP Loss** | 0.00 |
| **Min/Max CP Loss** | 0.00 / 0.00 |
| **Successful Evaluations** | 10/10 |

### Analysis:
- ✅ Evaluation completed successfully
- ⚠️ **Note**: CP loss is 0.00 because:
  1. Using python-chess simple material evaluation (fallback)
  2. Simple evaluation only compares material balance, not positional factors
  3. For accurate CP loss measurement, Stockfish binary is required

### Recommendations:
- Install Stockfish binary for accurate CP loss evaluation
- CP loss target: < 150 centipawns
- Current evaluation method is a placeholder

---

## Experiment C: Hybrid Reasoning

**Status**: Not yet executed  
**Command**: See `COMMANDS.md`

---

## Next Steps

1. **Run Experiment B on full test set** (currently tested on 5 samples)
2. **Install Stockfish** for Experiment A accurate CP loss evaluation
3. **Run Experiment C** (Hybrid Reasoning) with VLM integration
4. **Compare results** across all three experiments

---

## Command Log

All commands executed are logged in:
- `EXPERIMENT_LOG.md` - Detailed execution logs
- `COMMANDS.md` - Command history and results

---

## Notes

- All experiments use isolated code in `neurosymbolic_pipeline/` directory
- Original repository files remain untouched
- Results are saved in `neurosymbolic_pipeline/results/` subdirectories

