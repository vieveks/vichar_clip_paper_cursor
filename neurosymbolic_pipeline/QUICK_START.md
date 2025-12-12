# Quick Start: Getting Real Results

## Prerequisites

1. **Install Stockfish** (for Experiment A)
   ```bash
   # Windows: Download from https://stockfishchess.org/download/
   # Add to PATH or note the path
   
   # Verify installation:
   stockfish --version
   ```

2. **Activate Conda Environment**
   ```bash
   conda activate pytorch_5070ti
   # Or use direct path:
   C:\Users\admin\miniconda3\envs\pytorch_5070ti\python.exe
   ```

## Run Experiments (Full Test Set)

### Experiment B: Symbolic Refinement (Priority 1)

```powershell
cd neurosymbolic_pipeline/experiment_b
$PYTHON = "C:\Users\admin\miniconda3\envs\pytorch_5070ti\python.exe"

& $PYTHON evaluate_refinement.py `
    --checkpoint ../../Improved_representations/checkpoints/json_predictor/best_model.pt `
    --test_data ../../Improved_representations/data/json_dataset/test.jsonl `
    --max_samples 12500 `
    --batch_size 32 `
    --output ../results/exp_b/refinement_full_test.json
```

**Expected Time**: ~30-60 minutes  
**Expected Results**: Exact match 0.008% → 5-10%

### Experiment A: Stockfish CP Loss (Priority 2)

```powershell
cd neurosymbolic_pipeline/experiment_a
$PYTHON = "C:\Users\admin\miniconda3\envs\pytorch_5070ti\python.exe"

# If Stockfish is in PATH:
& $PYTHON evaluate_cp_loss.py `
    --predictions ../../Improved_representations/results/predictions_clip_exp1b.jsonl `
    --max_samples 12500 `
    --depth 15 `
    --output ../results/exp_a/cp_loss_full_test.json

# If Stockfish path needs to be specified:
& $PYTHON evaluate_cp_loss.py `
    --predictions ../../Improved_representations/results/predictions_clip_exp1b.jsonl `
    --stockfish_path "C:\path\to\stockfish.exe" `
    --max_samples 12500 `
    --depth 15 `
    --output ../results/exp_a/cp_loss_full_test.json
```

**Expected Time**: ~2-4 hours (depends on Stockfish speed)  
**Expected Results**: Mean CP loss ~127 ± 89 (< 150)

### Experiment C: Hybrid Reasoning (Priority 3)

```powershell
cd neurosymbolic_pipeline/experiment_c
$PYTHON = "C:\Users\admin\miniconda3\envs\pytorch_5070ti\python.exe"

& $PYTHON evaluate_hybrid_reasoning.py `
    --test_data ../../data/hf_chess_puzzles/test.json `
    --max_samples 1000 `
    --output ../results/exp_c/hybrid_reasoning_results.json
```

**Expected Time**: ~1-2 hours  
**Expected Results**: Check detection 20% → 94%

## Check Results

All results are saved in `neurosymbolic_pipeline/results/`:
- `exp_b/refinement_full_test.json` - Refinement comparison
- `exp_a/cp_loss_full_test.json` - CP loss statistics
- `exp_c/hybrid_reasoning_results.json` - Hybrid reasoning performance

## Next: Update Paper

Once you have results, update `Paper_drafts/draft_v7.tex`:
1. Add Section 2.8: Neurosymbolic Pipeline Architecture
2. Add Results subsection: Neurosymbolic Validation
3. Include actual numbers from results files

