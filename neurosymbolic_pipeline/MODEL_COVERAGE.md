# Model Coverage for Neurosymbolic Pipeline

This document tracks which models have been tested with the neurosymbolic pipeline experiments.

## JSON-Based Models

The repository contains 4 JSON-based models for chess board state prediction:

| Model | Description | Checkpoint | Status |
|-------|-------------|------------|--------|
| **Exp 1A** | Base CLIP, frozen encoder | `checkpoints/exp1a_base_frozen/best_model.pt` | ⚠️ Not tested |
| **Exp 1B** | Fine-tuned CLIP, frozen encoder | `checkpoints/json_predictor/best_model.pt` | ✅ **Tested** |
| **Exp 1C** | Qwen2-VL fine-tuned | `checkpoints/qwen2vl_json/checkpoint-189/` | ⚠️ Not tested |
| **Exp 1D** | Base CLIP, unfrozen encoder | `checkpoints/exp1d_base_unfrozen/best_model.pt` | ⚠️ Not tested |

## Experiment Coverage

### Experiment A: Stockfish CP Loss

**Status**: ✅ Implemented (using Lichess API)

**Models Tested**:
- ✅ Exp 1B (10 samples tested)

**Models Pending**:
- ⚠️ Exp 1A
- ⚠️ Exp 1C (Qwen2-VL)
- ⚠️ Exp 1D

**Note**: All JSON models should be tested to compare CP loss across different architectures.

### Experiment B: Symbolic Refinement

**Status**: ✅ Implemented

**Models Tested**:
- ✅ Exp 1B (5 samples tested, 60% valid JSON improvement)

**Models Pending**:
- ⚠️ Exp 1A (classifier-based, should work)
- ⚠️ Exp 1C (Qwen2-VL, generative model - may need different approach)
- ⚠️ Exp 1D (classifier-based, should work)

**Note**: 
- Exp 1A, 1B, 1D are classifier-based (per-square classification) - refinement should work similarly
- Exp 1C is generative (Qwen2-VL outputs JSON directly) - may need different refinement approach

### Experiment C: Hybrid Reasoning

**Status**: ⚠️ Not yet implemented

**Models to Test**: All JSON models (1A, 1B, 1C, 1D)

## Recommendations

1. **Priority 1**: Test Exp 1A and 1D with refinement (same architecture as 1B)
2. **Priority 2**: Test Exp 1C with refinement (may need generative-specific refinement)
3. **Priority 3**: Run Experiment A on all models to compare CP loss
4. **Priority 4**: Implement Experiment C for all models

## Model-Specific Notes

### Exp 1A, 1B, 1D (Classifier-Based)
- All use per-square classification (64 squares × 13 piece types)
- Refinement should work identically
- Can use same evaluation scripts

### Exp 1C (Qwen2-VL Generative)
- Generates JSON directly via text generation
- May have different error patterns
- Refinement may need to handle:
  - JSON parsing errors
  - Invalid JSON structure
  - Different confidence metrics (if available)

## Testing Commands

### Test Exp 1A with Refinement
```bash
cd neurosymbolic_pipeline/experiment_b
python evaluate_refinement.py \
    --checkpoint ../../Improved_representations/checkpoints/exp1a_base_frozen/best_model.pt \
    --test_data ../../Improved_representations/data/json_dataset/test.jsonl \
    --max_samples 100 \
    --output ../results/exp_b/refinement_exp1a.json
```

### Test Exp 1D with Refinement
```bash
cd neurosymbolic_pipeline/experiment_b
python evaluate_refinement.py \
    --checkpoint ../../Improved_representations/checkpoints/exp1d_base_unfrozen/best_model.pt \
    --test_data ../../Improved_representations/data/json_dataset/test.jsonl \
    --max_samples 100 \
    --output ../results/exp_b/refinement_exp1d.json
```

### Test Exp 1C with Refinement (if applicable)
```bash
# Note: May need different script for generative models
cd neurosymbolic_pipeline/experiment_b
python evaluate_refinement_qwen.py \
    --model_path ../../Improved_representations/checkpoints/qwen2vl_json/checkpoint-189/ \
    --test_data ../../Improved_representations/data/vlm_dataset/test.json \
    --max_samples 100 \
    --output ../results/exp_b/refinement_exp1c.json
```

