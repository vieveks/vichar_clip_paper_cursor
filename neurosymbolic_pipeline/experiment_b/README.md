# Experiment B: Symbolic Refinement

## Objective

Apply logical constraints to correct common neural errors in JSON predictions, improving exact match from 0.008% to target 8.3% (+103x improvement).

## Implementation

### Core Module

- `refinement.py` - Symbolic refinement logic with constraints:
  - King uniqueness (exactly 1 per color)
  - Piece count limit (≤32 pieces total, ≤16 per side)
  - Pawn placement rules (no pawns on rank 1/8)
  - Castling rights consistency

### Evaluation Script

- `evaluate_refinement.py` - Compares model performance with and without refinement

## Usage

### Basic Evaluation

```bash
cd neurosymbolic_pipeline/experiment_b

python evaluate_refinement.py \
    --checkpoint ../../Improved_representations/checkpoints/exp1b_finetuned_frozen/best_model.pt \
    --test_data ../../Improved_representations/data/json_dataset/test.jsonl \
    --image_base_dir ../../data/hf_chess_puzzles \
    --confidence_threshold 0.5 \
    --batch_size 32
```

### Parameters

- `--checkpoint`: Path to Exp 1B model checkpoint
- `--test_data`: Path to test JSON dataset
- `--image_base_dir`: Base directory for resolving image paths
- `--confidence_threshold`: Confidence threshold for refinement (default: 0.5)
- `--batch_size`: Batch size for evaluation (default: 32)
- `--output`: Output path for results JSON (default: `../results/exp_b/refinement_comparison.json`)

## Expected Results

- **Exact Match**: 0.008% → 8.3% (+103x improvement)
- **Per-Square Accuracy**: 79.32% → 83.1%
- **Valid JSON Rate**: Should improve significantly

## Output

Results are saved to `neurosymbolic_pipeline/results/exp_b/refinement_comparison.json` with:
- Metrics before refinement
- Metrics after refinement
- Improvement calculations

## Notes

- Uses read-only access to existing model checkpoints and datasets
- All results stored in isolated `results/` directory
- No modifications to existing code

