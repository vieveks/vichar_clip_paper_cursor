# Experiment C: Hybrid Reasoning Engine

## Objective

Implement symbolic checker for logic-based questions and hybrid routing to demonstrate the value of combining symbolic and neural approaches.

**Target**: Check detection accuracy improvement from 20% → 94% (+1780% improvement)

## Implementation

- `symbolic_checker.py` - Rule-based checker for logic questions
- `hybrid_router.py` - Question routing logic
- `evaluate_hybrid_reasoning.py` - Evaluation script

## Usage

### Basic Evaluation

```bash
cd neurosymbolic_pipeline/experiment_c

python evaluate_hybrid_reasoning.py \
    --test_data ../../data/hf_chess_puzzles/test.json \
    --question_type check_status \
    --max_samples 100
```

### Parameters

- `--test_data`: Path to test data JSON file
- `--question_type`: Question type to evaluate (default: 'check_status')
- `--max_samples`: Maximum samples to evaluate (default: all)
- `--output`: Output path for results JSON (default: `../results/exp_c/hybrid_reasoning_results.json`)

## Expected Results

- **Check Detection Accuracy**: 20% → 94% (+1780% improvement)
- **Method**: Symbolic checker provides rule-based answers for logic questions

## Question Routing

- **Symbolic Checker Path**: check_status, castling_rights, piece_location, legal_moves
- **VLM Path**: best_move, tactical_pattern, positional_advice
- **Hybrid Path**: material_balance, threat_assessment (combines both)

## Output

Results are saved to `neurosymbolic_pipeline/results/exp_c/hybrid_reasoning_results.json` with:
- Accuracy metrics
- Comparison with baseline
- Improvement calculations

## Notes

- Uses read-only access to existing benchmarking infrastructure
- Symbolic checker uses python-chess for rule-based evaluation
- All results stored in isolated `results/` directory

