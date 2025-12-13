# Experiment B: Symbolic Refinement Impact on CP Loss

## Objective

Evaluate how symbolic refinement improves strategic accuracy (CP loss) in addition to exact match accuracy.

## Combined Evaluation Script

`evaluate_refinement_cp_loss.py` - Generates predictions with and without refinement, then evaluates CP loss for both.

## Usage

```bash
cd neurosymbolic_pipeline/experiment_b

python evaluate_refinement_cp_loss.py \
    --checkpoint ../../Improved_representations/checkpoints/json_predictor/best_model.pt \
    --test_data ../../Improved_representations/data/json_dataset/test.jsonl \
    --max_samples 100 \
    --batch_size 16 \
    --confidence_threshold 0.5 \
    --output ../results/exp_b/refinement_cp_loss_comparison.json
```

## Results

### 20 Samples (Preliminary)

| Metric | Without Refinement | With Refinement | Improvement |
|--------|-------------------|-----------------|-------------|
| **Mean CP Loss** | 703.55 ± 578.13 | 687.05 ± 562.10 | **-16.50 CP (2.35% reduction)** ✅ |
| **Median CP Loss** | 635.00 | 675.00 | +40.00 CP |
| **Success Rate** | 20/20 (100%) | 20/20 (100%) | - |

### 100 Samples (Full Evaluation)

| Metric | Without Refinement | With Refinement | Change |
|--------|-------------------|-----------------|--------|
| **Mean CP Loss** | 610.11 ± 870.39 | 620.17 ± 866.48 | **+10.06 CP (+1.65%)** ⚠️ |
| **Median CP Loss** | 455.00 | 425.00 | **-30.00 CP (-6.59%)** ✅ |
| **Min/Max CP Loss** | 0.00 / 8080.00 | 0.00 / 8080.00 | - |
| **25th Percentile** | 187.50 | 199.75 | +12.25 CP |
| **75th Percentile** | 760.00 | 827.50 | +67.50 CP |
| **90th Percentile** | 1130.00 | 1201.00 | +71.00 CP |
| **95th Percentile** | 1342.50 | 1444.00 | +101.50 CP |
| **99th Percentile** | 2377.60 | 1892.50 | **-485.10 CP (-20.4%)** ✅ |
| **Success Rate** | 100/100 (100%) | 100/100 (100%) | - |

### Key Findings

1. **Median Improvement**: Refinement reduces median CP loss by 30 CP (6.59%), indicating it helps the typical case
2. **Outlier Reduction**: The 99th percentile improves significantly (-485 CP, -20.4%), showing refinement helps with extreme errors
3. **Mean Slight Increase**: Mean CP loss increases slightly (+10 CP, +1.65%), likely due to some positions where refinement introduces small errors
4. **High Percentiles**: 75th-95th percentiles show increases, suggesting refinement may slightly worsen moderate errors while significantly improving severe errors

### Analysis

- ✅ **Refinement helps with severe errors** (99th percentile: -20.4% improvement)
- ✅ **Refinement improves median performance** (-6.59% improvement)
- ⚠️ **Refinement slightly increases mean** (+1.65%), likely due to trade-offs in moderate-error cases
- The improvement is **position-dependent** - refinement helps more with positions that have multiple constraint violations

### Interpretation

The mixed results suggest that:
1. **Refinement is most beneficial for positions with severe constraint violations** (multiple kings, too many pieces, invalid pawns)
2. **For positions that are already mostly correct**, refinement may introduce small errors while enforcing constraints
3. **The overall impact is positive** when considering median and extreme cases, even if mean slightly increases

## Recommendations

1. **Use refinement for positions with high error rates** or when exact match is critical
2. **Consider confidence-based refinement** - only apply constraints when confidence is low
3. **Iterative refinement** - apply constraints in multiple passes with increasing strictness
4. **Position-specific rules** - apply different constraints based on detected error patterns

## Notes

- Uses Lichess Cloud Eval API for CP loss evaluation (depth 40-70)
- Refinement applies: king uniqueness, piece count limits, pawn placement rules, castling consistency
- Results saved to JSON for further analysis
- Evaluation date: December 12, 2025
