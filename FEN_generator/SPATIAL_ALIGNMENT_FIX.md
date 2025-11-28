# Spatial Alignment Fix: 7×7 to 8×8 Interpolation + Expanded FEN

## Problem Identified

The FEN generator was achieving 0% exact match accuracy despite low training loss. Root cause analysis revealed:

1. **Spatial Misalignment**: CLIP ViT-B/32 produces 7×7 patch features (224/32 = 7), but chess boards are 8×8 squares. This misalignment prevents the model from accurately mapping features to specific board squares.

2. **Counting Challenge**: Standard FEN notation uses numbers (e.g., `3` for three empty squares) which requires the model to "count" empty spaces, making the task harder than necessary.

## Solution Implemented

### 1. Spatial Feature Interpolation (model.py)

**Change**: Added bilinear interpolation to upsample CLIP features from 7×7 to 8×8, aligning features with chess board squares.

```python
# Before: [B, 768, 7, 7] -> [B, 49, 768] -> [B, 49, 512]
# After:  [B, 768, 7, 7] -> [B, 768, 8, 8] -> [B, 64, 768] -> [B, 64, 512]
upsampled_features = F.interpolate(
    spatial_features, 
    size=(8, 8), 
    mode='bilinear', 
    align_corners=False
)  # [B, 768, 8, 8]
```

**Impact**: Each of the 64 feature vectors now roughly corresponds to one chess square, enabling precise piece localization.

### 2. Expanded FEN Format (dataset.py)

**Change**: Convert standard FEN to expanded format where numbers are replaced with repeated '1' tokens.

- Standard: `r3k2r` (7 tokens, requires counting)
- Expanded: `r111k11r` (8 tokens, just labeling)

**Functions Added**:
- `expand_fen(fen)`: Converts standard → expanded
- `collapse_fen(expanded_fen)`: Converts expanded → standard (for evaluation)

**Impact**: 
- Every row is exactly 8 tokens (consistent structure)
- Model learns to label each square rather than count
- Total length: 64 squares + 7 slashes = 71 tokens (vs ~42-44 for standard FEN)

### 3. Tokenizer Update (tokenizer.py)

**Change**: Removed digits 2-8 from vocabulary, keeping only '1' for empty squares.

**Before**: 27 tokens (including digits 1-8)
**After**: 20 tokens (only '1' for empty squares)

**Impact**: Simpler vocabulary, model only needs to predict piece types or empty ('1'), not count.

### 4. Evaluation Update (evaluate.py)

**Change**: Added FEN collapsing to compare generated FENs in standard format.

- Model generates expanded FEN
- Evaluation collapses both GT and generated to standard format
- Comparison done on standard FEN (for compatibility with existing metrics)

## Expected Improvements

1. **Spatial Alignment**: 8×8 features should enable accurate piece localization
2. **Consistent Structure**: Expanded FEN ensures every row is 8 tokens, teaching the model the board structure
3. **Simpler Task**: Labeling squares is easier than counting empty spaces
4. **Better Training Signal**: Model learns that output must be exactly 64 squares + 7 slashes

## Training Notes

- **max_len**: 80 is sufficient (expanded FEN is ~71 tokens + SOS/EOS = 73)
- **min_length**: Should be updated to ~71-73 for generation (expanded FEN is longer)
- **Vocabulary size**: Automatically handled via `len(tokenizer)` (now 20 instead of 27)

## Files Modified

1. `FEN_generator/model.py`: Added interpolation in `forward()` and `generate()`
2. `FEN_generator/dataset.py`: Added `expand_fen()` and `collapse_fen()` functions, updated `__getitem__()`
3. `FEN_generator/tokenizer.py`: Removed digits 2-8, kept only '1'
4. `FEN_generator/evaluate.py`: Added FEN collapsing for comparison

## Next Steps

1. Retrain model with these changes (Stage 1: frozen encoder, Stage 2: fine-tuning)
2. Monitor if model now generates complete sequences (71 tokens for expanded FEN)
3. Evaluate on test set to measure improvement in exact match accuracy
4. If successful, consider further optimizations (scheduled sampling, etc.)

## Testing

To verify the changes work:

```python
from FEN_generator.dataset import expand_fen, collapse_fen

# Test expansion
fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R"
expanded = expand_fen(fen)
print(f"Expanded: {expanded}")
# Should output: r111k11r/pppppppp/11111111/11111111/11111111/11111111/PPPPPPPP/R111K11R

# Test collapse
collapsed = collapse_fen(expanded)
print(f"Collapsed: {collapsed}")
# Should output: r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R
assert collapsed == fen.split()[0]  # Should match original (board part only)
```

