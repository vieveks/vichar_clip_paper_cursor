# Benchmarking Improved Representation Models

This script benchmarks the **trained models themselves** (not predicted FEN as context) on the 10-question reasoning benchmark.

## What This Tests

1. **LLaVA with CLIP encoders** from JSON predictor models (Exp 1A, 1B, 1D)
   - Extracts the trained CLIP encoder from JSON predictor checkpoints
   - Plugs it into LLaVA as the vision encoder
   - Tests on chess reasoning questions

2. **Fine-tuned Qwen2-VL** (Exp 1C)
   - Tests the fine-tuned Qwen2-VL model directly
   - No encoder swapping needed (it's a complete VLM)

## Prerequisites

1. **Trained models**:
   - JSON predictor checkpoints (`.pt` files) from `Improved_representations/checkpoints/`
   - Fine-tuned Qwen2-VL model from `Improved_representations/checkpoints/qwen2vl_lora/`

2. **Test dataset**:
   - `data/hf_chess_puzzles/test.csv`
   - `data/hf_chess_puzzles/test/images/`

3. **Dependencies**:
   ```bash
   pip install transformers torch pillow tqdm pandas python-dotenv
   pip install open-clip-torch  # For CLIP models
   pip install peft  # For Qwen2-VL LoRA
   ```

4. **OpenAI API Key** (Optional):
   - Required for LLM judge scoring
   - Set via: `export OPENAI_API_KEY="your-key"`
   - Or create `.env` file with `OPENAI_API_KEY=your-key`
   - Or use `--no_llm_judge` flag to skip LLM scoring

## Usage

### Test LLaVA with CLIP encoders only

```bash
cd benchmarking

# With LLM judge (requires OpenAI API key)
python benchmark_improved_models.py \
    --clip_checkpoints \
        ../Improved_representations/checkpoints/exp1a/best.pt \
        ../Improved_representations/checkpoints/exp1b/best.pt \
        ../Improved_representations/checkpoints/exp1d/best.pt \
    --dataset_csv ../data/hf_chess_puzzles/test.csv \
    --images_dir ../data/hf_chess_puzzles/test/images \
    --num_images 10 \
    --skip_qwen \
    --output_dir benchmark_results_improved

# Without LLM judge (no API key needed)
python benchmark_improved_models.py \
    --clip_checkpoints \
        ../Improved_representations/checkpoints/exp1a/best.pt \
        ../Improved_representations/checkpoints/exp1b/best.pt \
        ../Improved_representations/checkpoints/exp1d/best.pt \
    --dataset_csv ../data/hf_chess_puzzles/test.csv \
    --images_dir ../data/hf_chess_puzzles/test/images \
    --num_images 10 \
    --skip_qwen \
    --no_llm_judge \
    --output_dir benchmark_results_improved
```

### Test Fine-tuned Qwen2-VL only

```bash
python benchmark_improved_models.py \
    --qwen_model_path ../Improved_representations/checkpoints/qwen2vl_lora \
    --dataset_csv ../data/hf_chess_puzzles/test.csv \
    --images_dir ../data/hf_chess_puzzles/test/images \
    --image_base_dir ../data/hf_chess_puzzles \
    --num_images 10 \
    --skip_llava \
    --output_dir benchmark_results_improved
```

### Test Both

```bash
python benchmark_improved_models.py \
    --clip_checkpoints \
        ../Improved_representations/checkpoints/exp1a/best.pt \
        ../Improved_representations/checkpoints/exp1b/best.pt \
        ../Improved_representations/checkpoints/exp1d/best.pt \
    --qwen_model_path ../Improved_representations/checkpoints/qwen2vl_lora \
    --dataset_csv ../data/hf_chess_puzzles/test.csv \
    --images_dir ../data/hf_chess_puzzles/test/images \
    --image_base_dir ../data/hf_chess_puzzles \
    --num_images 10 \
    --output_dir benchmark_results_improved
```

## Output

Results saved to `benchmark_results_improved/`:
- `detailed_results.json` - Per-question results for each model
- `summary.json` - Aggregate statistics

### Summary Format

```json
{
  "overall": {
    "total_tests": 80,
    "avg_score": 0.45,
    "accuracy": 42.5
  },
  "by_model": {
    "LLaVA+CLIP(exp1a_best.pt)": {
      "avg_score": 0.43,
      "accuracy": 40.0,
      "total_tests": 80
    },
    "LLaVA+CLIP(exp1b_best.pt)": {
      "avg_score": 0.47,
      "accuracy": 45.0,
      "total_tests": 80
    },
    "Qwen2-VL-Finetuned": {
      "avg_score": 0.44,
      "accuracy": 41.0,
      "total_tests": 80
    }
  },
  "by_question_type": {
    "fen_extraction": {...},
    "piece_count": {...},
    ...
  }
}
```

## Questions Being Tested

The 10-question benchmark includes:
1. FEN Extraction
2. Piece Count
3. Check Status
4. Material Balance
5. Best Move
6. Tactical Patterns
7. Castling Rights
8. Piece Location

## How It Works

### For CLIP-based models (Exp 1A, 1B, 1D):
1. Loads the JSON predictor checkpoint
2. Extracts the `clip_model` component (the trained vision encoder)
3. Creates a LLaVA model and replaces its vision tower with the custom CLIP
4. Tests on reasoning questions

### For Qwen2-VL (Exp 1C):
1. Loads the fine-tuned Qwen2-VL model with LoRA weights
2. Tests directly on reasoning questions (no encoder swapping needed)

## Difference from `benchmark_json_models.py`

- **`benchmark_json_models.py`**: Tests VLMs (like GPT-4o) with predicted FEN as context
- **`benchmark_improved_models.py`**: Tests the trained models themselves as VLMs on reasoning tasks

## Expected Results

Based on the models' training objectives:

| Model | Expected Performance | Rationale |
|-------|---------------------|-----------|
| LLaVA + Exp1A CLIP | Moderate | Base CLIP, trained for JSON prediction |
| LLaVA + Exp1B CLIP | Better | Fine-tuned CLIP, trained for JSON prediction |
| LLaVA + Exp1D CLIP | Similar to 1A | Unfrozen encoder, trained for JSON prediction |
| Qwen2-VL Finetuned | Variable | Trained on JSON generation, not reasoning |

The CLIP encoders were trained to predict piece positions (79% per-square accuracy), which may help with spatial understanding for reasoning tasks.

## Troubleshooting

1. **Checkpoint not found**:
   - Make sure models are trained first
   - Check paths to checkpoint files

2. **Out of memory**:
   - Reduce `--num_images`
   - Use smaller batch sizes (modify code if needed)

3. **Missing dependencies**:
   ```bash
   pip install transformers torch pillow tqdm pandas peft open-clip-torch
   ```

## Technical Implementation Details

Integrating custom CLIP encoders (768-dim output) into LLaVA-v1.6 (Expects 1024-dim, multi-crop) required specific adaptations:

### 1. Dimension Mismatch (768 vs 1024)
- **Problem**: Custom encoders output 768 dimensions, while LLaVA's projector expects 1024.
- **Solution**: The script automatically detects this mismatch and slices the `linear_1` layer of the LLaVA multi-modal projector to match the 768 input dimension.

### 2. Token Alignment (Single Crop Force)
- **Problem**: LLaVA-Next's image processor typically generates 5 image crops (2620+ tokens) or 2 crops (1176 tokens), but our custom `OpenCLIPWrapper` was producing features for a single image, causing a shape mismatch (e.g., "Expected 1176 tokens, got 576 features").
- **Solution**: We implemented a "Single Crop Force" strategy:
    1. **Input Resizing**: Images are resized to **336x336** (single crop resolution).
    2. **Processor Config**: `do_image_splitting=False` is set to prevent multi-crop token generation.
    3. **Feature Interpolation**: The `OpenCLIPWrapper` interpolates features to a 24x24 grid (576 tokens) + CLS token.
    4. **Safety Truncation**: The script manually truncates `input_ids` to ensure exactly 576 image tokens are present, preventing generic "shape mismatch" errors during generation.

This ensures a functional pipeline where the LLaVA LLM attends to the 576 patches from our custom Grid-CLIP encoder.
