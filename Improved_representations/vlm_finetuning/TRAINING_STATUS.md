# Qwen2-VL-2B Fine-tuning Status

## Current Status

**Training Started**: 2025-12-07  
**Status**: Initializing (model loading, dataset preparation)

### Dataset Creation ✅
- **Status**: Complete
- **Train**: 99,999 samples (~491 MB)
- **Val**: 12,500 samples (~219 MB)  
- **Test**: 12,500 samples (~218 MB)
- **Location**: `Improved_representations/data/vlm_dataset/`

### Training Configuration

- **Model**: Qwen2-VL-2B-Instruct
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Config**:
  - Rank (r): 16
  - Alpha: 32
  - Dropout: 0.1
- **Training Hyperparameters**:
  - Batch size: 2
  - Gradient accumulation: 8 (effective batch size: 16)
  - Learning rate: 2e-4
  - Epochs: 3
  - Max sequence length: 2048
- **Checkpointing**:
  - Save every: 1000 steps
  - Evaluate every: 1000 steps
  - Log every: 50 steps

### Output Directory
- **Checkpoints**: `Improved_representations/checkpoints/qwen2vl_json/`

## Next Steps

1. **Monitor Training**: Check logs in output directory
2. **Evaluate Model**: After training completes, run evaluation script
3. **Compare Results**: Compare with CLIP-based JSON predictor (79.32% per-square accuracy)

## Evaluation Command

After training completes:

```bash
python -m Improved_representations.vlm_finetuning.evaluate_qwen \
    --model_path Improved_representations/checkpoints/qwen2vl_json \
    --test_data Improved_representations/data/vlm_dataset/test.json \
    --image_base_dir data/hf_chess_puzzles \
    --output Improved_representations/results/qwen2vl_eval.json
```

## Expected Results

We will compare:
- **Valid JSON rate**: Percentage of responses that are valid JSON
- **Per-square accuracy**: Average accuracy of piece predictions
- **FEN accuracy**: Accuracy of FEN reconstruction from JSON
- **Exact board match**: Percentage of positions with all 64 squares correct

## Comparison Baseline

- **CLIP-based JSON Predictor**: 79.32% per-square accuracy, 0.008% exact board match
- **Baseline Qwen2-VL-2B**: To be evaluated (without fine-tuning)

