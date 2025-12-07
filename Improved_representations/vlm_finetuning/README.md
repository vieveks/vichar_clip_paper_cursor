# Qwen2-VL-2B Fine-tuning for JSON Prediction

This module fine-tunes Qwen2-VL-2B-Instruct to predict JSON representations of chess positions from images.

## Overview

The fine-tuning process:
1. Converts JSON dataset to instruction-following format
2. Fine-tunes Qwen2-VL-2B using LoRA for efficiency
3. Evaluates JSON prediction accuracy

## Setup

### Requirements

```bash
pip install torch torchvision transformers
pip install peft accelerate
pip install pillow tqdm
```

### HuggingFace Token

Set your HuggingFace token:
```bash
export HF_TOKEN=your_token_here
# or
export HUGGINGFACE_TOKEN=your_token_here
```

## Usage

### Step 1: Create VLM Dataset

Convert JSON dataset to VLM instruction format:

```bash
python -m Improved_representations.vlm_finetuning.create_vlm_dataset \
    --data_dir Improved_representations/data/json_dataset \
    --output_dir Improved_representations/data/vlm_dataset \
    --base_dir . \
    --max_samples 1000  # Optional: limit for testing
```

This creates:
- `train.json` - Training data in VLM format
- `val.json` - Validation data
- `test.json` - Test data

### Step 2: Fine-tune Model

Fine-tune Qwen2-VL-2B:

```bash
python -m Improved_representations.vlm_finetuning.train_qwen \
    --data_dir Improved_representations/data/vlm_dataset \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --output_dir Improved_representations/checkpoints/qwen2vl_json \
    --image_base_dir data/hf_chess_puzzles \
    --use_lora \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_epochs 3 \
    --save_steps 500 \
    --eval_steps 500
```

### Step 3: Evaluate Model

Evaluate fine-tuned model:

```bash
python -m Improved_representations.vlm_finetuning.evaluate_qwen \
    --model_path Improved_representations/checkpoints/qwen2vl_json \
    --test_data Improved_representations/data/vlm_dataset/test.json \
    --image_base_dir data/hf_chess_puzzles \
    --output Improved_representations/results/qwen2vl_eval.json
```

## Dataset Format

The VLM dataset uses this format:

```json
{
  "id": "chess_000001",
  "image": "train/images/train_000001.png",
  "conversations": [
    {
      "from": "user",
      "value": "<image>\nAnalyze this chess board image and describe the position in JSON format..."
    },
    {
      "from": "assistant",
      "value": "{\n  \"pieces\": [...],\n  \"metadata\": {...}\n}"
    }
  ]
}
```

## Training Configuration

- **Model**: Qwen2-VL-2B-Instruct
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **LoRA Config**:
  - Rank (r): 16
  - Alpha: 32
  - Dropout: 0.1
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Training**:
  - Batch size: 4
  - Gradient accumulation: 4 (effective batch size: 16)
  - Learning rate: 2e-4
  - Epochs: 3
  - Mixed precision: bf16 (if available) or fp16

## Evaluation Metrics

The evaluation script computes:
- **Valid JSON rate**: Percentage of responses that are valid JSON
- **Valid position rate**: Percentage of valid chess positions
- **Exact JSON match**: Exact match with ground truth JSON
- **FEN accuracy**: Accuracy of FEN reconstruction from JSON
- **Per-square accuracy**: Average accuracy of piece predictions per square
- **Exact board match**: Percentage of positions with all 64 squares correct

## Comparison with Baseline

After fine-tuning, compare results with:
- CLIP-based JSON predictor (79.32% per-square accuracy)
- Baseline Qwen2-VL-2B without fine-tuning
- Other VLM approaches

## Notes

- The training uses LoRA for efficient fine-tuning, reducing memory requirements
- Image paths in the dataset should be relative to `image_base_dir`
- The model processes images at 224x224 resolution (Qwen2-VL default)
- For best results, use the full dataset (not limited samples)

