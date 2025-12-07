# Quick Start Guide - Qwen2-VL-2B Fine-tuning

## Training is Running! 🚀

The training has been started in the background with the `pytorch_5070ti` conda environment.

## Monitor Training Progress

### Option 1: Check Checkpoint Directory
```bash
# Check if checkpoints are being created
Get-ChildItem Improved_representations/checkpoints/qwen2vl_json/ -Recurse | Select-Object Name, LastWriteTime | Format-Table

# Or use the monitoring script
python Improved_representations/vlm_finetuning/monitor_training.py
```

### Option 2: Check Training Logs
The training will create logs in the checkpoint directory. Look for:
- `trainer_state.json` - Training state and metrics
- `training.log` - Training output (if logging to file)
- `checkpoint-*/` - Model checkpoints

### Option 3: Check Process
```bash
# Check if Python process is running
Get-Process python | Where-Object {$_.Path -like "*pytorch_5070ti*"}
```

## Expected Timeline

- **Model Loading**: 2-5 minutes (first time, downloads model)
- **Dataset Loading**: 1-2 minutes
- **Training Start**: After initialization
- **First Checkpoint**: After 1000 steps (depends on dataset size and batch size)

## Training Configuration

- **Effective Batch Size**: 16 (batch_size=2 × gradient_accumulation=8)
- **Total Steps per Epoch**: ~6,250 steps (99,999 samples / 16)
- **Total Training Steps**: ~18,750 steps (3 epochs)
- **Checkpoint Frequency**: Every 1000 steps (~6 checkpoints per epoch)

## What to Expect

1. **Initial Phase** (0-5 min):
   - Model downloading/loading
   - Dataset preparation
   - LoRA setup

2. **Training Phase**:
   - Loss decreasing
   - Checkpoints saved every 1000 steps
   - Evaluation every 1000 steps

3. **Completion**:
   - Final model saved
   - Training metrics logged

## If Training Fails

Check for common issues:
1. **Out of Memory**: Reduce batch_size or gradient_accumulation_steps
2. **Missing Dependencies**: Install `peft` and `accelerate`
3. **HuggingFace Token**: Set `HF_TOKEN` environment variable if needed

## Next Steps After Training

Once training completes, evaluate the model:

```bash
conda activate pytorch_5070ti
python -m Improved_representations.vlm_finetuning.evaluate_qwen \
    --model_path Improved_representations/checkpoints/qwen2vl_json \
    --test_data Improved_representations/data/vlm_dataset/test.json \
    --image_base_dir data/hf_chess_puzzles \
    --output Improved_representations/results/qwen2vl_eval.json
```

