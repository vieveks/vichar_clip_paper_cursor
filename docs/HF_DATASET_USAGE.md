# Quick Start Guide: Training with Hugging Face Chess Dataset

## Step 1: Download the Dataset

First, download and prepare the dataset:

```bash
python utils/download_hf_chess_dataset.py \
    --output_dir data/hf_chess_puzzles \
    --format csv
```

This will:
- Download ~125k chess positions from Hugging Face
- Save images locally
- Create CSV files for train/validation/test splits
- Generate dataset metadata

**Expected time:** 10-30 minutes depending on internet speed  
**Disk space:** ~4.5 GB

## Step 2: Train the Model

### Using Native Splits (Recommended)

Use the dataset's original splits (100k/12.5k/12.5k):

```bash
python train_clip_hf_dataset.py \
    --data_dir data/hf_chess_puzzles \
    --split_method native \
    --model ViT-B-32 \
    --pretrained laion2B-s34B-b79K \
    --out_dir runs/clip_hf_chess \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-4 \
    --fp16 \
    --num_workers 4
```

### Using Custom Splits

Create custom train/val/test splits (e.g., 80/10/10):

```bash
python train_clip_hf_dataset.py \
    --data_dir data/hf_chess_puzzles \
    --split_method custom \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --split_seed 42 \
    --model ViT-B-32 \
    --pretrained laion2B-s34B-b79K \
    --out_dir runs/clip_hf_chess_custom \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-4 \
    --fp16
```

## Step 3: Monitor Training

Training logs are saved to:
- Console output (real-time)
- `training.log` file
- `runs/clip_hf_chess/training_history.json` (losses per epoch)

## Step 4: Check Results

After training, check the output directory:

```
runs/clip_hf_chess/
├── best_model.pt              # Best model (lowest validation loss)
├── checkpoint_epoch_*.pt      # Checkpoints for each epoch
├── training_metadata.json      # Complete training configuration
├── training_history.json       # Training/validation losses
└── training.log               # Training logs
```

## Command Line Arguments

### Dataset Arguments
- `--data_dir`: Directory containing train.csv, validation.csv, test.csv
- `--split_method`: `native` (use dataset splits) or `custom` (create custom splits)
- `--train_ratio`: Training set ratio (for custom splits, default: 0.8)
- `--val_ratio`: Validation set ratio (for custom splits, default: 0.1)
- `--test_ratio`: Test set ratio (for custom splits, default: 0.1)
- `--split_seed`: Random seed for custom splits (default: 42)

### Model Arguments
- `--model`: CLIP model architecture (default: `ViT-B-32`)
- `--pretrained`: Pretrained weights (default: `laion2B-s34B-b79K`)

### Training Arguments
- `--out_dir`: Output directory for checkpoints (default: `runs/clip_hf_chess`)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 1e-4)
- `--fp16`: Enable mixed precision training (faster, less memory)
- `--num_workers`: Data loader workers (default: 4)
- `--resume`: Path to checkpoint to resume from

## Example: Resume Training

If training is interrupted, resume from a checkpoint:

```bash
python train_clip_hf_dataset.py \
    --data_dir data/hf_chess_puzzles \
    --split_method native \
    --out_dir runs/clip_hf_chess \
    --resume runs/clip_hf_chess/checkpoint_epoch_5.pt \
    --epochs 10
```

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` (e.g., 64 or 32)
- Enable `--fp16` for mixed precision
- Reduce `--num_workers`

### Slow Training
- Increase `--batch_size` if memory allows
- Enable `--fp16`
- Increase `--num_workers` (but not more than CPU cores)

### Dataset Not Found
- Make sure you've run the download script first
- Check that `--data_dir` points to the correct directory
- Verify that train.csv, validation.csv, test.csv exist

## Next Steps

After training:
1. Evaluate the model on the test set
2. Compare with previous models trained on custom dataset
3. Analyze training history and metrics
4. Document results for paper publication

See `docs/HF_DATASET_METHODOLOGY.md` for detailed methodology and paper documentation guidelines.

