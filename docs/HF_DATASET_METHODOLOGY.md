# Hugging Face Chess Dataset - Methodology and Documentation

## Dataset Information

**Dataset Name:** `bingbangboom/chess-puzzles-images-mini`  
**Source:** [Hugging Face Datasets](https://huggingface.co/datasets/bingbangboom/chess-puzzles-images-mini)  
**License:** CC0-1.0 (Public Domain)

### Dataset Statistics

- **Total Samples:** 125,000 chess positions
- **Source:** Derived from Lichess puzzles
- **Image Format:** JPG, 512x512 pixels
- **Pre-split Distribution:**
  - Training: 100,000 samples (80%)
  - Validation: 12,500 samples (10%)
  - Test: 12,500 samples (10%)

### Dataset Fields

1. **`image`**: Visual representation of the chess board (512x512 JPG)
2. **`board_state`**: Shortened FEN (Forsyth–Edwards Notation) string representing piece placement
3. **`active_color`**: Player to move ("w" for White, "b" for Black)
4. **`castling_rights`**: Remaining castling options ("K", "Q", "k", "q", or "-")
5. **`en_passant_target_square`**: En passant target square in algebraic notation, or "-"
6. **`best_continuation`**: Solution to the puzzle (best move sequence)

## Dataset Preparation

### Download Script

The dataset is downloaded and prepared using `utils/download_hf_chess_dataset.py`:

```bash
python utils/download_hf_chess_dataset.py \
    --output_dir data/hf_chess_puzzles \
    --format csv
```

This script:
- Downloads the dataset from Hugging Face
- Processes each split (train/validation/test)
- Saves images locally
- Creates CSV files with image paths and FEN strings
- Generates `dataset_metadata.json` with dataset statistics

### Output Structure

```
data/hf_chess_puzzles/
├── train.csv
├── validation.csv
├── test.csv
├── dataset_metadata.json
├── train/
│   └── images/
│       └── train_000000.png, train_000001.png, ...
├── validation/
│   └── images/
│       └── validation_000000.png, ...
└── test/
    └── images/
        └── test_000000.png, ...
```

## Training Methodology

### Split Strategy

We provide two split strategies:

1. **Native Splits (Recommended for Paper)**
   - Uses the dataset's original train/validation/test splits
   - Train: 100,000 samples
   - Validation: 12,500 samples
   - Test: 12,500 samples
   - Ensures reproducibility and comparability with other research

2. **Custom Splits**
   - Allows custom train/val/test ratios (default: 80/10/10)
   - Useful for experimentation and ablation studies
   - Uses fixed random seed (42) for reproducibility

### Training Configuration

**Model Architecture:**
- Base: CLIP (Contrastive Language-Image Pre-Training)
- Variant: ViT-B-32 (Vision Transformer Base, 32 patches)
- Pretrained: LAION-2B (2 billion image-text pairs)

**Training Hyperparameters:**
- Batch Size: 128 (adjustable)
- Learning Rate: 1e-4 (adjustable)
- Optimizer: AdamW
- Loss Function: CLIP Loss (contrastive loss)
- Mixed Precision: FP16 (optional, for faster training)
- Epochs: 10 (adjustable)

**Image Preprocessing:**
- Resize: 224x224 (CLIP standard)
- Normalization: CLIP ImageNet statistics
  - Mean: [0.48145466, 0.4578275, 0.40821073]
  - Std: [0.26862954, 0.26130258, 0.27577711]

### Training Script

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

### Evaluation Metrics

The training script tracks:
- Training loss per epoch
- Validation loss per epoch
- Best model checkpoint (lowest validation loss)
- Training history (saved as JSON)

## Documentation for Paper Publication

### Dataset Citation

If using this dataset in a paper, please cite:

```bibtex
@dataset{chess_puzzles_images_mini,
  title={Chess Puzzles Images (mini)},
  author={bingbangboom},
  year={2024},
  url={https://huggingface.co/datasets/bingbangboom/chess-puzzles-images-mini},
  license={CC0-1.0}
}
```

### Methodology Section Template

**Dataset:**
We use the `bingbangboom/chess-puzzles-images-mini` dataset, which contains 125,000 chess positions derived from Lichess puzzles. The dataset is pre-split into training (100,000 samples), validation (12,500 samples), and test (12,500 samples) sets. Each sample consists of a 512×512 pixel chess board image and its corresponding FEN (Forsyth–Edwards Notation) string.

**Training:**
We fine-tune a CLIP model (ViT-B-32 architecture) pretrained on LAION-2B. The model is trained using contrastive learning to map chess board images and FEN strings into a shared embedding space. Training is performed with a batch size of 128, learning rate of 1e-4, and AdamW optimizer. We use mixed precision (FP16) training for efficiency. The model is trained for 10 epochs, with the best model selected based on validation loss.

**Evaluation:**
Model performance is evaluated on the held-out test set using standard CLIP evaluation metrics (top-k accuracy, average rank, etc.).

### Reproducibility

All training configurations, dataset splits, and random seeds are saved in:
- `training_metadata.json`: Complete training configuration
- `training_history.json`: Training and validation losses per epoch
- `dataset_metadata.json`: Dataset statistics and information

These files ensure full reproducibility of experiments.

## Comparison with Previous Dataset

### Differences from Custom Dataset

1. **Size:** 125k samples vs. previous custom dataset size
2. **Source:** Lichess puzzles (real game positions) vs. generated positions
3. **Split:** Pre-defined splits vs. random splits
4. **Format:** Standardized Hugging Face format vs. custom format
5. **Metadata:** Includes additional fields (castling rights, en passant, best continuation)

### Advantages

- Larger dataset size
- Real game positions (more realistic)
- Standardized format (easier to share and reproduce)
- Pre-defined splits (ensures consistency)
- Additional metadata for future experiments

## Future Experiments

Potential experiments using this dataset:
1. Compare performance on real game positions vs. generated positions
2. Use `best_continuation` field for move prediction tasks
3. Experiment with full FEN construction using metadata fields
4. Compare different split strategies (native vs. custom)
5. Ablation studies on dataset size

