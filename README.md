# Vichar-CLIP: Chess Position Identification

This repository contains code for training and running a CLIP (Contrastive Language-Image Pre-Training) model to identify the FEN (Forsyth-Edwards Notation) representation of a chess position from a board image.

## Project Overview

The core idea is to train a model that can look at an image of a chessboard and accurately predict its corresponding FEN string. This is framed as an image-text matching problem, where board images and FEN strings are projected into a shared embedding space.

Two main experiments were conducted:
1.  **FEN Only:** The model is trained to match a board image to its exact FEN string.
2.  **FEN + Move:** The model is trained to match a board image to a text string containing both the FEN and the last move played (e.g., "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1 | e4").

## Results

The models were evaluated on a fresh test dataset, yielding the following performance metrics:

| Model Type | Top-1 Accuracy | Top-5 Accuracy | Top-10 Accuracy | Average Rank | Average Confidence | Median Rank |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FEN Only** | 96.67% | 100% | 100% | 1.07 | 40.88% | 1.0 |
| **FEN + Move**| 96.67% | 100% | 100% | 1.03 | 36.81% | 1.0 |

Both models achieve very high accuracy, demonstrating a strong capability to map board images to their correct FEN representations.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/vichar-clip.git
    cd vichar-clip
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install torch torchvision open_clip_torch pandas Pillow 'chess<2.0' cairosvg
    ```

## Usage

### Data Preparation

The training script expects a CSV file mapping image paths to their corresponding FEN strings. You will need to generate your own dataset of chess board images and a `fen_image_pairs.csv` file with `image_path` and `fen` columns.

### Training

To train a new model, use the `train_clip.py` script.

**Example:**
```bash
python train_clip.py \
    --csv path/to/your/fen_image_pairs.csv \
    --out_dir runs/my_clip_run \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-4 \
    --model ViT-B-32 \
    --fp16
```

**Arguments:**
*   `--csv`: (Required) Path to the FEN–image CSV file.
*   `--out_dir`: Directory to save checkpoints and logs.
*   `--epochs`: Number of training epochs.
*   `--batch_size`: Training batch size.
*   `--lr`: Learning rate.
*   `--model`: CLIP model architecture (e.g., `ViT-B-32`).
*   `--fp16`: Use mixed-precision training.
*   `--resume`: Path to a checkpoint to resume training from.

### Inference

To run inference, use the `inference_clip.py` script. You need a trained model checkpoint, a board image, and a CSV file with FEN candidates.

The script can also generate sample data for a quick demonstration.

**Example with your own data:**
```bash
python inference_clip.py \
    --checkpoint_path runs/my_clip_run/best_model.pt \
    --image_path path/to/your/board.png \
    --candidates_csv path/to/your/candidates.csv \
    --top_k 5
```

**Example with generated sample data:**
This will create `sample_chess_board.png` and `candidates.csv`.
```bash
python inference_clip.py --checkpoint_path runs/my_clip_run/best_model.pt
```
