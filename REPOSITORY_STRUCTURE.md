# Repository Structure

This document describes the cleaned and organized structure of the Chess CLIP repository.

## Directory Structure

```
vichar_clip/
├── README.md                          # Main project README
├── LICENSE                            # License file
│
├── Notebooks/                         # Core training and inference code
│   ├── train_clip.py                 # Main training script
│   ├── inference.py                  # Inference script
│   ├── dataset_loader.py             # Dataset loading utilities
│   ├── dataset_prep.py               # Dataset preparation (main)
│   ├── dataset_prep_simple.py        # Simplified dataset prep
│   ├── comprehensive_evaluation.py   # Comprehensive evaluation script
│   │
│   ├── checkpoints/                  # Model checkpoints
│   │   └── large_1000/
│   │       ├── fen_only_model/       # FEN-only model checkpoints
│   │       └── fen_move_model/       # FEN+Move model checkpoints
│   │
│   └── large_datasets/                # Training datasets
│       ├── fen_only/                  # FEN-only dataset
│       └── fen_move/                  # FEN+Move dataset
│
├── docs/                              # Documentation files
│   ├── EXPERIMENTS_AND_RESULTS.md
│   ├── EXPERIMENTAL_VALIDATION_DOCUMENTATION.md
│   ├── PROJECT_COMPLETION_SUMMARY.md
│   ├── FUTURE_EXPERIMENTS.md
│   ├── chess_clip_paper.md
│   ├── ieee_chess_clip_paper.md
│   ├── ieee_chess_clip_paper_v2.md
│   └── results_individual_tests.md
│
├── data/                              # Data files
│   └── pgn_files/                     # PGN chess game files
│       ├── anand_pgns/
│       │   └── Anand.pgn
│       └── lichess_games_2013-01.pgn
│
├── testing_files/                     # Testing and evaluation files
│   ├── benchmark_*.py                 # Benchmark scripts
│   ├── test_*.py                      # Test scripts
│   ├── run_comprehensive_evaluation.py
│   ├── comprehensive_evaluation_results.json
│   ├── train_val_test_comparison.csv
│   ├── fresh_test_images/             # Fresh test images
│   ├── random_test_images/             # Random test images
│   ├── independent_test_set/          # Independent test set
│   └── *.png, *.csv, *.json           # Test results and images
│
├── utils/                             # Utility scripts
│   ├── download_pgn.py                # Download PGN files from Lichess
│   ├── check_data_independence.py     # Check data overlap
│   ├── dataset_prep_pillow.py         # Alternative dataset prep
│   └── Chess_zip.py                   # Archive utilities
│
└── runs/                              # Training run outputs
    └── clip_fen/
```

## Key Files for Training and Testing

### Training
- **`Notebooks/train_clip.py`** - Main training script
- **`Notebooks/dataset_prep.py`** - Prepare dataset from PGN files
- **`Notebooks/dataset_loader.py`** - Dataset loading utilities

### Inference
- **`Notebooks/inference.py`** - Run inference on chess board images

### Evaluation
- **`Notebooks/comprehensive_evaluation.py`** - Comprehensive evaluation with overfitting analysis
- **`testing_files/benchmark_*.py`** - Benchmark scripts
- **`testing_files/test_*.py`** - Various test scripts

### Utilities
- **`utils/download_pgn.py`** - Download PGN files from Lichess database
- **`utils/check_data_independence.py`** - Check for data overlap between datasets

## Quick Start

### 1. Prepare Dataset
```bash
cd Notebooks
python dataset_prep.py ../data/pgn_files/lichess_games_2013-01.pgn ./large_datasets/fen_only --max_games 1000
```

### 2. Train Model
```bash
python train_clip.py ./large_datasets/fen_only ./checkpoints/my_model --epochs 5 --batch_size 32
```

### 3. Run Inference
```bash
python inference.py --checkpoint_path ./checkpoints/my_model/clip_chess_epoch_5.pt --image_path path/to/board.png
```

### 4. Evaluate Model
```bash
python comprehensive_evaluation.py --model_path ./checkpoints/my_model/clip_chess_epoch_5.pt --data_dir ./large_datasets/fen_only
```

## Notes

- All documentation has been moved to `docs/`
- All test files and results are in `testing_files/`
- PGN files are organized in `data/pgn_files/`
- Utility scripts are in `utils/`
- Core training/inference code remains in `Notebooks/`

