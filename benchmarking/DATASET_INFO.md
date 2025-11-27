# Benchmarking Dataset Information

## Recommended Dataset

For benchmarking the VLM performance, we use the **test split** from the Hugging Face chess puzzles dataset that was used for CLIP training.

### Dataset Details

- **Source:** `bingbangboom/chess-puzzles-images-mini` (Hugging Face)
- **Test Split Location:** `data/hf_chess_puzzles/test/`
- **Number of Images:** 12,500 test images
- **Image Format:** PNG/JPG, 512×512 pixels
- **FEN Data:** Available in `data/hf_chess_puzzles/test.csv`

### Why Use the Test Split?

1. **Held-Out Data:** The test set was not used during CLIP training, ensuring fair evaluation
2. **Ground Truth Available:** Each image has a corresponding FEN string in the CSV
3. **Best Moves Included:** The `best_continuation` field provides puzzle solutions (ground truth for best move questions)
4. **Consistent Format:** Same format as training data, ensuring compatibility
5. **Large Sample Size:** 12,500 images provide statistically meaningful results

### Dataset Structure

```
data/hf_chess_puzzles/test/
├── images/
│   ├── test_000000.png
│   ├── test_000001.png
│   └── ... (12,500 images total)
└── test.csv (contains image paths and FEN strings)
```

### CSV Format

The `test.csv` file contains:
- **`image_path`**: Path to the chess board image
- **`fen`**: FEN string for the position
- **`active_color`**: Player to move ("w" or "b")
- **`castling_rights`**: Available castling options
- **`en_passant_target_square`**: En passant target square
- **`best_continuation`**: **Best move sequence** (solution to the puzzle) - **This is used as ground truth for best move questions!**

### Using Best Continuation

The `best_continuation` field contains the puzzle solution, which is used as ground truth for the "best move" question. The benchmark system:
1. **First tries** to use `best_continuation` from the CSV (faster, more reliable)
2. **Falls back** to Lichess API if not found in dataset

This approach is:
- **Faster:** No API calls needed for positions in the dataset
- **More Reliable:** Uses the actual puzzle solutions (verified by Lichess)
- **Offline-capable:** Works without internet connection for dataset positions

### Usage in Benchmark

The benchmark script uses:
1. **Images:** From `data/hf_chess_puzzles/test/images/`
2. **FEN Candidates:** From `data/hf_chess_puzzles/test.csv` (for CLIP FEN extraction)
3. **Ground Truth:**
   - **Best moves:** From `best_continuation` field in CSV (primary source)
   - **Other answers:** Extracted from FEN using pychess and Lichess API
   - **Fallback:** Lichess API if best move not found in dataset

### Alternative Datasets

You can also benchmark on:
- **Custom images:** Provide your own chess board images
- **Validation set:** `data/hf_chess_puzzles/validation/` (12,500 images)
- **Training set:** `data/hf_chess_puzzles/train/` (99,999 images) - not recommended as it was used for training

### Benchmarking on Subset

For faster testing, you can use a subset:

```bash
# Test on first 10 images
python benchmarking/benchmark.py \
    --clip_checkpoint runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt \
    --images $(ls data/hf_chess_puzzles/test/images/*.png | head -10) \
    --fen_candidates data/hf_chess_puzzles/test.csv \
    --output_dir benchmark_results
```

### Dataset Statistics

- **Total Test Images:** 12,500
- **Image Resolution:** 512×512 pixels
- **Format:** PNG (converted from original JPG)
- **Source:** Lichess puzzles (real game positions)
- **License:** CC0-1.0 (Public Domain)
