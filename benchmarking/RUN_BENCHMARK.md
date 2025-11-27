# Running the Benchmark

## Quick Command

```bash
python benchmarking/benchmark.py \
    --clip_checkpoint runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt \
    --images_dir data/hf_chess_puzzles/test/images \
    --fen_candidates data/hf_chess_puzzles/test.csv \
    --output_dir benchmark_results
```

## What It Does (Brief Overview)

### Step-by-Step Process

1. **Loads CLIP Model**: Loads your trained CLIP model to extract FEN from images

2. **For Each Image**:
   - **Extracts FEN**: Uses CLIP to predict FEN from the chess board image
   - **Gets Ground Truth**: Uses the predicted FEN (not ground truth!) to get correct answers via:
     - pychess (for board state: piece locations, check status, etc.)
     - Lichess API (for engine analysis: best moves, evaluations)

3. **Tests VLM Twice** (for each question):
   - **Without FEN**: VLM answers using only the image + question
   - **With FEN**: VLM answers using image + question + predicted FEN context

4. **Scores Responses**: Compares VLM answers to ground truth (0-1 score)

5. **Generates Report**: Creates summary showing improvement when FEN is provided

### Key Points

- ✅ Uses **CLIP-predicted FEN** (not ground truth from dataset)
- ✅ Tests full pipeline: Image → CLIP → FEN → Ground Truth → VLM
- ✅ Compares VLM performance with vs without FEN context
- ✅ Generates detailed results and summary statistics

## Output Files

Results are saved in `benchmark_results/`:
- `detailed_results.json` - Complete results for each test
- `results.csv` - CSV format for analysis
- `summary.json` - Summary statistics

## Example Output

```
BENCHMARK SUMMARY
============================================================
Total images tested: 10
Total questions: 6
Total tests: 60

Overall Performance:
  Average score without FEN: 0.450
  Average score with FEN: 0.720
  Average improvement: +0.270
  Improvement percentage: +60.0%
```

## Options

### Use Mock VLM (for testing without loading models)

```bash
python benchmarking/benchmark.py \
    --clip_checkpoint runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt \
    --images_dir data/hf_chess_puzzles/test/images \
    --fen_candidates data/hf_chess_puzzles/test.csv \
    --use_mock_vlm \
    --output_dir benchmark_results
```

### Test on Single Image

```bash
python benchmarking/benchmark.py \
    --clip_checkpoint runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt \
    --images data/hf_chess_puzzles/test/images/test_000000.png \
    --fen_candidates data/hf_chess_puzzles/test.csv \
    --output_dir benchmark_results
```

### Test on Subset (first 10 images)

```bash
python benchmarking/benchmark.py \
    --clip_checkpoint runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt \
    --images $(ls data/hf_chess_puzzles/test/images/*.png | head -10) \
    --fen_candidates data/hf_chess_puzzles/test.csv \
    --output_dir benchmark_results
```

