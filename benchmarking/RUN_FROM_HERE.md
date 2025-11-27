# Running Benchmark from This Directory

## Quick Command

### Windows (PowerShell)
```powershell
python benchmark.py --clip_checkpoint ../runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt --images_dir ../data/hf_chess_puzzles/test/images --fen_candidates ../data/hf_chess_puzzles/test.csv --dataset_csv ../data/hf_chess_puzzles/test.csv --output_dir benchmark_results
```

### Windows (Command Prompt)
```cmd
python benchmark.py --clip_checkpoint ../runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt --images_dir ../data/hf_chess_puzzles/test/images --fen_candidates ../data/hf_chess_puzzles/test.csv --dataset_csv ../data/hf_chess_puzzles/test.csv --output_dir benchmark_results
```

### Linux/Mac
```bash
python benchmark.py \
    --clip_checkpoint ../runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt \
    --images_dir ../data/hf_chess_puzzles/test/images \
    --fen_candidates ../data/hf_chess_puzzles/test.csv \
    --dataset_csv ../data/hf_chess_puzzles/test.csv \
    --output_dir benchmark_results
```

## Using Scripts

### Windows PowerShell
```powershell
.\run_benchmark.ps1
```

### Linux/Mac
```bash
chmod +x run_benchmark.sh
./run_benchmark.sh
```

## Results Location

Results will be saved in: `benchmarking/benchmark_results/`

Files created:
- `detailed_results.json`
- `results.csv`
- `summary.json`

## Options

### Use Mock VLM (for testing)
Add `--use_mock_vlm` flag:
```powershell
python benchmark.py --clip_checkpoint ../runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt --images_dir ../data/hf_chess_puzzles/test/images --fen_candidates ../data/hf_chess_puzzles/test.csv --output_dir benchmark_results --use_mock_vlm
```

### Test on Single Image
```powershell
python benchmark.py --clip_checkpoint ../runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt --images ../data/hf_chess_puzzles/test/images/test_000000.png --fen_candidates ../data/hf_chess_puzzles/test.csv --output_dir benchmark_results
```

### Specify LLaVA Model Path (if using local model)
```powershell
python benchmark.py --clip_checkpoint ../runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt --images_dir ../data/hf_chess_puzzles/test/images --fen_candidates ../data/hf_chess_puzzles/test.csv --vlm_model /path/to/your/llava/model --output_dir benchmark_results
```

