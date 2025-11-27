# Run benchmark from benchmarking directory (PowerShell)

python benchmark.py `
    --clip_checkpoint ../runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt `
    --images_dir ../data/hf_chess_puzzles/test/images `
    --fen_candidates ../data/hf_chess_puzzles/test.csv `
    --dataset_csv ../data/hf_chess_puzzles/test.csv `
    --output_dir benchmark_results

