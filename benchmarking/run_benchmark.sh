#!/bin/bash
# Run benchmark from benchmarking directory

# The default model is llava-hf/llava-v1.6-mistral-7b-hf
# If you have it downloaded locally, set LOCAL_VLM_PATH to use it faster
# export LOCAL_VLM_PATH="/path/to/your/llava-v1.6-mistral-7b-hf"

# Number of images to test (use 1 for quick test, 100 for full benchmark)
NUM_IMAGES=${NUM_IMAGES:-1}

if [ -z "$LOCAL_VLM_PATH" ]; then
    echo "Using HuggingFace model: llava-hf/llava-v1.6-mistral-7b-hf"
    echo "  (Set LOCAL_VLM_PATH to use local model for faster loading)"
    echo "Testing on $NUM_IMAGES image(s)"
    python benchmark.py \
        --clip_checkpoint ../runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt \
        --images_dir ../data/hf_chess_puzzles/test/images \
        --fen_candidates ../data/hf_chess_puzzles/test.csv \
        --output_dir benchmark_results \
        --dataset_csv ../data/hf_chess_puzzles/test.csv \
        --num_images $NUM_IMAGES
else
    echo "Using local LLaVA model: $LOCAL_VLM_PATH"
    echo "Testing on $NUM_IMAGES image(s)"
    python benchmark.py \
        --clip_checkpoint ../runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt \
        --images_dir ../data/hf_chess_puzzles/test/images \
        --fen_candidates ../data/hf_chess_puzzles/test.csv \
        --output_dir benchmark_results \
        --dataset_csv ../data/hf_chess_puzzles/test.csv \
        --local_vlm_path "$LOCAL_VLM_PATH" \
        --num_images $NUM_IMAGES
fi

