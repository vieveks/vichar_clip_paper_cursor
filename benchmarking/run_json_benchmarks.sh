#!/bin/bash
# Run benchmarks for JSON-based models (Exp 1A, 1B, 1C, 1D)
# Usage: ./run_json_benchmarks.sh

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="$SCRIPT_DIR"
PREDICTIONS_DIR="$PROJECT_ROOT/Improved_representations/results"
DATASET_CSV="$PROJECT_ROOT/data/hf_chess_puzzles/test.csv"
IMAGES_DIR="$PROJECT_ROOT/data/hf_chess_puzzles/test/images"
NUM_IMAGES=10
VLM_MODEL="gpt-4o"

echo "=============================================="
echo "JSON Model Benchmark Runner"
echo "=============================================="
echo "Project Root: $PROJECT_ROOT"
echo "Dataset: $DATASET_CSV"
echo "Images: $IMAGES_DIR"
echo "VLM Model: $VLM_MODEL"
echo "Num Images: $NUM_IMAGES"
echo "=============================================="

# Experiments to run
declare -A experiments
experiments["exp1a"]="$PREDICTIONS_DIR/predictions_clip_exp1a.jsonl"
experiments["exp1b"]="$PREDICTIONS_DIR/predictions_clip_exp1b.jsonl"
experiments["exp1c"]="$PREDICTIONS_DIR/predictions_qwen_exp1c.jsonl"
experiments["exp1d"]="$PREDICTIONS_DIR/predictions_clip_exp1d.jsonl"

for exp in "${!experiments[@]}"; do
    predictions_file="${experiments[$exp]}"
    
    echo ""
    echo "----------------------------------------------"
    echo "Running benchmark for: $exp"
    echo "----------------------------------------------"
    
    if [ -f "$predictions_file" ]; then
        echo "Predictions found: $predictions_file"
        
        python "$BENCHMARK_DIR/benchmark_json_models.py" \
            --predictions "$predictions_file" \
            --dataset_csv "$DATASET_CSV" \
            --images_dir "$IMAGES_DIR" \
            --vlm_model "$VLM_MODEL" \
            --num_images $NUM_IMAGES \
            --output_dir "$BENCHMARK_DIR/benchmark_results_json" \
            --experiment_name "$exp"
    else
        echo "Predictions not found: $predictions_file"
        echo "Skipping $exp"
    fi
done

echo ""
echo "=============================================="
echo "All benchmarks completed!"
echo "=============================================="
