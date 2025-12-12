#!/bin/bash
# Script to run all experiments with proper conda environment
# Usage: bash run_experiments.sh

# Set conda environment
CONDA_ENV="pytorch_5070ti"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

# Set Python path
PYTHON_PATH=$(which python)
echo "Using Python: $PYTHON_PATH"

# Log start time
echo "=== Experiment Run Started: $(date) ===" | tee -a neurosymbolic_pipeline/EXPERIMENT_LOG.md

# Experiment B: Symbolic Refinement
echo ""
echo "=========================================="
echo "Running Experiment B: Symbolic Refinement"
echo "=========================================="
cd neurosymbolic_pipeline/experiment_b
$PYTHON_PATH evaluate_refinement.py \
    --checkpoint ../../Improved_representations/checkpoints/json_predictor/best_model.pt \
    --test_data ../../Improved_representations/data/json_dataset/test.jsonl \
    --image_base_dir ../../data/hf_chess_puzzles \
    --max_samples 100 \
    --batch_size 16 \
    2>&1 | tee -a ../EXPERIMENT_LOG.md

# Experiment A: Stockfish CP Loss
echo ""
echo "=========================================="
echo "Running Experiment A: Stockfish CP Loss"
echo "=========================================="
cd ../experiment_a
$PYTHON_PATH evaluate_cp_loss.py \
    --predictions ../../Improved_representations/results/predictions_clip_exp1b.jsonl \
    --max_samples 100 \
    2>&1 | tee -a ../EXPERIMENT_LOG.md

# Experiment C: Hybrid Reasoning
echo ""
echo "=========================================="
echo "Running Experiment C: Hybrid Reasoning"
echo "=========================================="
cd ../experiment_c
$PYTHON_PATH evaluate_hybrid_reasoning.py \
    --test_data ../../data/hf_chess_puzzles/test.json \
    --max_samples 50 \
    2>&1 | tee -a ../EXPERIMENT_LOG.md

echo ""
echo "=== Experiment Run Completed: $(date) ===" | tee -a ../EXPERIMENT_LOG.md

