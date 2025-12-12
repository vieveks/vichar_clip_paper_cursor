# PowerShell script to run all experiments with proper conda environment
# Usage: .\run_experiments.ps1

# Set conda environment
$CONDA_ENV = "pytorch_5070ti"
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $PROJECT_ROOT

# Activate conda environment
Write-Host "Activating conda environment: $CONDA_ENV"
& conda activate $CONDA_ENV

# Get Python path from conda environment
$CONDA_BASE = conda info --base
$PYTHON_PATH = Join-Path $CONDA_BASE "envs\$CONDA_ENV\python.exe"
if (-not (Test-Path $PYTHON_PATH)) {
    # Try alternative path
    $PYTHON_PATH = (conda run -n $CONDA_ENV python -c "import sys; print(sys.executable)").Trim()
}

# Verify Python path exists
if (-not (Test-Path $PYTHON_PATH)) {
    Write-Host "Error: Python not found at $PYTHON_PATH"
    Write-Host "Please verify conda environment '$CONDA_ENV' exists"
    exit 1
}

Write-Host "Using Python: $PYTHON_PATH"

# Log file
$LOG_FILE = Join-Path $PSScriptRoot "EXPERIMENT_LOG.md"

# Log start time
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $LOG_FILE -Value "`n=== Experiment Run Started: $timestamp ==="

# Experiment B: Symbolic Refinement
Write-Host ""
Write-Host "=========================================="
Write-Host "Running Experiment B: Symbolic Refinement"
Write-Host "=========================================="
Set-Location (Join-Path $PSScriptRoot "experiment_b")
& $PYTHON_PATH evaluate_refinement.py `
    --checkpoint ../../Improved_representations/checkpoints/json_predictor/best_model.pt `
    --test_data ../../Improved_representations/data/json_dataset/test.jsonl `
    --image_base_dir ../../data/hf_chess_puzzles `
    --max_samples 100 `
    --batch_size 16 `
    2>&1 | Tee-Object -FilePath (Join-Path $PSScriptRoot "EXPERIMENT_LOG.md") -Append

# Experiment A: Stockfish CP Loss
Write-Host ""
Write-Host "=========================================="
Write-Host "Running Experiment A: Stockfish CP Loss"
Write-Host "=========================================="
Set-Location (Join-Path $PSScriptRoot "experiment_a")
& $PYTHON_PATH evaluate_cp_loss.py `
    --predictions ../../Improved_representations/results/predictions_clip_exp1b.jsonl `
    --max_samples 100 `
    2>&1 | Tee-Object -FilePath (Join-Path $PSScriptRoot "EXPERIMENT_LOG.md") -Append

# Experiment C: Hybrid Reasoning
Write-Host ""
Write-Host "=========================================="
Write-Host "Running Experiment C: Hybrid Reasoning"
Write-Host "=========================================="
Set-Location (Join-Path $PSScriptRoot "experiment_c")
& $PYTHON_PATH evaluate_hybrid_reasoning.py `
    --test_data ../../data/hf_chess_puzzles/test.json `
    --max_samples 50 `
    2>&1 | Tee-Object -FilePath (Join-Path $PSScriptRoot "EXPERIMENT_LOG.md") -Append

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $LOG_FILE -Value "`n=== Experiment Run Completed: $timestamp ==="

Write-Host ""
Write-Host "All experiments completed. Check EXPERIMENT_LOG.md for details."

