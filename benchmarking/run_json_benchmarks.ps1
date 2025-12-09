# Run benchmarks for JSON-based models (Exp 1A, 1B, 1C, 1D)
# Usage: .\run_json_benchmarks.ps1

$ErrorActionPreference = "Stop"

# Configuration
$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$BENCHMARK_DIR = $PSScriptRoot
$PREDICTIONS_DIR = "$PROJECT_ROOT\Improved_representations\results"
$DATASET_CSV = "$PROJECT_ROOT\data\hf_chess_puzzles\test.csv"
$IMAGES_DIR = "$PROJECT_ROOT\data\hf_chess_puzzles\test\images"
$NUM_IMAGES = 10
$VLM_MODEL = "gpt-4o"

Write-Host "=============================================="
Write-Host "JSON Model Benchmark Runner"
Write-Host "=============================================="
Write-Host "Project Root: $PROJECT_ROOT"
Write-Host "Dataset: $DATASET_CSV"
Write-Host "Images: $IMAGES_DIR"
Write-Host "VLM Model: $VLM_MODEL"
Write-Host "Num Images: $NUM_IMAGES"
Write-Host "=============================================="

# Check if predictions exist
$experiments = @{
    "exp1a" = "$PREDICTIONS_DIR\predictions_clip_exp1a.jsonl"
    "exp1b" = "$PREDICTIONS_DIR\predictions_clip_exp1b.jsonl"
    "exp1c" = "$PREDICTIONS_DIR\predictions_qwen_exp1c.jsonl"
    "exp1d" = "$PREDICTIONS_DIR\predictions_clip_exp1d.jsonl"
}

foreach ($exp in $experiments.GetEnumerator()) {
    $name = $exp.Key
    $predictions_file = $exp.Value
    
    Write-Host "`n----------------------------------------------"
    Write-Host "Running benchmark for: $name"
    Write-Host "----------------------------------------------"
    
    if (Test-Path $predictions_file) {
        Write-Host "Predictions found: $predictions_file"
        
        python "$BENCHMARK_DIR\benchmark_json_models.py" `
            --predictions "$predictions_file" `
            --dataset_csv "$DATASET_CSV" `
            --images_dir "$IMAGES_DIR" `
            --vlm_model "$VLM_MODEL" `
            --num_images $NUM_IMAGES `
            --output_dir "$BENCHMARK_DIR\benchmark_results_json" `
            --experiment_name "$name"
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Warning: Benchmark failed for $name"
        }
    } else {
        Write-Host "Predictions not found: $predictions_file"
        Write-Host "Skipping $name"
    }
}

Write-Host "`n=============================================="
Write-Host "All benchmarks completed!"
Write-Host "=============================================="
