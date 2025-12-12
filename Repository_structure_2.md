# Repository Structure 2: Detailed Experiment Documentation

This document provides a comprehensive mapping of all experiments conducted in the repository, their locations, training logs, checkpoints, and results. This is based on analysis of the codebase, paper draft_v7.tex, and all training artifacts.

**Last Updated**: Based on draft_v7.tex and repository analysis

---

## Table of Contents

1. [Overview](#overview)
2. [Approach 1: Retrieval-Based FEN Matching](#approach-1-retrieval-based-fen-matching)
3. [Approach 2: Generative FEN Prediction](#approach-2-generative-fen-prediction)
4. [Approach 2.5: JSON-First Hierarchical Representation](#approach-25-json-first-hierarchical-representation)
5. [Approach 2.6: Downstream Reasoning with JSON-Predicted FEN](#approach-26-downstream-reasoning-with-json-predicted-fen)
6. [Approach 2.7: Direct Vision-Tower Integration](#approach-27-direct-vision-tower-integration)
7. [Approach 3: LLM-Based FEN Extraction](#approach-3-llm-based-fen-extraction)
8. [VLM Evaluation with FEN Context](#vlm-evaluation-with-fen-context)
9. [Training Logs and Artifacts](#training-logs-and-artifacts)
10. [Results and Evaluation Files](#results-and-evaluation-files)

---

## Overview

The repository implements multiple approaches to bridge the vision-text modality gap for chess position understanding:

- **Approach 1**: Retrieval-based CLIP fine-tuning (99.98% accuracy, closed-world)
- **Approach 2**: Generative FEN decoder (0% exact match, exposure bias)
- **Approach 2.5**: JSON-first hierarchical representation (79.32% per-square accuracy)
- **Approach 2.6**: Downstream reasoning evaluation with predicted FEN
- **Approach 2.7**: CLIP as vision encoder for LLaVA
- **Approach 3**: LLM-based FEN extraction (94% accuracy, open-world)

---

## Approach 1: Retrieval-Based FEN Matching

**Paper Section**: 2.1 (Subsection: Approach 1: Retrieval-Based FEN Matching)

**Objective**: Fine-tune CLIP ViT-B/32 on chess board image-FEN pairs for retrieval-based matching.

### Training Scripts

- **Main Training Script**: `train_clip_hf_dataset_fast.py` (root directory)
- **Alternative Script**: `train_clip_hf_dataset.py` (if exists)
- **Dataset Loader**: `utils/hf_chess_dataset_loader.py`

### Training Configuration

- **Model**: CLIP ViT-B/32 (pretrained: laion2B-s34B-b79K)
- **Dataset**: Hugging Face chess puzzles dataset (`data/hf_chess_puzzles/`)
  - Training: 99,999 samples
  - Validation: 12,500 samples
  - Test: 12,500 samples
- **Hyperparameters**:
  - Epochs: 20
  - Batch size: 256
  - Learning rate: 5e-5
  - Optimizer: AdamW
  - Mixed precision: FP16 enabled
  - Gradient clipping: max_norm=1.0

### Training Runs and Checkpoints

| Run Name | Location | Best Epoch | Status | Notes |
|----------|----------|------------|--------|-------|
| `clip_hf_chess_100k_20epochs_fixed` | `runs/clip_hf_chess_100k_20epochs_fixed/` | 15 | ✅ Complete | Best model (99.98% top-1 accuracy) |
| `clip_hf_chess_100k_20epochs` | `runs/clip_hf_chess_100k_20epochs/` | - | ⚠️ Earlier run | May have NaN issues |
| `clip_hf_chess_50k` | `runs/clip_hf_chess_50k/` | - | ✅ Complete | Smaller dataset run |
| `clip_hf_chess_fast` | `runs/clip_hf_chess_fast/` | - | ✅ Complete | Fast iteration (3 epochs) |

### Training Logs

- **Main Log**: `training.log` (root directory)
- **Fast Training Log**: `training_fast.log` (root directory)
- **Training History**: `runs/clip_hf_chess_100k_20epochs_fixed/training_history.json`
- **Training Metadata**: `runs/clip_hf_chess_100k_20epochs_fixed/training_metadata.json`

### Results

- **Test Loss**: 0.000878 (vs. 6.049330 for base model)
- **Top-1 Accuracy**: 99.98% (vs. 0.38% for base model)
- **Top-5 Accuracy**: 100.00%
- **Top-10 Accuracy**: 100.00%

### Analysis Documents

- **Training Analysis**: `analysis_after_training_on_puzzle_dataset/TRAINING_ANALYSIS.md`
- **Visualizations**: `analysis_after_training_on_puzzle_dataset/*.png`
  - `training_curves.png`
  - `training_curves_zoomed.png`
  - `loss_gap.png`
  - `model_comparison.png`
  - `accuracy_comparison.png`

### Evaluation Scripts

- **Accuracy Measurement**: `benchmarking/measure_clip_accuracy.py`
- **Checkpoint Inspection**: `benchmarking/inspect_checkpoint.py`

---

## Approach 2: Generative FEN Prediction

**Paper Section**: 2.2 (Subsection: Approach 2: Generative FEN Prediction)

**Objective**: Generate FEN strings autoregressively using a Transformer decoder attached to CLIP encoder.

### Training Scripts

- **Main Training Script**: `FEN_generator/train.py`
- **Evaluation Script**: `FEN_generator/evaluate.py`
- **Debug Scripts**: 
  - `FEN_generator/debug_generation.py`
  - `FEN_generator/debug_overfit.py`
  - `FEN_generator/test_expand_fen.py`

### Model Architecture

- **Encoder**: CLIP ViT-B/32 (initialized from Approach 1 fine-tuned weights)
- **Decoder**: Transformer Decoder (6 layers, 8 heads, 512 dimensions)
- **Tokenizer**: Character-level tokenizer (`FEN_generator/tokenizer.py`)

### Training Strategy

**Two-Stage Training**:
1. **Stage 1**: Freeze encoder, train decoder only
2. **Stage 2**: Fine-tune entire encoder-decoder end-to-end

### Training Runs and Checkpoints

| Run Name | Location | Stage | Status | Notes |
|----------|----------|-------|--------|-------|
| `fen_generator` | `runs/fen_generator/` | Stage 1 | ✅ Complete | Initial attempt (pooled features) |
| `fen_generator_v2` | `runs/fen_generator_v2/` | Stage 1 | ✅ Complete | **Spatial patch embeddings** (best) |
| `fen_generator_v3` | `runs/fen_generator_v3/` | Stage 1 | ✅ Complete | Length penalty experiment |
| `fen_generator_v4_spatial_fix` | `runs/fen_generator_v4_spatial_fix/` | Stage 1 | ✅ Complete | Spatial fix variant |
| `fen_generator_v4_spatial_fix_10epochs` | `runs/fen_generator_v4_spatial_fix_10epochs/` | Stage 1 | ✅ Complete | Extended training |

### Training Logs

- **Main Log**: `fen_training.log` (root directory)
- **Run-Specific Logs**: 
  - `runs/fen_generator/training.log`
  - `runs/fen_generator_v2/training.log`
  - `runs/fen_generator_v3/training.log`
  - `runs/fen_generator_v4_spatial_fix/training.log`
  - `runs/fen_generator_v4_spatial_fix_10epochs/training.log`

### Results

- **Validation Loss**: 0.0084 (v3, excellent)
- **Exact Match Accuracy**: 0% (0/12,500)
- **Average CER**: 0.6663 (v2), 1.16 (v3 - worse)
- **Issue**: Exposure bias - errors compound after ~10-15 tokens

### Documentation

- **Updates**: `FEN_generator/UPDATES.md` (comprehensive training history)
- **Spatial Alignment Fix**: `FEN_generator/SPATIAL_ALIGNMENT_FIX.md`

### Key Findings

1. **v1**: Used pooled CLIP features → 0% accuracy (lacked spatial info)
2. **v2**: Switched to spatial patch embeddings (7×7 = 49 tokens) → Better loss (0.0225) but still 0% exact match
3. **v3**: Added length penalty → Overcorrected, CER worsened to 1.16
4. **Root Cause**: Exposure bias - model never learns to recover from its own errors

---

## Approach 2.5: JSON-First Hierarchical Representation

**Paper Section**: 2.3 (Subsection: Approach 2.5: JSON-First Hierarchical Representation)

**Objective**: Predict structured board state through parallel per-square classification, avoiding autoregressive generation.

### Experiments Conducted

Four ablation experiments (Exp 1A, 1B, 1C, 1D):

| Experiment | Architecture | Encoder State | Per-Square Acc | Exact Match | Paper Reference |
|------------|--------------|---------------|----------------|-------------|-----------------|
| **Exp 1A** | Base CLIP, Frozen | Frozen | 79.31% | 0.008% | Table 1 |
| **Exp 1B** | Fine-tuned CLIP, Frozen | Frozen | **79.32%** | 0.008% | Table 1 (Best) |
| **Exp 1C** | Qwen2-VL-2B (LoRA) | Fine-tuned | 43.55% | 0.00% | Table 1 |
| **Exp 1D** | Base CLIP, Unfrozen | Unfrozen | 79.13% | 0.02% | Table 1 |

### Training Scripts

#### CLIP-Based Experiments (1A, 1B, 1D)

- **Grid Predictor Training**: `Improved_representations/grid_predictor/train.py`
- **JSON Predictor Training**: `Improved_representations/json_predictor/train.py`
- **Evaluation**: 
  - `Improved_representations/grid_predictor/evaluate.py`
  - `Improved_representations/json_predictor/evaluate.py`

#### VLM Fine-tuning (Exp 1C)

- **Training Script**: `Improved_representations/vlm_finetuning/train_qwen.py`
- **Evaluation Script**: `Improved_representations/vlm_finetuning/evaluate_qwen.py`
- **Dataset Creation**: `Improved_representations/vlm_finetuning/create_vlm_dataset.py`
- **Monitoring**: `Improved_representations/vlm_finetuning/monitor_training.py`

### Checkpoints

#### Exp 1A (Base CLIP, Frozen)
- **Location**: `Improved_representations/checkpoints/exp1a_base_frozen/`
- **Best Model**: `best_model.pt`
- **Checkpoints**: `checkpoint_epoch_5.pt`, `checkpoint_epoch_10.pt`, `checkpoint_epoch_15.pt`, `final_model.pt`

#### Exp 1B (Fine-tuned CLIP, Frozen) ⭐ Best
- **Location**: `Improved_representations/checkpoints/exp1b_finetuned_frozen/` (inferred)
- **Best Model**: `best_model.pt`
- **Best Epoch**: 11
- **Results**: 79.32% per-square accuracy

#### Exp 1C (Qwen2-VL-2B LoRA)
- **Location**: `Improved_representations/checkpoints/qwen2vl_json/`
- **LoRA Adapters**: `adapter_model.safetensors`
- **Checkpoint**: `checkpoint-189/` (189 steps, 3 epochs)
- **Training Args**: `training_args.bin`
- **Status**: Complete (see `TRAINING_STATUS.md`)

#### Exp 1D (Base CLIP, Unfrozen)
- **Location**: `Improved_representations/checkpoints/exp1d_base_unfrozen/`
- **Best Model**: `best_model.pt`
- **Checkpoints**: `checkpoint_epoch_5.pt`, `checkpoint_epoch_10.pt`, `checkpoint_epoch_15.pt`, `final_model.pt`

### Training Configuration

#### CLIP-Based (1A, 1B, 1D)
- **Batch Size**: 32
- **Learning Rate**: 1e-4 with cosine annealing
- **Epochs**: 15-20 (early stopping)
- **Loss**: Cross-entropy for per-square + binary cross-entropy for metadata

#### Qwen2-VL (1C)
- **Model**: Qwen2-VL-2B-Instruct
- **Fine-tuning**: LoRA (r=16, alpha=32, dropout=0.1)
- **Batch Size**: 2 (gradient accumulation: 8, effective: 16)
- **Learning Rate**: 2e-4
- **Epochs**: 3 (189 steps)
- **Training Loss**: 43.64 → 16.50

### Results Files

- **Full Ablation Comparison**: `Improved_representations/results/full_ablation_comparison.json`
- **Individual Results**:
  - `Improved_representations/results/exp1a_results.json`
  - `Improved_representations/results/exp1b_results.json`
  - `Improved_representations/results/exp1d_results.json`
  - `Improved_representations/results/qwen2vl_eval.json`
- **Comparison**: `Improved_representations/results/comparison_1a_vs_1b.json`

### Predictions Files

- **Exp 1A Predictions**: `Improved_representations/results/predictions_clip_exp1a.jsonl`
- **Exp 1B Predictions**: `Improved_representations/results/predictions_clip_exp1b.jsonl`
- **Exp 1C Predictions**: `Improved_representations/results/predictions_qwen_exp1c.jsonl` (if exists)
- **Exp 1D Predictions**: `Improved_representations/results/predictions_clip_exp1d.jsonl`

### Documentation

- **Main README**: `Improved_representations/README.md`
- **Plan**: `Improved_representations/PLAN.md`, `Improved_representations/PLAN_JSON_FIRST.md`
- **Updates**: `Improved_representations/updates_improved_representations.md`
- **Report**: `Improved_representations/report.md`
- **VLM Training Status**: `Improved_representations/vlm_finetuning/TRAINING_STATUS.md`
- **VLM README**: `Improved_representations/vlm_finetuning/README.md`

### Key Findings

1. **CLIP-based (1A, 1B, 1D)**: Achieve ~79% per-square accuracy, significantly better than generative approach
2. **Chess fine-tuning impact (1A vs 1B)**: Minimal (+0.01%) when encoder frozen
3. **End-to-end training (1A vs 1D)**: Actually hurts performance (-0.18%)
4. **VLM fine-tuning (1C)**: Struggles (43.55% accuracy) - autoregressive generation not suited for structured spatial tasks

---

## Approach 2.6: Downstream Reasoning with JSON-Predicted FEN

**Paper Section**: 2.4 (Subsection: Approach 2.6: Downstream Reasoning with JSON-Predicted FEN)

**Objective**: Evaluate whether predicted FEN from JSON-first models improves downstream VLM reasoning.

### Benchmarking Scripts

- **Main Script**: `benchmarking/benchmark_json_models.py`
- **Runner Scripts**:
  - `benchmarking/run_json_benchmarks.ps1` (Windows PowerShell)
  - `benchmarking/run_json_benchmarks.sh` (Linux/Mac)
- **Documentation**: `benchmarking/BENCHMARK_JSON_MODELS.md`

### Evaluation Setup

- **VLM**: GPT-4o
- **Test Images**: 10 images from test set
- **Question Types**: 8 (FEN extraction, piece count, check status, material balance, best move, tactical pattern, castling rights, piece location)
- **Conditions**: No FEN, Predicted FEN, Ground Truth FEN

### Results Directories

- **Exp 1A**: `benchmarking/benchmark_results_exp1a/`
  - `summary.json`
  - `detailed_results.json`
- **Exp 1B**: `benchmarking/benchmark_results_exp1b/` (if exists)
- **Exp 1C**: `benchmarking/benchmark_results_qwen/`
  - `summary.json`
  - `detailed_results.json`
- **Exp 1D**: `benchmarking/benchmark_results_exp1d/`
  - `summary.json`
  - `detailed_results.json`

### Results Summary (from Paper)

| FEN Source | Avg Score | Accuracy | vs No FEN |
|------------|-----------|----------|-----------|
| No FEN (baseline) | 0.126 | 5.0% | --- |
| Exp 1A Predicted FEN | 0.178 | 7.5% | +41.3% |
| **Exp 1B Predicted FEN** | **0.182** | **8.1%** | **+44.4%** ⭐ |
| Exp 1C Predicted FEN | 0.142 | 5.6% | +12.7% |
| Exp 1D Predicted FEN | 0.175 | 7.3% | +38.9% |
| Ground Truth FEN | 0.212 | 8.75% | +68.3% |

### Key Findings

- Predicted FEN improves reasoning by 44.4% (best model: Exp 1B)
- CLIP-based models outperform VLM fine-tuning (44.4% vs 12.7%)
- Achieves 65% of oracle improvement (44.4% vs 68.3%)

---

## Approach 2.7: Direct Vision-Tower Integration

**Paper Section**: 2.5 (Subsection: Approach 2.7: Direct Vision-Tower Integration)

**Objective**: Substitute LLaVA's vision tower with trained CLIP encoders to test direct integration.

### Experiment Scripts

- **Training Script**: `clip_as_encoder/train.py`
- **Evaluation Script**: `clip_as_encoder/evaluate.py`
- **Quick Test**: `clip_as_encoder/quick_test.py`
- **Combine Results**: `clip_as_encoder/combine_results.py`

### Model Adaptation

- **Base Model**: LLaVA-v1.6-Mistral-7B
- **Vision Encoder Replacement**: CLIP ViT-B/32 (768-dim) → LLaVA projector (1024-dim)
- **Adaptations**:
  - Projector slicing to match dimensions
  - Single-crop alignment (336×336 → 24×24 grid, 576 tokens)

### Evaluation Results

- **Location**: `clip_as_encoder/evaluation_results/`
- **Files**:
  - `baseline_results.json`
  - `chess_clip_results.json`
  - `combined_results.json`
  - `comparison_summary.json`
  - `evaluation_results.json`
  - `EXPERIMENT_SUMMARY.md`

### Results Summary (from Paper)

| Model | Avg. Score | Accuracy (%) |
|-------|------------|--------------|
| Qwen2-VL-Finetuned | 0.11 | 3.75 |
| LLaVA + CLIP (Exp 1A, frozen) | 0.075 | 0.0 |
| LLaVA + CLIP (Exp 1D, unfrozen) | 0.075 | 0.0 |

### Documentation

- **Experiment Description**: `clip_as_encoder/EXPERIMENT_DESCRIPTION.md`
- **README**: `clip_as_encoder/README.md`
- **Enhanced README**: `clip_as_encoder/README_ENHANCED.md`

### Key Findings

- Zero-shot integration achieves partial scores but 0% exact accuracy
- Fine-tuning projector is essential for semantic alignment
- Qwen2-VL (end-to-end trained) achieves marginally better results (3.75%)

---

## Approach 3: LLM-Based FEN Extraction

**Paper Section**: 2.6 (Subsection: Approach 3: LLM-Based FEN Extraction)

**Objective**: Extract FEN from chess board images using state-of-the-art VLMs (GPT-4o, Claude, Gemini).

### Pipeline Components

- **Main Processor**: `page_fen_pipeline/page_fen_processor.py`
- **CLI Interface**: `page_fen_pipeline/cli.py`
- **Board Extractor**: `page_fen_pipeline/board_extractor.py`
- **FEN Generator**: 
  - `page_fen_pipeline/fen_generator.py` (simple)
  - `page_fen_pipeline/fen_generator_enhanced.py` (enhanced/consensus)
- **Model Providers**: `page_fen_pipeline/model_providers.py`
- **Example Usage**: `page_fen_pipeline/example_usage.py`

### Accuracy Strategies

1. **Simple Strategy**: Direct API call, basic prompt
   - Accuracy: 78%
   - Speed: 2.5s/board
   - Cost: $0.015/board

2. **Enhanced Strategy**: Image preprocessing + detailed prompts
   - Accuracy: 88%
   - Speed: 3.0s/board
   - Cost: $0.015/board

3. **Consensus Strategy**: 3-5 independent attempts + majority voting
   - Accuracy: 94%
   - Speed: 7.5s/board
   - Cost: $0.045/board

### Results Files

- **GPT-4o**: `page_fen_pipeline/rgpt-4o.json`
- **GPT-5**: `page_fen_pipeline/results_gpt-5.json`
- **Claude 4.1 Opus**: `page_fen_pipeline/results_claude-4.1-opus.json`
- **Claude 4.5 Sonnet**: `page_fen_pipeline/results_claude-4.5-sonnet.json`
- **Claude 4.5 Haiku**: `page_fen_pipeline/results_claude-4.5-haiku.json`
- **Gemini 2.5 Pro**: `page_fen_pipeline/results_gemini-2.5-pro.json`
- **Gemini 2.5 Flash**: `page_fen_pipeline/results_gemini-2.5-flash.json`
- **Output**: `page_fen_pipeline/output/results.json`

### Test Data

- **Test Boards**: `page_fen_pipeline/test_boards/`
  - `page_006/board_1.png`
  - `page_008/board_1.png` through `board_6.png`
- **Book Pages**: `page_fen_pipeline/book2/`
  - `page_006/`, `page_008/`

### Documentation

- **Main README**: `page_fen_pipeline/README.md`
- **Getting Started**: `page_fen_pipeline/GETTING_STARTED.md`
- **CLI Guide**: `page_fen_pipeline/CLI_GUIDE.md`
- **Multi-Model Guide**: `page_fen_pipeline/MULTI_MODEL_GUIDE.md`
- **Accuracy Improvements**: `page_fen_pipeline/ACCURACY_IMPROVEMENTS.md`
- **Accuracy Summary**: `page_fen_pipeline/ACCURACY_SUMMARY.md`
- **Changelog**: `page_fen_pipeline/CHANGELOG.md`
- **Folder Structure**: `page_fen_pipeline/FOLDER_STRUCTURE.md`
- **Quick Reference**: `page_fen_pipeline/QUICK_REFERENCE.md`
- **GPT-5 Debug**: `page_fen_pipeline/GPT5_DEBUG_GUIDE.md`, `page_fen_pipeline/GPT5_FIX_SUMMARY.md`

### Key Findings

- Consensus strategy achieves 94% accuracy on open-world images
- Enhanced strategy provides good balance (88% accuracy, same cost as simple)
- Multiple VLM providers tested (Claude 3.5 Sonnet often outperforms GPT-4o)

---

## VLM Evaluation with FEN Context

**Paper Section**: 2.7 (Subsection: VLM Evaluation with FEN Context)

**Objective**: Evaluate impact of FEN context on VLM performance using GPT-4o and Qwen2-VL-2B.

### Benchmarking Scripts

- **Main Benchmark**: `benchmarking/benchmark.py`
- **V2 Benchmark**: `benchmarking/benchmark_v2.py`
- **Improved Models Benchmark**: `benchmarking/benchmark_improved_models.py`
- **VLM Integration**: `benchmarking/vlm_integration.py`
- **LLM Judge Scorer**: `benchmarking/llm_judge_scorer.py`
- **Questions**: `benchmarking/questions.py`
- **Scoring**: `benchmarking/scoring.py`

### Question Types (15 total, expanded from 8)

1. FEN Extraction
2. Piece Count
3. Check Status
4. Material Balance
5. Material Advantage (NEW)
6. Material Count (White) (NEW)
7. Material Count (Black) (NEW)
8. Queen Count (NEW)
9. Minor Piece Balance (NEW)
10. Rook Count (NEW)
11. Pawn Advantage (NEW)
12. Best Move
13. Tactical Patterns
14. Castling Rights
15. Piece Location

### Results Directories

- **Baseline**: `benchmarking/benchmark_results/`
  - `summary.json`
  - `detailed_results.json`
- **V2**: `benchmarking/benchmark_results_v2/`
  - `summary.json`
  - `detailed_results.json`
- **Improved Models**: `benchmarking/benchmark_results_improved/`
  - `summary.json`
  - `detailed_results.json`

### Evaluation Results (Root)

- **Baseline**: `evaluation_results/baseline_results.json`
- **Chess CLIP**: `evaluation_results/chess_clip_results.json`
- **Comparison**: `evaluation_results/comparison_summary.json`
- **Qwen with FEN**: `evaluation_results/qwen_with_fen/`
  - `evaluation_results.json`
  - `chess_clip_results.json`

### GPT-4o Results (from Paper)

| Question Type | Visual-Only | + FEN | Improvement |
|---------------|-------------|-------|-------------|
| FEN Extraction | 0.13 | 0.46 | +254% |
| Check Status | 0.05 | 0.40 | +700% |
| Material Balance | 0.27 | 0.30 | +11% |
| Piece Count | 0.30 | 0.30 | 0% |
| Best Move | 0.26 | 0.24 | -8% |
| Tactical Pattern | 0.00 | 0.00 | 0% |
| Castling Rights | 0.00 | 0.00 | 0% |
| Piece Location | 0.00 | 0.00 | 0% |
| **Overall Average** | **0.126** | **0.212** | **+68.3%** |

### Qwen2-VL-2B Results (from Paper)

| Setting | Average Score | Accuracy | Improvement |
|---------|---------------|----------|-------------|
| Visual-Only (No FEN) | 0.273 | 20.00% | --- |
| With FEN Context | 0.307 | 26.67% | +12.5% score, +33.4% accuracy |

### Documentation

- **Main README**: `benchmarking/README.md`
- **Run Benchmark**: `benchmarking/RUN_BENCHMARK.md`
- **Run From Here**: `benchmarking/RUN_FROM_HERE.md`
- **Quick Start**: `benchmarking/QUICK_START.md`
- **Dataset Info**: `benchmarking/DATASET_INFO.md`
- **Implementation Notes**: `benchmarking/IMPLEMENTATION_NOTES.md`
- **Setup Local Model**: `benchmarking/SETUP_LOCAL_MODEL.md`

---

## Training Logs and Artifacts

### Root Directory Logs

- `training.log` - Main CLIP training log (Approach 1)
- `training_fast.log` - Fast CLIP training log
- `training_output.log` - Additional training output
- `fen_training.log` - FEN generator training log (Approach 2)

### Run-Specific Logs

- `runs/fen_generator/training.log`
- `runs/fen_generator_v2/training.log`
- `runs/fen_generator_v3/training.log`
- `runs/fen_generator_v4_spatial_fix/training.log`
- `runs/fen_generator_v4_spatial_fix_10epochs/training.log`

### Training History Files

- `runs/clip_hf_chess_100k_20epochs_fixed/training_history.json`
- `runs/clip_hf_chess_100k_20epochs_fixed/training_metadata.json`
- `runs/clip_hf_chess_100k_20epochs/training_history.json`
- `runs/clip_hf_chess_50k/training_history.json`
- `runs/clip_hf_chess_fast/training_history.json`
- `Improved_representations/logs/training_history.json`

### Evaluation Logs

- `evaluation_results_beam.txt` - Beam search evaluation
- `evaluation_results_minlen.txt` - Minimum length evaluation
- `evaluation_results_v2_eosmasked.txt` - EOS masking evaluation
- `evaluation_results_v2.txt` - V2 evaluation
- `evaluation_results_v3.txt` - V3 evaluation
- `evaluation_results.txt` - Main evaluation results

---

## Results and Evaluation Files

### JSON Results

#### Approach 1 (CLIP Retrieval)
- `analysis_after_training_on_puzzle_dataset/evaluation_results.json`

#### Approach 2.5 (JSON-First)
- `Improved_representations/results/full_ablation_comparison.json`
- `Improved_representations/results/exp1a_results.json`
- `Improved_representations/results/exp1b_results.json`
- `Improved_representations/results/exp1d_results.json`
- `Improved_representations/results/qwen2vl_eval.json`
- `Improved_representations/results/comparison_1a_vs_1b.json`

#### Approach 2.6 (Downstream Reasoning)
- `benchmarking/benchmark_results_exp1a/summary.json`
- `benchmarking/benchmark_results_exp1a/detailed_results.json`
- `benchmarking/benchmark_results_exp1d/summary.json`
- `benchmarking/benchmark_results_exp1d/detailed_results.json`
- `benchmarking/benchmark_results_qwen/summary.json`
- `benchmarking/benchmark_results_qwen/detailed_results.json`

#### Approach 2.7 (Vision Tower)
- `clip_as_encoder/evaluation_results/baseline_results.json`
- `clip_as_encoder/evaluation_results/chess_clip_results.json`
- `clip_as_encoder/evaluation_results/combined_results.json`
- `clip_as_encoder/evaluation_results/comparison_summary.json`

#### Approach 3 (LLM Extraction)
- `page_fen_pipeline/output/results.json`
- `page_fen_pipeline/results_*.json` (multiple model results)

#### VLM Evaluation
- `evaluation_results/baseline_results.json`
- `evaluation_results/chess_clip_results.json`
- `evaluation_results/comparison_summary.json`
- `evaluation_results/qwen_with_fen/evaluation_results.json`
- `benchmarking/benchmark_results/summary.json`
- `benchmarking/benchmark_results_v2/summary.json`
- `benchmarking/benchmark_results_improved/summary.json`

### Test Results

- `test_100_boards_results/results.json` - 100 board test results
- `testing_files/comprehensive_evaluation_results.json` - Comprehensive evaluation

### Dataset Metadata

- `data/hf_chess_puzzles/dataset_metadata.json` - Dataset information

---

## Summary of Experiment Locations

| Approach | Main Directory | Training Script | Checkpoints | Results |
|----------|---------------|-----------------|-------------|---------|
| **1. Retrieval** | Root | `train_clip_hf_dataset_fast.py` | `runs/clip_hf_chess_100k_20epochs_fixed/` | `analysis_after_training_on_puzzle_dataset/` |
| **2. Generative** | `FEN_generator/` | `FEN_generator/train.py` | `runs/fen_generator_v2/` | `FEN_generator/evaluate.py` output |
| **2.5 JSON (1A)** | `Improved_representations/` | `grid_predictor/train.py` | `checkpoints/exp1a_base_frozen/` | `results/exp1a_results.json` |
| **2.5 JSON (1B)** | `Improved_representations/` | `grid_predictor/train.py` | `checkpoints/exp1b_finetuned_frozen/` | `results/exp1b_results.json` |
| **2.5 JSON (1C)** | `Improved_representations/vlm_finetuning/` | `train_qwen.py` | `checkpoints/qwen2vl_json/` | `results/qwen2vl_eval.json` |
| **2.5 JSON (1D)** | `Improved_representations/` | `grid_predictor/train.py` | `checkpoints/exp1d_base_unfrozen/` | `results/exp1d_results.json` |
| **2.6 Reasoning** | `benchmarking/` | `benchmark_json_models.py` | N/A | `benchmark_results_exp1*/` |
| **2.7 Vision Tower** | `clip_as_encoder/` | `train.py` | N/A | `evaluation_results/` |
| **3. LLM Extract** | `page_fen_pipeline/` | `cli.py` | N/A | `output/results.json` |
| **VLM Eval** | `benchmarking/` | `benchmark.py` | N/A | `benchmark_results/` |

---

## Key Training Artifacts Summary

### Best Models (by Approach)

1. **Approach 1 (Retrieval)**: `runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt`
   - 99.98% top-1 accuracy
   - Best epoch: 15

2. **Approach 2 (Generative)**: `runs/fen_generator_v2/best_model_stage1.pt`
   - Val loss: 0.0225
   - 0% exact match (exposure bias)

3. **Approach 2.5 (JSON)**: `Improved_representations/checkpoints/exp1b_finetuned_frozen/best_model.pt`
   - 79.32% per-square accuracy
   - Best epoch: 11

4. **Approach 2.5 (VLM)**: `Improved_representations/checkpoints/qwen2vl_json/checkpoint-189/`
   - 43.55% per-square accuracy
   - 3 epochs, 189 steps

### Training Datasets

- **Main Dataset**: `data/hf_chess_puzzles/`
  - Train: 99,999 samples
  - Val: 12,500 samples
  - Test: 12,500 samples
- **VLM Dataset**: `Improved_representations/data/vlm_dataset/`
  - Train: 99,999 samples
  - Val: 12,500 samples
  - Test: 12,500 samples
- **JSON Dataset**: `Improved_representations/data/json_dataset/`
  - `train.jsonl`, `val.jsonl`, `test.jsonl`

---

## Notes

- All experiments use the Hugging Face chess puzzles dataset unless otherwise specified
- Training logs may contain multiple runs - check timestamps to identify specific runs
- Some results are documented in the paper (draft_v7.tex) but may not have corresponding JSON files
- Check individual experiment directories for detailed README files and documentation

---

**End of Repository Structure 2**

