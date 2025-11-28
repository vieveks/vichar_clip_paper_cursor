# Vichar-CLIP: Chess Position Identification and FEN Generation Project

## Project Overview

This project develops and evaluates multimodal AI systems for chess position understanding, focusing on translating chess board images into structured FEN (Forsyth-Edwards Notation) representations. The work explores two complementary approaches: (1) **Retrieval-based CLIP matching** for known positions, and (2) **Generative FEN prediction** for novel positions. The ultimate goal is to enhance Vision Language Models (VLMs) with symbolic chess context to improve their performance on chess-related reasoning tasks.

## Core Objectives

1. **Image-to-FEN Translation**: Develop models capable of accurately identifying chess positions from board images
2. **Symbolic Context Integration**: Provide structured FEN representations as context for VLMs to improve chess reasoning
3. **Educational Applications**: Enable chess learning tools that work with visual inputs without requiring notation knowledge
4. **Research Contribution**: Advance multimodal AI in specialized domains through systematic experimentation

## Project Architecture

### High-Level System Design

```
Chess Board Image
    ↓
[CLIP Encoder] → Visual Embeddings
    ↓
[Two Approaches]
    ├─→ [Retrieval System] → Match to Known FEN (Database Lookup)
    └─→ [Generative Decoder] → Generate FEN String (Autoregressive)
    ↓
FEN String
    ↓
[VLM Integration] → Enhanced Chess Reasoning with Symbolic Context
```

## Main Components

### 1. CLIP-Based Retrieval System

**Purpose**: Match chess board images to their corresponding FEN strings from a database of known positions.

**Architecture**:
- **Base Model**: OpenAI CLIP ViT-B/32 (Vision Transformer Base, 32×32 patches)
- **Pretrained Weights**: LAION-2B (trained on 2 billion image-text pairs)
- **Fine-tuning**: Domain-specific adaptation on chess puzzle dataset (125k samples)
- **Embedding Dimension**: 512-dimensional shared space

**Training Details**:
- **Dataset**: Hugging Face chess-puzzles-images-mini (125k samples)
- **Splits**: 100k train / 12.5k validation / 12.5k test
- **Training Configuration**:
  - Batch size: 256
  - Learning rate: 5e-5
  - Epochs: 20
  - Mixed precision (FP16) training
  - Gradient clipping (max_norm=1.0) for stability
- **Loss Function**: Contrastive loss (standard CLIP objective)

**Results**:
- **Test Set Performance**:
  - Top-1 Accuracy: 99.98%
  - Top-5 Accuracy: 100.00%
  - Top-10 Accuracy: 100.00%
  - Test Loss: 0.0009 (99.99% reduction from base model)
- **Base Model Comparison**:
  - Base model (pretrained only): 0.38% Top-1 accuracy
  - Fine-tuned model: 99.98% Top-1 accuracy (+99.59% improvement)

**Limitations**:
- Only works for positions in the training database
- Cannot handle novel/unseen positions
- Requires maintaining a large FEN database

**Location**: 
- Training scripts: `train_clip_hf_dataset.py`, `train_clip_hf_dataset_fast.py`
- Inference: `inference_clip.py`
- Model checkpoints: `runs/clip_hf_chess_100k_20epochs_fixed/`

### 2. Generative FEN Generator

**Purpose**: Generate FEN strings for any chess position, including novel positions not seen during training.

**Architecture**:
- **Encoder**: Fine-tuned CLIP ViT-B/32 (initialized from retrieval model)
  - Uses spatial patch embeddings: `[B, 49, 768]` (49 patch tokens from 7×7 feature map)
  - Projects to decoder dimension: `[B, 49, 512]`
- **Decoder**: Transformer Decoder
  - 6 decoder layers
  - 8 attention heads
  - 512-dimensional model
  - Character-level tokenizer for FEN strings (~30 token vocabulary)
- **Training Strategy**: Two-stage approach
  - **Stage 1**: Freeze encoder, train decoder only (learns FEN syntax)
  - **Stage 2**: Unfreeze encoder, fine-tune end-to-end (learns spatial precision)

**Training Details**:
- **Dataset**: Same Hugging Face chess puzzles dataset
- **Tokenizer**: Character-level FEN tokenizer (SOS, EOS, PAD tokens)
- **Training Configuration**:
  - Stage 1: 3 epochs, frozen encoder
  - Stage 2: 0-3 epochs (optional), unfrozen encoder with lower LR (1e-6)
  - Batch size: 64
  - Mixed precision training
  - Gradient clipping for stability
- **Loss Function**: Cross-entropy loss with optional length penalty

**Generation Approaches Explored**:

1. **Greedy Decoding** (Baseline)
   - Results: 0% exact match, CER 0.6663
   - Issue: Premature EOS prediction (~15-20 tokens vs expected 42-44)

2. **Beam Search** (beam_size=5)
   - Results: 0% exact match, CER 0.7120
   - Finding: Beam search didn't help (issue is training, not selection)

3. **Minimum Length Constraint** (min_length=35)
   - Results: 0% exact match, CER 0.7204
   - Issue: Forced longer generation caused repetitive garbage

4. **EOS Token Masking** (min_length=35, mask EOS until minimum reached)
   - Results: 0% exact match, CER 0.7387
   - Implementation: Mask EOS token logits during generation until sequence length ≥ 35
   - Finding: Prevents early stopping but produces repetitive patterns
   - Root cause: Model never trained to generate beyond ~20 tokens

5. **Length Penalty in Loss** (v3)
   - Results: CER 1.16 (worse than baseline)
   - Issue: Overcorrected, generates excessive length with repetitions

**Current Status**:
- Best performing: v2 model with greedy decoding (CER 0.70)
- Challenge: Exposure bias - model sees ground truth during training but must condition on own predictions at inference
- Future directions: Scheduled sampling, better curriculum learning, or accepting partial FENs for VLM context

**Location**:
- Code: `FEN_generator/` directory
  - `model.py`: Model architecture
  - `train.py`: Training script
  - `evaluate.py`: Evaluation script
  - `tokenizer.py`: FEN tokenizer
  - `dataset.py`: Dataset loader
- Documentation: `FEN_generator/UPDATES.md`
- Model checkpoints: `runs/fen_generator_v2/`, `runs/fen_generator_v3/`

### 3. VLM Benchmarking System

**Purpose**: Evaluate whether providing FEN context (from CLIP predictions) improves Vision Language Model performance on chess-related questions.

**Architecture**:
```
Chess Board Image
    ↓
[CLIP Model] → PREDICTED FEN String
    ↓
[Ground Truth Extractor] → Ground Truth Answers (from predicted FEN)
    ↓
[VLM] → Answer (without FEN)
[VLM] → Answer (with PREDICTED FEN)
    ↓
[Scorer] → Scores (0-1)
    ↓
[Analysis] → Comparison Report
```

**Key Design Decisions**:
1. **Uses PREDICTED FEN, not ground truth**: Tests full pipeline (image → CLIP → FEN → VLM)
2. **Two-tier testing**: Each question tested with and without FEN context
3. **Question types**: 10 different chess reasoning tasks (piece location, best move, check status, etc.)

**Question Types**:
1. Piece Location (weight: 1.0)
2. Best Move (weight: 1.0)
3. Winning Assessment (weight: 0.0)
4. Position Strength (weight: 0.5)
5. Previous Move Quality (weight: 1.0)
6. Knight Attacks (weight: 1.0)
7. Material Count (weight: 1.0)
8. Check Status (weight: 1.0)
9. Castling Rights (weight: 1.0)
10. Threats (weight: 1.0)

**Ground Truth Sources**:
- **pychess**: Board state analysis (piece locations, attacks, check status, castling rights)
- **Lichess API**: Engine analysis (best moves, evaluations, threats)

**Location**:
- Code: `benchmarking/` directory
  - `benchmark.py`: Main benchmarking script
  - `questions.py`: Question definitions
  - `clip_fen_extractor.py`: FEN extraction utility
  - `ground_truth.py`: Ground truth extraction
  - `vlm_integration.py`: VLM integration
  - `scoring.py`: Scoring utilities
- Documentation: `benchmarking/README.md`, `benchmarking/IMPLEMENTATION_NOTES.md`

## Dataset

### Primary Dataset: Hugging Face Chess Puzzles

- **Source**: `chess-puzzles-images-mini` dataset from Hugging Face
- **Size**: 125,000 chess position samples
- **Splits**: 
  - Train: 99,999 samples
  - Validation: 12,500 samples
  - Test: 12,500 samples
- **Format**: 
  - Images: 350×350 PNG chess board images
  - Labels: FEN strings (board placement only)
- **Usage**: Used for both CLIP fine-tuning and FEN generator training

### Dataset Statistics

- **Average FEN length**: 42.37 characters (44.37 with SOS/EOS tokens)
- **FEN structure**: All complete (7 slashes, 8 ranks)
- **No data corruption**: Verified through analysis scripts

**Location**: `data/hf_chess_puzzles/`

## Training Methodology

### CLIP Fine-tuning

1. **Initialization**: LAION-2B pretrained weights
2. **Objective**: Contrastive learning (image-text matching)
3. **Optimization**:
   - Adam optimizer
   - Learning rate: 5e-5 (reduced from 1e-4 for stability)
   - Gradient clipping: max_norm=1.0
   - Mixed precision (FP16)
4. **Monitoring**: Validation loss tracking, best model checkpointing

### FEN Generator Training

1. **Two-Stage Strategy**:
   - Stage 1: Freeze CLIP encoder, train decoder (learns FEN syntax)
   - Stage 2: Unfreeze encoder, fine-tune end-to-end (spatial precision)
2. **Optimization**:
   - Adam optimizer
   - Encoder LR: 1e-6 (Stage 2 only)
   - Decoder LR: 1e-4
   - Gradient clipping for stability
   - Mixed precision training
3. **Loss Function**: Cross-entropy with optional length penalty

## Evaluation Metrics

### CLIP Retrieval System
- **Top-K Accuracy**: Percentage of queries where correct FEN is in top-K matches
- **Average Rank**: Average position of correct FEN in ranked results
- **Confidence Scores**: Similarity scores for ranking

### FEN Generator
- **Exact Match Accuracy**: Percentage of perfectly generated FEN strings
- **Character Error Rate (CER)**: Edit distance normalized by target length
- **Sequence Length Analysis**: Comparison of generated vs target lengths

### VLM Benchmarking
- **Score with FEN**: Average score when FEN context is provided
- **Score without FEN**: Average score without FEN context
- **Improvement**: Difference between with/without FEN scores
- **Per-question breakdown**: Performance on individual question types

## Key Results

### CLIP Retrieval System
- ✅ **99.98% Top-1 accuracy** on test set
- ✅ **100% Top-5 accuracy**
- ✅ **99.99% loss reduction** from base model
- ✅ Successfully adapted from general vision-language to chess domain

### FEN Generator
- ⚠️ **0% exact match accuracy** (ongoing challenge)
- ⚠️ **CER ~0.70** (best performing approach: v2 greedy decoding)
- ⚠️ **Premature stopping issue**: Model predicts EOS too early
- ✅ **Partial FENs useful**: Even incomplete FENs may help VLM context

### VLM Benchmarking
- 🔄 **In progress**: Full benchmark results pending
- **Design**: Tests full pipeline with predicted FEN (not ground truth)
- **Goal**: Demonstrate FEN context improves VLM chess reasoning

## Project Structure

```
vichar_clip_paper_cursor/
├── README.md                          # Main project README
├── updates.md                         # Project updates log
├── Description_entire_project.md      # This file
│
├── FEN_generator/                     # Generative FEN prediction
│   ├── model.py                      # Model architecture
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Evaluation script
│   ├── tokenizer.py                  # FEN tokenizer
│   ├── dataset.py                    # Dataset loader
│   └── UPDATES.md                    # FEN generator updates
│
├── benchmarking/                      # VLM benchmarking system
│   ├── benchmark.py                 # Main benchmark script
│   ├── questions.py                 # Question definitions
│   ├── clip_fen_extractor.py        # FEN extraction
│   ├── ground_truth.py              # Ground truth extraction
│   ├── vlm_integration.py           # VLM integration
│   ├── scoring.py                   # Scoring utilities
│   └── README.md                    # Benchmarking documentation
│
├── data/                             # Datasets
│   └── hf_chess_puzzles/            # Hugging Face chess dataset
│       ├── train/                   # Training images
│       ├── validation/             # Validation images
│       ├── test/                   # Test images
│       └── *.csv                   # FEN labels
│
├── runs/                             # Training outputs
│   ├── clip_hf_chess_100k_20epochs_fixed/  # CLIP model checkpoints
│   ├── fen_generator_v2/            # FEN generator v2
│   └── fen_generator_v3/            # FEN generator v3
│
├── docs/                             # Documentation
│   ├── CLIP_MODEL_EXPLANATION.md
│   ├── EXPERIMENTS_AND_RESULTS.md
│   ├── HF_DATASET_METHODOLOGY.md
│   └── PROJECT_COMPLETION_SUMMARY.md
│
├── Paper_drafts/                     # Research paper drafts
│   └── draft_v4_detailed.tex        # Latest paper draft
│
├── utils/                            # Utility scripts
│   ├── download_hf_chess_dataset.py
│   └── hf_chess_dataset_loader.py
│
├── train_clip_hf_dataset.py          # CLIP training script
├── train_clip_hf_dataset_fast.py     # Fast CLIP training
└── inference_clip.py                 # CLIP inference script
```

## Key Technical Contributions

1. **Domain Adaptation**: Successfully fine-tuned CLIP for chess-specific visual understanding
2. **Spatial Feature Extraction**: Used patch-level embeddings instead of pooled features for better spatial awareness
3. **Two-Stage Training**: Developed effective strategy for adapting pretrained encoders to generation tasks
4. **Systematic Evaluation**: Comprehensive evaluation of multiple generation strategies
5. **Full Pipeline Testing**: VLM benchmarking with predicted FEN (not ground truth) for realistic evaluation

## Challenges and Solutions

### Challenge 1: Early Stopping in FEN Generation
- **Problem**: Model predicts EOS token too early (~15-20 tokens vs 42-44 expected)
- **Root Cause**: Training with teacher forcing + padding strategy allows model to minimize loss by predicting EOS early
- **Attempted Solutions**:
  - EOS masking during generation (didn't help - produces garbage)
  - Length penalty in loss (overcorrected - excessive repetitions)
  - Minimum length constraints (model not trained for longer sequences)
- **Current Status**: Ongoing research, best approach is v2 greedy decoding with CER 0.70

### Challenge 2: Training Stability
- **Problem**: Gradient explosion during long training runs
- **Solution**: Gradient clipping (max_norm=1.0) + reduced learning rate (5e-5)
- **Result**: Stable 20-epoch training completed successfully

### Challenge 3: Spatial Information Loss
- **Problem**: Using pooled CLIP features lost spatial information needed for piece localization
- **Solution**: Extract spatial patch embeddings (49 tokens from 7×7 feature map)
- **Result**: 97% improvement in validation loss (v1: 0.68 → v2: 0.0225)

## Future Directions

1. **FEN Generator Improvements**:
   - Scheduled sampling to address exposure bias
   - Curriculum learning (start with short FENs, gradually increase)
   - Better loss functions for sequence generation

2. **VLM Integration**:
   - Complete benchmarking with full test set
   - Analysis of which question types benefit most from FEN context
   - Integration with multiple VLM architectures

3. **Applications**:
   - Educational chess tutoring systems
   - Mobile applications for position analysis
   - Accessibility tools for chess notation

4. **Research Extensions**:
   - Cross-game generalization (Go, checkers, etc.)
   - Real-time video analysis
   - Multi-board scenarios

## Hardware and Software

- **Hardware**: NVIDIA RTX 5070 Ti (16GB VRAM)
- **Software**: 
  - PyTorch 2.9.0 with CUDA
  - Python 3.x
  - open-clip-torch for CLIP models
  - transformers for VLMs
- **Environment**: Conda environment (pytorch_5070ti)

## Reproducibility

All training configurations, hyperparameters, and dataset splits are documented in:
- `updates.md`: Training history and configurations
- `FEN_generator/UPDATES.md`: FEN generator development log
- `docs/HF_DATASET_METHODOLOGY.md`: Dataset and training methodology
- `analysis_after_training_on_puzzle_dataset/TRAINING_ANALYSIS.md`: Comprehensive training analysis

Model checkpoints are saved in `runs/` directory with training metadata for full reproducibility.

## Citation and Usage

This project is part of research on multimodal AI for chess understanding. For questions or collaboration, please refer to the documentation files or contact the project maintainers.

---

**Last Updated**: Based on project state as of latest evaluation runs
**Status**: Active research project with ongoing improvements

