# Improved Representations for Chess VLMs

This module implements a hierarchical multi-representation system for chess position understanding, addressing the 0% FEN accuracy problem by learning multiple complementary representations.

## Overview

Instead of only predicting FEN strings (which suffers from exposure bias), we implement a three-level hierarchical system:

1. **Level 1: Grid Prediction** - Per-square piece classification (64×13-way classification)
2. **Level 2: Structured Representations** - FEN, JSON, Graph, and Natural Language decoders
3. **Level 3: Tactical Analysis** - Pattern detection (pins, forks, checks)

## Architecture

```
Input Image (512×512)
    ↓
Vision Encoder (CLIP ViT-B/32)
    ↓
Spatial Aligner (7×7 → 8×8)
    ↓
┌─────────────────────────────────┐
│  Level 1: Grid Predictor       │
│  64×13 classification           │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Level 2: Multi-Representation  │
│  - FEN Decoder                  │
│  - JSON Decoder                 │
│  - Graph Decoder (GNN)          │
│  - Natural Language Decoder     │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Level 3: Tactical Analyzer    │
│  - Pin detection                │
│  - Fork detection               │
│  - Check detection              │
└─────────────────────────────────┘
```

## Quick Start

### 1. Enrich Dataset

First, enrich your dataset with all representations:

```bash
cd Improved_representations

# Enrich test split (small sample for testing)
python -m data_processing.enrich_dataset \
    --csv_path ../data/hf_chess_puzzles/test.csv \
    --output_path data/enriched_dataset/test.json \
    --max_samples 100

# Enrich full train split (takes time)
python -m data_processing.enrich_dataset \
    --csv_path ../data/hf_chess_puzzles/train.csv \
    --output_path data/enriched_dataset/train.json
```

### 2. Train Grid Predictor

```bash
python -m grid_predictor.train \
    --data_path data/enriched_dataset/train.json \
    --checkpoint_dir checkpoints/grid_predictor
```

### 3. Train Multi-Representation Model

```bash
python -m multi_representation.train \
    --data_path data/enriched_dataset/train.json \
    --grid_checkpoint checkpoints/grid_predictor/best.pt \
    --checkpoint_dir checkpoints/multi_representation
```

## Directory Structure

```
Improved_representations/
├── data_processing/          # Dataset enrichment scripts
├── data/                     # Enriched datasets
├── grid_predictor/          # Level 1: Grid prediction
├── multi_representation/    # Level 2: Multi-representation decoders
├── tactical_analyzer/       # Level 3: Tactical analysis
├── adaptive_inference/      # Adaptive representation selection
├── benchmarking/            # Evaluation scripts
├── checkpoints/             # Model checkpoints
├── results/                 # Evaluation results
├── README.md                # This file
├── PLAN.md                  # Implementation plan
├── report.md                # Progress report
├── representation_updates.md # Living document with updates
└── requirements.txt         # Dependencies
```

## Representations

### Grid (8×8 Matrix)
- 13 classes per square (empty + 6 white + 6 black pieces)
- Direct spatial alignment with board
- Foundation for all other representations

### JSON (Structured)
- Explicit list of all pieces with squares
- Relationships (attacks, defends, pins)
- Metadata (material, castling, to_move)

### Graph (NetworkX/PyTorch Geometric)
- Nodes: pieces with features
- Edges: relationships between pieces
- Enables GNN-based reasoning

### Natural Language
- Human-readable descriptions
- VLM-friendly format
- Includes material, check status, etc.

### Tactics
- Pin detection
- Fork detection
- Check detection
- Hanging piece detection

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- `python-chess` - Chess board manipulation
- `torch` - PyTorch for models
- `transformers` - For NL decoder
- `networkx` - For graph representation
- `open-clip-torch` - CLIP models

## Status

See `representation_updates.md` for latest progress and results.

