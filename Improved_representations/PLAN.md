# Hierarchical Multi-Representation System - Implementation Plan

This document contains the complete implementation plan for the improved representations approach.

**Note**: This is a reference document. For progress updates, see `representation_updates.md`.

## Problem Statement

The current generative FEN decoder achieves 0% exact match accuracy despite low training loss, indicating exposure bias and sequence modeling challenges. Instead of only predicting FEN strings, we implement a hierarchical system that learns multiple complementary representations, with grid-based classification as the foundation.

## Architecture Overview

The system has three levels:

1. **Level 1: Grid Prediction** - Per-square piece classification (64×13-way classification)
2. **Level 2: Structured Representations** - FEN, JSON, Graph, and Natural Language decoders
3. **Level 3: Tactical Analysis** - Pattern detection (pins, forks, checks)

## Implementation Phases

### Phase 1: Data Enrichment ✅

**Status**: Completed

**Files Created**:
- `data_processing/representations.py` - All representation conversion functions
- `data_processing/enrich_dataset.py` - Dataset enrichment script
- `data_processing/__init__.py` - Package initialization

**Key Functions**:
- `fen_to_grid()` - Convert FEN to 8×8 integer matrix
- `board_to_json()` - Extract structured JSON with pieces, relationships, metadata
- `board_to_graph()` - Convert to graph representation
- `board_to_natural_language()` - Generate descriptive text
- `analyze_tactics()` - Detect pins, forks, checks

### Phase 2: Grid-Based Model (In Progress)

**Goal**: Implement Level 1 - per-square classification

**Files to Create**:
- `grid_predictor/model.py` - Spatial grid predictor architecture
- `grid_predictor/train.py` - Training script
- `grid_predictor/evaluate.py` - Evaluation metrics

**Architecture**:
- Vision encoder: CLIP ViT-B/32 (reuse existing fine-tuned weights)
- Spatial aligner: Learnable upsampling from 7×7 patches to 8×8 board squares
- Grid classifier: 64 independent 13-way classifiers

**Expected Performance**: 70-90% per-square accuracy, 50-70% exact board match

### Phase 3: Multi-Representation Decoders

**Goal**: Add Level 2 decoders

**Files to Create**:
- `multi_representation/model.py` - Complete hierarchical model
- `multi_representation/decoders.py` - FEN, JSON, Graph, NL decoders
- `multi_representation/train.py` - Multi-task training script

### Phase 4: Tactical Analysis

**Goal**: Add Level 3 - tactical pattern detection

**Files to Create**:
- `tactical_analyzer/model.py` - Pattern detection network
- `tactical_analyzer/patterns.py` - Pattern detection logic

### Phase 5: Adaptive Inference & Evaluation

**Goal**: Implement adaptive representation selection

**Files to Create**:
- `adaptive_inference/selector.py` - Confidence-based selection
- `adaptive_inference/evaluator.py` - Multi-representation evaluation
- `benchmarking/multi_rep_benchmark.py` - Extended benchmark

### Phase 6: Integration & Paper Updates

**Goal**: Integrate with existing codebase and update paper

**Files to Modify**:
- `benchmarking/vlm_integration.py` - Add multi-representation support
- `Paper_drafts/draft_v6_qwen.tex` - Update methodology section

## Key Technical Decisions

1. **Grid as Foundation**: Start with 64×13 classification instead of sequence generation to avoid exposure bias
2. **Spatial Alignment**: Use learnable upsampling (not just bilinear) to align 7×7 patches to 8×8 squares
3. **Graph-Based Representation**: Use GNN to explicitly model piece relationships
4. **Multi-Task Learning**: Train all representations jointly with weighted loss
5. **Progressive Training**: Curriculum learning from simple (grid) to complex (graph/tactics)

## Expected Improvements

- **Grid accuracy**: 70-90% per-square (vs 0% FEN exact match)
- **FEN accuracy**: 40-60% exact match (improved from 0% by using grid constraints)
- **JSON accuracy**: 60-80% field-level accuracy
- **VLM performance**: Better than FEN-only due to richer representations

## Success Criteria

1. Grid predictor achieves >70% per-square accuracy
2. FEN decoder improves from 0% to >40% exact match when using grid constraints
3. JSON representation enables better VLM performance on relationship questions
4. System can adaptively select best representation for each question type
5. Paper methodology section clearly describes hierarchical approach

