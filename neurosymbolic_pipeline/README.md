# Neurosymbolic Pipeline for Chess Position Understanding

This directory contains the implementation of a 3-stage neurosymbolic pipeline that combines neural perception with symbolic reasoning to address the "0.008% exact match" problem.

## Structure

- `experiment_a/` - Stockfish CP Loss validation (Experiment A)
- `experiment_b/` - Symbolic Refinement module (Experiment B) ⭐ Priority
- `experiment_c/` - Hybrid Reasoning Engine (Experiment C)
- `results/` - All experimental results (isolated from existing results)
- `shared/` - Shared utilities across experiments
- `tests/` - Unit and integration tests

## Quick Start

### Experiment B: Symbolic Refinement (Highest Priority)

```bash
cd neurosymbolic_pipeline/experiment_b
python evaluate_refinement.py
```

### Experiment A: Stockfish CP Loss

```bash
cd neurosymbolic_pipeline/experiment_a
python evaluate_cp_loss.py
```

### Experiment C: Hybrid Reasoning

```bash
cd neurosymbolic_pipeline/experiment_c
python evaluate_hybrid_reasoning.py
```

## Key Features

- **Complete Isolation**: All code is separate from existing implementations
- **Read-Only Access**: Only reads from existing files, never modifies them
- **Clear Results**: All results stored in `results/` subdirectories
- **Reproducible**: Each experiment has its own README and documentation

## Dependencies

- `python-chess` (already in project requirements)
- `torch` (for loading model checkpoints)
- `numpy` (for numerical operations)

## Status

- ✅ Directory structure created
- 🚧 Experiment B: In progress
- ⏳ Experiment A: Pending
- ⏳ Experiment C: Pending

