# JSON-First Representation Training Plan

## Revised Approach: JSON as Primary Representation

Instead of directly predicting FEN strings (which fails due to exposure bias), we use **JSON as an intermediate structured representation** that:
1. Can be deterministically converted to/from FEN
2. Is easier for VLMs to understand and parse
3. Provides explicit piece-square mappings

## Architecture Overview

```
Image → CLIP Encoder → JSON Predictor → JSON Output
                                              ↓
                              Deterministic JSON→FEN Converter
                                              ↓
                                          FEN String
```

## What We Need to Build

### 1. Deterministic Converters (Already Partially Done)

**FEN → JSON** (`representations.py::board_to_json`)
- Already implemented
- Converts FEN to structured JSON with pieces, relationships, metadata

**JSON → FEN** (NEW - needs to be added)
- Deterministic conversion back to FEN
- Validates JSON structure before conversion
- Handles edge cases (castling, en passant)

### 2. JSON-Based Dataset

Create enriched dataset with JSON representations stored in `Improved_representations/data/`

### 3. JSON Predictor Model

Train a model to predict JSON structure from images:
- Option A: Predict as structured output (classification per field)
- Option B: Predict as text sequence (JSON string generation)

**Recommended: Option A** - Predict pieces per square, then deterministically construct JSON

## Detailed Implementation Plan

### Phase 1: Deterministic Converters

**Files to create/update in `Improved_representations/`:**

1. `data_processing/converters.py` - New file with:
   - `json_to_fen(json_repr)` - Convert JSON back to FEN
   - `validate_json_position(json_repr)` - Validate JSON structure
   - `round_trip_test(fen)` - Test FEN→JSON→FEN consistency

2. Update `data_processing/representations.py`:
   - Ensure `board_to_json()` output is round-trip compatible

### Phase 2: Dataset Creation

**Files in `Improved_representations/data/`:**

```
Improved_representations/
├── data/
│   ├── json_dataset/
│   │   ├── train.json      # [{image_path, fen, json_repr}, ...]
│   │   ├── val.json
│   │   └── test.json
│   └── raw/                 # Symlinks or copies of original images
```

**Enrichment script**: `data_processing/create_json_dataset.py`
- Reads original CSV files
- Converts each FEN to JSON representation
- Saves as structured JSON files
- Verifies round-trip consistency

### Phase 3: JSON Predictor Model

**Architecture** (in `Improved_representations/json_predictor/`):

Since JSON has a fixed structure for chess:
- 64 squares × 13 piece types = Grid prediction
- Plus metadata (to_move, castling, en_passant)

**Model components:**
1. `json_predictor/model.py` - Main model
2. `json_predictor/dataset.py` - Dataset loader
3. `json_predictor/train.py` - Training script
4. `json_predictor/evaluate.py` - Evaluation

**Output format:**
```python
{
    "pieces": [
        {"piece": "white_pawn", "square": "e4", ...},
        ...
    ],
    "metadata": {
        "to_move": "white",
        "material": {"white": 39, "black": 35}
    }
}
```

### Phase 4: Training

**Stage 1: Create Dataset**
```bash
cd Improved_representations
python -m data_processing.create_json_dataset \
    --input_csv ../data/hf_chess_puzzles/test.csv \
    --output_dir data/json_dataset \
    --split test
```

**Stage 2: Train Model**
```bash
python -m json_predictor.train \
    --data_dir data/json_dataset \
    --checkpoint_dir checkpoints/json_predictor \
    --encoder_checkpoint ../runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt \
    --epochs 20
```

**Stage 3: Evaluate**
```bash
python -m json_predictor.evaluate \
    --checkpoint checkpoints/json_predictor/best.pt \
    --test_data data/json_dataset/test.json \
    --output results/json_results.json
```

## File Structure (All in Improved_representations/)

```
Improved_representations/
├── data_processing/
│   ├── __init__.py
│   ├── representations.py      # FEN→JSON (exists)
│   ├── converters.py           # NEW: JSON→FEN, validation
│   └── create_json_dataset.py  # NEW: Dataset creation script
├── data/
│   └── json_dataset/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── json_predictor/             # NEW: JSON prediction model
│   ├── __init__.py
│   ├── model.py               # JSON prediction model
│   ├── dataset.py             # Dataset loader
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation script
├── checkpoints/
│   └── json_predictor/
├── results/
│   └── json_results.json
├── README.md
├── PLAN_JSON_FIRST.md         # This file
├── report.md
└── updates_improved_representations.md
```

## Key Design Decisions

### JSON Format for Chess Position

```json
{
    "pieces": [
        {"piece": "white_king", "square": "g1", "type": "king", "color": "white", "value": 0},
        {"piece": "white_queen", "square": "d1", "type": "queen", "color": "white", "value": 9},
        {"piece": "white_pawn", "square": "e4", "type": "pawn", "color": "white", "value": 1},
        ...
    ],
    "metadata": {
        "to_move": "white",
        "castling_rights": {"white": ["K", "Q"], "black": []},
        "en_passant": null,
        "material": {"white": 39, "black": 35},
        "material_balance": 4
    }
}
```

### Why JSON First?

1. **Deterministic conversion**: JSON↔FEN is lossless and deterministic
2. **VLM-friendly**: Modern VLMs understand JSON well
3. **Explicit structure**: Each piece is explicitly listed with its square
4. **Validation**: Easy to validate (exactly 2 kings, piece counts, etc.)
5. **Extensible**: Can add relationships, tactics without breaking format

### Prediction Strategy

The model predicts:
1. **For each of 64 squares**: Which of 13 piece types (including empty)
2. **Metadata**: to_move (binary), castling (4 binary), en_passant (square or none)

Then deterministically constructs JSON from predictions.

## Success Criteria

1. **Round-trip accuracy**: FEN→JSON→FEN should be 100% identical
2. **Piece prediction accuracy**: >85% per-square accuracy
3. **Exact board match**: >60% of boards fully correct
4. **JSON validity**: 100% of predictions should be valid JSON
5. **FEN extraction**: Generated FEN should be valid chess positions

---

## Full Experimental Roadmap

### Approach 1: CLIP-Based JSON Predictor (CURRENT)

**Status**: In Progress

Train a CLIP-based model to predict JSON representation:
- Use CLIP ViT-B/32 as visual encoder (with fine-tuned weights)
- Per-square classification (64 x 13)
- Metadata prediction (to_move, castling)
- Deterministic JSON/FEN reconstruction

**Files**: `json_predictor/model.py`, `train.py`, `evaluate.py`

#### Ablation Experiments (Before LLaVA)

To understand what contributes to performance, run these comparisons:

| Exp | Vision Encoder | Frozen? | Purpose |
|-----|----------------|---------|---------|
| **1A** | Base CLIP (no chess fine-tuning) | Yes | Baseline - generic CLIP features |
| **1B** | Chess fine-tuned CLIP | Yes | **(Current)** - value of domain pre-training |
| **1C** | Chess fine-tuned CLIP | No | End-to-end training benefit |
| **1D** | Base CLIP | No | Train from scratch for this task |

**Key questions answered:**
- **1A vs 1B**: Does chess-specific CLIP fine-tuning improve grid prediction?
- **1B vs 1C**: Does unfreezing encoder help further?
- **1A vs 1D**: How much does task-specific training matter vs frozen generic features?

**Training commands:**
```bash
# 1A: Base CLIP, frozen
python -m json_predictor.train --freeze_encoder --epochs 20 ...

# 1B: Fine-tuned CLIP, frozen (CURRENT)
python -m json_predictor.train --encoder_checkpoint ../runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt --freeze_encoder --epochs 20 ...

# 1C: Fine-tuned CLIP, unfrozen
python -m json_predictor.train --encoder_checkpoint ../runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt --epochs 20 ...

# 1D: Base CLIP, unfrozen
python -m json_predictor.train --epochs 20 ...
```

### Approach 2: LLaVA/Vision-Language Model Fine-tuning (NEXT)

**Status**: Planned

Fine-tune LLaVA or similar VLM to directly generate JSON from chess images:
- Input: Chess board image + prompt ("Describe this chess position in JSON format")
- Output: JSON string with pieces and metadata
- Advantages: Natural language interface, can ask follow-up questions
- Framework: Hugging Face transformers, PEFT/LoRA for efficient fine-tuning

**Files to create**:
- `vlm_finetuning/llava_dataset.py` - Dataset for VLM fine-tuning
- `vlm_finetuning/train_llava.py` - LLaVA fine-tuning script
- `vlm_finetuning/evaluate_llava.py` - Evaluation script

**Candidate models**:
- LLaVA-1.5-7B (most tested)
- Qwen-VL (good multilingual support)
- InternVL (strong visual understanding)

### Approach 3: Other Methods (FUTURE)

**Status**: Planned

Potential alternative approaches:
1. **Encoder-Decoder with Attention**: CNN/ViT encoder + Transformer decoder for JSON generation
2. **Diffusion-based**: Generate piece placements using diffusion model
3. **Graph Neural Network**: Model board as graph, predict node types
4. **Hybrid**: Combine multiple approaches with voting/ensemble

---

## Implementation Progress Checklist

### Approach 1: CLIP-Based JSON Predictor
- [x] Phase 1: Deterministic converters (FEN↔JSON)
- [x] Phase 2: Dataset creation script
- [x] Phase 3: Model architecture
- [x] Phase 4a: Generate full datasets (train/val/test JSONL)
- [x] Phase 4b: Training script
- [x] Phase 4c: Evaluation script
- [ ] Phase 4d: Train Exp 1B (Fine-tuned CLIP, frozen) - IN PROGRESS
- [ ] Phase 4e: Evaluate Exp 1B
- [ ] Phase 4f: Train Exp 1A (Base CLIP, frozen)
- [ ] Phase 4g: Train Exp 1C (Fine-tuned CLIP, unfrozen)
- [ ] Phase 4h: Train Exp 1D (Base CLIP, unfrozen)
- [ ] Phase 4i: Compare all ablation results

### Approach 2: LLaVA Fine-tuning
- [ ] Create VLM dataset format (image + JSON pairs)
- [ ] Setup LLaVA/Qwen-VL environment
- [ ] Create fine-tuning script with LoRA
- [ ] Train and evaluate
- [ ] Compare with Approach 1

### Approach 3: Other Methods
- [ ] To be determined based on results from Approach 1 & 2

