# Implementation Plan: Neurosymbolic Pipeline Improvements

Based on suggestions in `detailed_guidance.txt`, this plan outlines the implementation of a 3-stage neurosymbolic pipeline to address the "0.008% exact match" problem and strengthen the paper's narrative.

## Important: Isolation Strategy

**All new code and results will be in a completely separate folder structure:**
- **New Code Location**: `neurosymbolic_pipeline/` (root directory)
- **New Results Location**: `neurosymbolic_pipeline/results/`
- **No modifications to existing files** - all existing code remains untouched
- **Clear separation** - easy to identify what's new vs existing

## Overview

The plan implements three key experiments that transform the current results into a cohesive neurosymbolic architecture:

1. **Experiment A**: Stockfish CP Loss validation (shows errors are "benign")
2. **Experiment B**: Symbolic Refinement module (improves exact match 0.008% → 5-10%)
3. **Experiment C**: Symbolic Checker for logic-based questions (check detection 20% → 90%+)

---

## Architecture Overview

```
Input Image
    ↓
Stage 1: Neural Perception & Grounding
    ├─ Approach 1 (Retrieval): 99.98% accuracy (closed-world)
    ├─ Approach 2.5 (JSON-First): 79.32% per-square accuracy (open-world)
    └─ Approach 3 (LLM Extraction): 94% accuracy (real-world)
    ↓
Stage 2: Symbolic Refinement
    ├─ Piece count validation (≤32 pieces)
    ├─ King uniqueness (exactly 1 per color)
    ├─ Pawn placement rules (no pawns on rank 1/8)
    ├─ Castling rights consistency
    └─ Confidence-based error correction
    ↓
Stage 3: Hybrid Reasoning Engine
    ├─ Symbolic Checker Path (check, castling, legal moves)
    └─ VLM Semantic Path (best move, tactical patterns)
```

---

## Experiment A: Stockfish CP Loss Validation

**Objective**: Validate that predicted FEN errors have minimal strategic impact by comparing Stockfish evaluations.

**Key Metric**: Mean CP (centipawn) loss < 150 between predicted and ground truth FENs

**Evaluation Method**: Use Lichess Cloud Evaluation API (already implemented in `benchmarking/ground_truth.py`)

### Implementation Steps

1. **Update Stockfish Evaluation Module**
   - **File**: `neurosymbolic_pipeline/experiment_a/stockfish_evaluator.py`
   - **Functionality**:
     - Use Lichess Cloud Evaluation API (via `benchmarking/ground_truth.GroundTruthExtractor`)
     - Evaluate position from FEN string (get CP score)
     - Compare predicted FEN vs ground truth FEN CP scores
     - Calculate CP loss (absolute difference)
     - Fallback to python-chess simple evaluation if API unavailable
   - **Dependencies**: `requests` (for Lichess API), `python-chess` (fallback)
   - **Note**: Lichess API is free, has rate limiting, but more accurate than simple material evaluation

2. **Update Evaluation Script**
   - **File**: `neurosymbolic_pipeline/experiment_a/evaluate_cp_loss.py`
   - **Functionality**:
     - Load predictions from any JSON model (Exp 1A, 1B, 1C, 1D)
     - For each test sample:
       - Get predicted FEN from JSON model
       - Get ground truth FEN
       - Evaluate both with Lichess API (via GroundTruthExtractor)
       - Calculate CP loss
     - Aggregate statistics: mean, std, percentiles
   - **Input**: `Improved_representations/results/predictions_clip_exp1*.jsonl` (read-only, existing files)
   - **Output**: `neurosymbolic_pipeline/results/exp_a/cp_loss_results.json` (new results folder)
   - **Models to Test**: Exp 1A, 1B, 1C, 1D (all JSON-based models)

3. **Expected Results**
   - Mean CP loss: ~127 ± 89 (from guidance)
   - Interpretation: Errors are "benign" - strategic evaluation preserved despite low exact match

### Files to Create

- `neurosymbolic_pipeline/__init__.py` (package marker)
- `neurosymbolic_pipeline/experiment_a/__init__.py`
- `neurosymbolic_pipeline/experiment_a/stockfish_evaluator.py`
- `neurosymbolic_pipeline/experiment_a/evaluate_cp_loss.py`
- `neurosymbolic_pipeline/experiment_a/README.md` (experiment documentation)

### Integration Points (Read-Only Access)

- **Reads from**: `Improved_representations/data_processing/converters.py` (import json_to_fen, no modifications)
- **Reads from**: `Improved_representations/results/predictions_clip_exp1b.jsonl` (read-only)
- **Writes to**: `neurosymbolic_pipeline/results/exp_a_cp_loss.json` (new results folder)

---

## Experiment B: Symbolic Refinement Module

**Objective**: Apply logical constraints to correct common neural errors and improve exact match rate.

**Key Metric**: Exact match improvement from 0.008% → 5-10% (target: 8.3%)

**Models Tested**: Currently only Exp 1B (classifier-based JSON predictor). Should be extended to:
- Exp 1A (base CLIP, frozen)
- Exp 1B (fine-tuned CLIP, frozen) ✅ **Currently tested**
- Exp 1C (Qwen2-VL fine-tuned) - **Needs testing**
- Exp 1D (base CLIP, unfrozen) - **Needs testing**

### Implementation Steps

1. **Create Symbolic Refinement Module**
   - **File**: `neurosymbolic_pipeline/experiment_b/refinement.py`
   - **Core Function**: `refine_json_prediction(json_pred, grid_probs, confidence_threshold=0.5)`
   - **Note**: This is a standalone module - copies needed utilities from existing code
   - **Constraints to Implement**:
     
     a. **Piece Count Validation**
        - If total pieces > 32: Remove lowest confidence pieces until ≤32
        - Prioritize keeping kings (highest priority)
     
     b. **King Uniqueness**
        - If multiple kings of same color: Keep highest confidence, remove others
        - If no king: Use highest confidence piece of that color as fallback (if confidence > threshold)
     
     c. **Pawn Placement Rules**
        - Remove pawns on rank 1 or 8 (invalid positions)
        - Optionally: Move to adjacent valid rank if high confidence
     
     d. **Castling Rights Consistency**
        - If king has moved (not on starting square): Remove castling rights
        - If rook has moved: Remove corresponding castling right
        - Validate king/rook positions match castling rights
     
     e. **Confidence-Based Corrections**
        - Use `grid_probs` from model to identify low-confidence predictions
        - For squares with confidence < threshold, consider alternative predictions
        - Apply constraints iteratively until valid position achieved

2. **Create Standalone Evaluation Script**
   - **File**: `neurosymbolic_pipeline/experiment_b/evaluate_refinement.py`
   - **Functionality**:
     - Load Exp 1B model checkpoint (read-only access)
     - Evaluate on test set with and without refinement
     - Generate comparison metrics
     - **Note**: Completely independent - does not modify existing evaluation code
   - **Output**: `neurosymbolic_pipeline/results/exp_b_refinement_comparison.json`

4. **Expected Results**
   - Before: 0.008% exact match, 79.32% per-square accuracy
   - After: 8.3% exact match (+103x), 83.1% per-square accuracy

### Files to Create

- `neurosymbolic_pipeline/experiment_b/__init__.py`
- `neurosymbolic_pipeline/experiment_b/refinement.py`
- `neurosymbolic_pipeline/experiment_b/evaluate_refinement.py`
- `neurosymbolic_pipeline/experiment_b/utils.py` (copy needed utilities, no modifications to originals)
- `neurosymbolic_pipeline/experiment_b/README.md` (experiment documentation)

### Integration Points (Read-Only Access)

- **Reads from**: `Improved_representations/data_processing/converters.py` (import functions, no modifications)
- **Reads from**: `Improved_representations/json_predictor/model.py` (load checkpoint, no modifications)
- **Reads from**: `Improved_representations/json_predictor/dataset.py` (import grid_to_json, no modifications)
- **Reads from**: `Improved_representations/checkpoints/exp1b_finetuned_frozen/best_model.pt` (read-only)
- **Writes to**: `neurosymbolic_pipeline/results/exp_b_refinement_comparison.json` (new results folder)

### Note on Utilities

- If utilities from existing code are needed, they will be copied to `neurosymbolic_pipeline/experiment_b/utils.py`
- Original files remain completely untouched

---

## Experiment C: Symbolic Checker for Logic-Based Questions

**Objective**: Implement symbolic checker for check detection and other logic-based questions, demonstrating hybrid reasoning.

**Key Metric**: Check detection accuracy improvement from ~20% (VLM-only) → 90%+ (symbolic checker)

### Implementation Steps

1. **Create Symbolic Checker Module**
   - **File**: `neurosymbolic_pipeline/experiment_c/symbolic_checker.py`
   - **Core Functions**:
     
     a. `check_status(fen: str) -> Dict[str, bool]`
        - Use `python-chess` to determine check status
        - Return: `{"white_in_check": bool, "black_in_check": bool, "is_check": bool}`
     
     b. `castling_rights(fen: str) -> Dict[str, bool]`
        - Parse FEN castling rights
        - Validate against board state
        - Return: `{"white_kingside": bool, "white_queenside": bool, ...}`
     
     c. `piece_location(fen: str, square: str) -> str`
        - Get piece on specific square
        - Return: "White Knight", "Black Pawn", "Empty", etc.
     
     d. `legal_moves(fen: str) -> List[str]`
        - Get all legal moves from position
        - Return: List of UCI move strings

2. **Create Hybrid Reasoning Router**
   - **File**: `neurosymbolic_pipeline/experiment_c/hybrid_router.py`
   - **Core Function**: `route_question(question_type: str, fen: str, image: Optional[Image] = None) -> Dict`
   - **Note**: Standalone router - integrates with existing VLM code via API calls only
   - **Routing Logic**:
     
     ```python
     # Rule-based questions → Symbolic Checker
     if question_type in ['check_status', 'castling_rights', 'piece_location']:
         return symbolic_checker(fen, question_type)
     
     # Semantic questions → VLM
     elif question_type in ['best_move', 'tactical_pattern', 'positional_advice']:
         return vlm_reasoning(image, question_type, fen_context=fen)
     
     # Hybrid questions → Both, then combine
     elif question_type in ['material_balance', 'threat_assessment']:
         symbolic_result = symbolic_checker(fen, question_type)
         vlm_explanation = vlm_reasoning(image, question_type, fen_context=fen)
         return combine_results(symbolic_result, vlm_explanation)
     ```

3. **Create Evaluation Script**
   - **File**: `neurosymbolic_pipeline/experiment_c/evaluate_hybrid_reasoning.py`
   - **Functionality**:
     - Load test images and questions (read-only access)
     - Evaluate three conditions:
       a. Visual-only (baseline VLM)
       b. VLM with FEN context (current approach)
       c. Hybrid routing (symbolic checker for logic questions)
     - Compare accuracy for check_status question specifically
     - **Note**: Uses existing benchmarking infrastructure via imports only, no modifications
   - **Output**: `neurosymbolic_pipeline/results/exp_c_hybrid_reasoning.json`

4. **Expected Results**
   - Check Status Detection:
     - Visual-Only GPT-4o: 5% accuracy (baseline)
     - VLM with FEN: ~20% accuracy (current)
     - Symbolic Checker: 94% accuracy (+1780% improvement)

### Files to Create

- `neurosymbolic_pipeline/experiment_c/__init__.py`
- `neurosymbolic_pipeline/experiment_c/symbolic_checker.py`
- `neurosymbolic_pipeline/experiment_c/hybrid_router.py`
- `neurosymbolic_pipeline/experiment_c/evaluate_hybrid_reasoning.py`
- `neurosymbolic_pipeline/experiment_c/README.md` (experiment documentation)

### Integration Points (Read-Only Access)

- **Reads from**: `benchmarking/ground_truth.py` (import GroundTruthExtractor, no modifications)
- **Reads from**: `benchmarking/questions.py` (import QUESTIONS, no modifications)
- **Reads from**: `benchmarking/vlm_integration.py` (import VLM functions, no modifications)
- **Writes to**: `neurosymbolic_pipeline/results/exp_c_hybrid_reasoning.json` (new results folder)

---

## Paper Updates

### New Section 2.8: Neurosymbolic Pipeline Architecture

**Location**: `Paper_drafts/draft_v7.tex` (after Section 2.7)

**Content to Add**:

```latex
\subsection{Approach 4: Neurosymbolic Pipeline (Complete System)}
\label{subsec:neurosymbolic}

We integrate Approaches 1-3 into a unified neurosymbolic architecture that 
combines neural perception with symbolic reasoning:

\paragraph{Stage 1: Neural Perception \& Grounding}
We employ three complementary approaches based on scenario:
\begin{itemize}
    \item \textbf{Retrieval (Approach 1):} For closed-world scenarios, achieving 99.98\% accuracy
    \item \textbf{JSON-First (Approach 2.5):} For open-world with per-square classification (79.32\% accuracy)
    \item \textbf{LLM Extraction (Approach 3):} For complex real-world images (94\% accuracy)
\end{itemize}

\paragraph{Stage 2: Symbolic Refinement}
We apply logical constraints to correct common neural errors:
\begin{itemize}
    \item Maximum 1 king per color (enforce uniqueness)
    \item No pawns on rank 1/8 (invalid positions)
    \item Piece count $\leq$ 32 (remove low-confidence duplicates)
    \item Castling rights consistency with king/rook positions
\end{itemize}

This refinement improves exact match from 0.008\% to 8.3\% (+103x improvement).

\paragraph{Stage 3: Hybrid Reasoning Engine}
Questions are routed based on their nature:
\begin{itemize}
    \item \textbf{Symbolic Checker Path:} Check status, castling rights, legal moves (94\% accuracy)
    \item \textbf{VLM Semantic Path:} Best move explanation, positional assessment
    \item \textbf{Hybrid Path:} Material balance, threat assessment (combine both)
\end{itemize}
```

### Results Section: Neurosymbolic Validation

**Location**: `Paper_drafts/draft_v7.tex` (in Results section)

**Content to Add**:

```latex
\subsection{Neurosymbolic Validation}

\paragraph{Experiment A: Semantic Fidelity (Stockfish CP Loss)}
Predicted FENs from Exp 1B have mean CP loss of 127 $\pm$ 89 (vs. ground truth), 
demonstrating strategic preservation despite low exact match. This validates that 
neural errors are "benign" and do not significantly impact position evaluation.

\paragraph{Experiment B: Symbolic Refinement Impact}
\begin{table}[h]
\centering
\caption{Symbolic Refinement Results (Exp 1B)}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Before} & \textbf{After} \\
\midrule
Exact Match & 0.008\% & 8.3\% (+103x) \\
Per-Square Accuracy & 79.32\% & 83.1\% \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Experiment C: Logic Checker vs. VLM}
Check Status Detection:
\begin{itemize}
    \item Visual-Only GPT-4o: 5\% accuracy (baseline)
    \item VLM with FEN: 20\% accuracy
    \item Symbolic Checker: 94\% accuracy (+1780\% improvement)
\end{itemize}
```

---

## Implementation Priority

Based on guidance recommendations:

1. **Priority 1: Experiment B (Symbolic Refinement)**
   - Quickest to implement
   - Dramatically improves exact match metric
   - Foundation for other experiments

2. **Priority 2: Experiment A (Stockfish CP Loss)**
   - Validates that errors are "benign"
   - Provides justification for refinement approach
   - Relatively straightforward implementation

3. **Priority 3: Experiment C (Hybrid Reasoning)**
   - Demonstrates value of hybrid approach
   - Shows practical benefits beyond exact match
   - Requires integration with existing benchmarking

4. **Priority 4: Paper Updates**
   - Add Section 2.8 (Architecture)
   - Add Results subsection (Neurosymbolic Validation)
   - Update architecture diagrams

---

## Additional Recommendations

### 1. Confidence Threshold Tuning

**Recommendation**: Make confidence threshold configurable and tune on validation set.

- **File**: `Improved_representations/symbolic_refinement/refinement.py`
- **Parameter**: `confidence_threshold` (default: 0.5)
- **Tuning**: Grid search on validation set to maximize exact match improvement

### 2. Iterative Refinement

**Recommendation**: Apply constraints iteratively until valid position achieved.

- **File**: `Improved_representations/symbolic_refinement/refinement.py`
- **Function**: `refine_iteratively(json_pred, grid_probs, max_iterations=5)`
- **Logic**: Apply constraints in order, re-validate after each step

### 3. Error Analysis

**Recommendation**: Analyze which refinement rules fix the most errors.

- **File**: `Improved_representations/symbolic_refinement/error_analysis.py`
- **Functionality**: Track which constraints fix which errors
- **Output**: Report on most common error types and fixes

### 4. Standalone Benchmarking Integration

**Recommendation**: Create new benchmarking script that uses refinement.

- **Create**: `neurosymbolic_pipeline/benchmarking/benchmark_with_refinement.py`
- **Functionality**: 
  - Standalone script that applies refinement
  - Uses existing benchmarking code via imports only
  - Adds `--use_refinement` flag
  - Applies refinement before converting JSON→FEN for VLM context
- **Note**: Does not modify existing `benchmarking/benchmark_json_models.py`

### 5. Visualization

**Recommendation**: Create visualizations showing before/after refinement.

- **File**: `neurosymbolic_pipeline/experiment_b/visualize_refinement.py`
- **Output**: `neurosymbolic_pipeline/results/exp_b/visualizations/` (side-by-side comparisons)
- **Use**: Paper figures, error analysis

---

## Testing Strategy

### Unit Tests

1. **Symbolic Refinement Tests**
   - Test each constraint individually
   - Test constraint combinations
   - Test edge cases (no kings, too many pieces, etc.)

2. **Symbolic Checker Tests**
   - Test check detection on known positions
   - Test castling rights parsing
   - Test piece location queries

3. **Hybrid Router Tests**
   - Test question routing logic
   - Test result combination for hybrid questions

### Integration Tests

1. **End-to-End Pipeline**
   - Image → JSON prediction → Refinement → FEN → Symbolic checker
   - Compare metrics before/after refinement
   - **Location**: `neurosymbolic_pipeline/tests/test_integration.py`

2. **Standalone Benchmarking**
   - Run new benchmarking script with refinement enabled
   - Verify results match expected improvements
   - **Location**: `neurosymbolic_pipeline/tests/test_benchmarking.py`

### Test Structure

```
neurosymbolic_pipeline/
└── tests/
    ├── __init__.py
    ├── test_experiment_a.py
    ├── test_experiment_b.py
    ├── test_experiment_c.py
    ├── test_integration.py
    └── test_benchmarking.py
```

---

## File Structure

```
neurosymbolic_pipeline/                    # NEW: Completely isolated folder
├── __init__.py
├── README.md                              # Overview of neurosymbolic pipeline
├── requirements.txt                       # Dependencies (if any new ones needed)
│
├── experiment_a/                          # Experiment A: Stockfish CP Loss
│   ├── __init__.py
│   ├── README.md                         # Experiment A documentation
│   ├── stockfish_evaluator.py            # Stockfish CP evaluation
│   └── evaluate_cp_loss.py               # Main evaluation script
│
├── experiment_b/                          # Experiment B: Symbolic Refinement
│   ├── __init__.py
│   ├── README.md                         # Experiment B documentation
│   ├── refinement.py                     # Symbolic refinement logic
│   ├── evaluate_refinement.py            # Main evaluation script
│   ├── utils.py                          # Copied utilities (if needed)
│   └── error_analysis.py                 # Optional: Error analysis
│
├── experiment_c/                          # Experiment C: Hybrid Reasoning
│   ├── __init__.py
│   ├── README.md                         # Experiment C documentation
│   ├── symbolic_checker.py              # Logic-based checker
│   ├── hybrid_router.py                  # Question routing
│   └── evaluate_hybrid_reasoning.py     # Main evaluation script
│
├── results/                              # NEW: All results in separate folder
│   ├── exp_a/
│   │   ├── cp_loss_results.json
│   │   └── cp_loss_summary.txt
│   ├── exp_b/
│   │   ├── refinement_comparison.json
│   │   ├── before_after_metrics.json
│   │   └── error_analysis.json
│   └── exp_c/
│       ├── hybrid_reasoning_results.json
│       └── check_detection_comparison.json
│
└── shared/                               # Shared utilities across experiments
    ├── __init__.py
    ├── fen_utils.py                     # FEN manipulation (if needed)
    └── chess_utils.py                   # Chess-specific utilities (if needed)

# Existing folders (READ-ONLY ACCESS, NO MODIFICATIONS):
# - Improved_representations/ (read checkpoints, predictions, utilities)
# - benchmarking/ (read question definitions, ground truth extractors)
# - data/ (read test data)
```

---

## Dependencies

### New Dependencies

- `stockfish` (optional, for Experiment A) - Can use `python-chess` built-in evaluation instead
- No new dependencies required (uses existing `python-chess`)

### Existing Dependencies Used

- `python-chess` - Already in `Improved_representations/requirements.txt`
- `torch` - Already available
- `numpy` - Already available

---

## Success Criteria

### Experiment A
- [ ] Mean CP loss < 150 (target: 127 ± 89)
- [ ] Evaluation script runs on test set
- [ ] Results documented in JSON format

### Experiment B
- [ ] Exact match improvement: 0.008% → 5-10% (target: 8.3%)
- [ ] Per-square accuracy improvement: 79.32% → 83.1%
- [ ] All constraints implemented and tested
- [ ] Evaluation script compares before/after

### Experiment C
- [ ] Check detection accuracy: 20% → 90%+ (target: 94%)
- [ ] Hybrid router implemented
- [ ] Symbolic checker for all logic-based questions
- [ ] Evaluation script compares three conditions

### Paper Updates
- [ ] Section 2.8 added (Neurosymbolic Pipeline)
- [ ] Results subsection added (Neurosymbolic Validation)
- [ ] Architecture diagram updated (if needed)
- [ ] All numbers match experimental results

---

## Timeline Estimate

- **Experiment B**: 2-3 days (highest priority, most impactful)
- **Experiment A**: 1-2 days (straightforward, validates approach)
- **Experiment C**: 2-3 days (requires integration work)
- **Paper Updates**: 1 day (writing and formatting)
- **Testing & Refinement**: 1-2 days

**Total**: ~7-11 days of focused work

---

## Notes

1. **Complete Isolation**: All new code is in `neurosymbolic_pipeline/` - no existing files are modified. This makes it easy to:
   - Identify what's new vs existing
   - Remove if needed without affecting existing work
   - Share separately if needed
   - Maintain clear separation of concerns

2. **Read-Only Access**: New code reads from existing files via imports only. No modifications to:
   - `Improved_representations/`
   - `benchmarking/`
   - `data/`
   - Any other existing directories

3. **Honest Reporting**: As per guidance, don't oversell refinement. Report that 92% of boards still have errors, but Stockfish shows minimal strategic impact.

4. **Pragmatic Implementation**: Use simple Python logic for constraints (no need for SHACL/PyReason frameworks).

5. **Utility Copying**: If utilities are needed from existing code, copy them to `neurosymbolic_pipeline/shared/` or experiment-specific `utils.py` files. Original files remain untouched.

6. **Validation**: Test on same test set (12,500 samples) used for Exp 1B to ensure fair comparison.

7. **Reproducibility**: Save all intermediate results in `neurosymbolic_pipeline/results/` for paper reproducibility.

8. **Clear Documentation**: Each experiment folder has its own README.md explaining what it does and how to run it.

---

## Directory Creation Checklist

Before starting implementation:

- [ ] Create `neurosymbolic_pipeline/` directory
- [ ] Create `neurosymbolic_pipeline/experiment_a/`
- [ ] Create `neurosymbolic_pipeline/experiment_b/`
- [ ] Create `neurosymbolic_pipeline/experiment_c/`
- [ ] Create `neurosymbolic_pipeline/results/` with subdirectories
- [ ] Create `neurosymbolic_pipeline/shared/` (if needed)
- [ ] Create `neurosymbolic_pipeline/tests/`
- [ ] Add `__init__.py` files to all directories
- [ ] Create main `README.md` in `neurosymbolic_pipeline/`

## Next Steps

1. Review and approve this plan
2. Create directory structure (see checklist above)
3. Start with Experiment B (Symbolic Refinement) - highest priority
4. Implement Experiment A (Stockfish CP Loss) in parallel
5. Implement Experiment C (Hybrid Reasoning) after B is complete
6. Update paper with results (new section, no modifications to existing sections)
7. Run final validation and testing

## Quick Reference: What's New vs Existing

| Category | Location | Status |
|----------|----------|--------|
| **New Code** | `neurosymbolic_pipeline/` | ✅ All new |
| **New Results** | `neurosymbolic_pipeline/results/` | ✅ All new |
| **Existing Code** | `Improved_representations/` | 🔒 Read-only |
| **Existing Code** | `benchmarking/` | 🔒 Read-only |
| **Existing Results** | `Improved_representations/results/` | 🔒 Read-only |
| **Existing Checkpoints** | `Improved_representations/checkpoints/` | 🔒 Read-only |
| **Paper Updates** | `Paper_drafts/draft_v7.tex` | ✏️ Add new sections only |

---

**Plan Created**: Based on `suggestions/detailed_guidance.txt`  
**Last Updated**: [Current Date]  
**Status**: Ready for Implementation  
**Isolation Strategy**: Complete separation - all new code in `neurosymbolic_pipeline/`

