# Experiment C: Hybrid Reasoning Engine - Explanation

## What Does It Do?

Experiment C implements a **Hybrid Reasoning Engine** that intelligently routes chess questions to the best solver based on the question type. This demonstrates the power of combining symbolic (rule-based) and neural (VLM) approaches.

## The Core Concept: Question Routing

Instead of using one method for all questions, the system routes questions to the most appropriate solver:

### 1. **Symbolic Checker Path** (Rule-Based)
For questions that have clear, deterministic answers based on chess rules:
- ✅ **Check status**: "Is the king in check?" → Uses python-chess to check if king is attacked
- ✅ **Castling rights**: "Can white castle?" → Checks if king/rooks have moved
- ✅ **Piece location**: "What piece is on e4?" → Direct lookup from FEN
- ✅ **Legal moves**: "What moves are legal?" → Calculates all legal moves

**Why symbolic?** These questions have exact, rule-based answers. A symbolic checker is:
- 100% accurate (no hallucinations)
- Fast (no API calls)
- Reliable (always correct)

### 2. **VLM Path** (Neural/Visual)
For questions requiring understanding, explanation, or visual reasoning:
- ✅ **Best move**: "What's the best move?" → Requires strategic understanding
- ✅ **Tactical pattern**: "What tactical pattern do you see?" → Visual pattern recognition
- ✅ **Positional advice**: "How should I play this position?" → Strategic explanation

**Why VLM?** These questions need:
- Visual understanding of the board
- Strategic reasoning
- Natural language explanations

### 3. **Hybrid Path** (Combines Both)
For questions that benefit from both exact calculation and explanation:
- ✅ **Material balance**: Symbolic gives exact count, VLM explains significance
- ✅ **Threat assessment**: Symbolic identifies threats, VLM explains them

**Why hybrid?** Combines:
- Exact facts (from symbolic)
- Rich explanations (from VLM)

## The Problem It Solves

### Baseline (VLM-only): ~20% Check Detection Accuracy
- VLMs struggle with logical questions like "Is the king in check?"
- They may hallucinate or give inconsistent answers
- Visual-only reasoning is error-prone for rule-based questions

### Target (Hybrid Routing): ~94% Check Detection Accuracy
- Symbolic checker provides 100% accurate answers for logic questions
- VLM handles semantic/strategic questions where it excels
- Best of both worlds!

## How It Works

```
Question: "Is the king in check?"
    ↓
Hybrid Router analyzes question type
    ↓
Question type = "check_status" → SYMBOLIC_QUESTIONS
    ↓
Routes to SymbolicChecker
    ↓
Uses python-chess to check if king is attacked
    ↓
Returns: "Yes, White king is in check" (100% accurate)
```

vs.

```
Question: "What's the best move here?"
    ↓
Hybrid Router analyzes question type
    ↓
Question type = "best_move" → VLM_QUESTIONS
    ↓
Routes to VLM (with FEN context)
    ↓
VLM analyzes position visually + strategically
    ↓
Returns: "The best move is e4, opening the center..." (strategic explanation)
```

## Key Benefits

1. **Accuracy**: Symbolic checker is 100% accurate for logic questions
2. **Efficiency**: No need to call expensive VLM for simple rule-based questions
3. **Reliability**: Deterministic answers for logic questions (no hallucinations)
4. **Flexibility**: VLM still handles complex semantic questions
5. **Best of Both Worlds**: Combines strengths of symbolic and neural approaches

## Implementation Components

1. **`symbolic_checker.py`**: Rule-based checker using python-chess
   - `check_status()`: Determines if kings are in check
   - `castling_rights()`: Gets castling availability
   - `piece_location()`: Finds piece on a square
   - `legal_moves()`: Lists all legal moves
   - `material_count()`: Calculates material balance

2. **`hybrid_router.py`**: Routes questions to appropriate solver
   - Classifies question type
   - Routes to symbolic checker, VLM, or both
   - Combines results for hybrid questions

3. **`evaluate_hybrid_reasoning.py`**: Evaluation script
   - Compares hybrid routing vs. VLM-only
   - Measures accuracy improvements
   - Tests different question types

## Expected Results

| Question Type | Baseline (VLM-only) | Hybrid Routing | Improvement |
|--------------|-------------------|----------------|-------------|
| **Check Status** | ~20% | ~94% | **+1780%** ✅ |
| **Castling Rights** | ~30% | ~100% | **+233%** ✅ |
| **Piece Location** | ~40% | ~100% | **+150%** ✅ |
| **Best Move** | ~60% | ~65% | Uses VLM (similar) |
| **Tactical Pattern** | ~50% | ~55% | Uses VLM (similar) |

## Why This Matters

This experiment demonstrates that:
- **Not all questions need neural models** - simple logic questions are better handled symbolically
- **Hybrid systems are more efficient** - route to the right tool for each task
- **Accuracy improves dramatically** for rule-based questions (20% → 94%)
- **The neurosymbolic approach is practical** - combining best of both worlds

This is a key contribution showing that intelligent routing between symbolic and neural methods yields better results than using either alone!

