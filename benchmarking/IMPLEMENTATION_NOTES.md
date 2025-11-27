# Implementation Notes

## Architecture Overview

The benchmarking system is designed to prove that providing FEN context improves VLM accuracy on chess questions. The system follows this workflow:

```
Chess Board Image
    ↓
[CLIP Model] → PREDICTED FEN String (not ground truth!)
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

## Key Design Decisions

### 1. FEN as Context, Not Encoder

As specified, the FEN is concatenated to the input prompt rather than being used as an encoder. This is because:
- The FEN signal is sparse and crucial for human understanding
- Directly training the image encoder to learn FEN would be difficult
- Providing FEN as context allows the VLM to leverage structured information

### 2. Using PREDICTED FEN, Not Ground Truth

**Critical:** We use CLIP-predicted FEN for ground truth extraction, NOT the ground truth FEN from the dataset. This ensures:
- We test the full pipeline: image → CLIP → FEN → ground truth
- We measure real-world performance (CLIP may make mistakes)
- We avoid "cheating" by using perfect FEN when testing VLM

**What we DO use from dataset CSV:**
- FEN candidates for CLIP matching (just options to choose from)
- Best moves (only if predicted FEN matches dataset FEN)

**What we DON'T use:**
- Ground truth FEN for extracting ground truth answers (would be cheating!)

### 3. Two-Tier Testing

Each question is tested twice:
- **Without FEN**: Pure VLM with image + question
- **With FEN**: VLM with image + question + PREDICTED FEN context

This allows direct comparison of improvement.

### 4. Ground Truth Sources

Ground truth is extracted from:
- **pychess**: For board state analysis (piece locations, attacks, check status, castling rights)
- **Lichess API**: For engine analysis (best moves, evaluations, threats)
- **Dataset CSV**: For best moves ONLY if predicted FEN matches dataset FEN (acceptable because we're using predicted FEN)

### 5. Scoring Methodology

Each question type has a custom scorer:
- **Exact matches**: For moves, material counts
- **Similarity matching**: For descriptions
- **Range-based**: For evaluations (within 50cp = perfect, 200cp = partial)

## Question Categories

### Objective Questions (Weight = 1.0)
These can be precisely scored:
1. Piece locations
2. Best move
3. Previous move quality
4. Knight attacks
5. Material count
6. Check status
7. Castling rights
8. Threats

### Subjective Questions (Weight < 1.0)
These are harder to score objectively:
- Winning assessment (weight 0.0)
- Position strength (weight 0.5)

## API Considerations

### Lichess Cloud Evaluation API

- **Rate Limiting**: Built-in delays between calls
- **Timeout**: 10 seconds per request
- **Error Handling**: Graceful fallback if API unavailable
- **FEN Format**: Strips move counters for API compatibility

### Alternative: Local Engine

For production use, consider using a local chess engine (Stockfish) instead of Lichess API:
- No rate limits
- Faster responses
- More control over analysis depth

## VLM Integration

### LLaVA Model

- **Default**: `llava-hf/llava-1.5-7b-hf` (7B parameters)
- **Memory**: Requires ~13GB VRAM
- **Alternative**: Can use smaller models or mock for testing

### Prompt Format

**Without FEN:**
```
{question_prompt}
```

**With FEN:**
```
{question_prompt}

FEN representation: {predicted_fen_string}
```

## Performance Considerations

### Batch Processing

Currently processes images sequentially. For large-scale benchmarking:
- Implement batch processing for VLM
- Parallelize ground truth extraction
- Cache FEN extractions

### Caching

Consider implementing:
- FEN extraction cache (same image = same FEN)
- Ground truth cache (same FEN = same answers)
- VLM response cache (for testing different scorers)

## Extensibility

### Adding New Questions

1. Add question definition to `questions.py`
2. Implement ground truth extraction in `ground_truth.py`
3. Implement scorer in `scoring.py`
4. Update `benchmark.py` to handle new question type

### Adding New VLMs

1. Create new VLM class in `vlm_integration.py`
2. Implement `answer_question()` method
3. Update `ChessVLMBenchmark` to support new VLM

### Adding New Ground Truth Sources

1. Extend `GroundTruthExtractor` class
2. Add new extraction methods
3. Update question handlers in `benchmark.py`

## Known Limitations

1. **Previous Move Quality**: Requires previous FEN, not always available
2. **Threat Analysis**: Complex to score objectively
3. **API Dependencies**: Lichess API may be unavailable
4. **Model Size**: LLaVA requires significant resources
5. **Scoring Subjectivity**: Some questions are inherently subjective
6. **CLIP FEN Accuracy**: If CLIP predicts wrong FEN, ground truth will be wrong (this is intentional - we're testing the full pipeline)

## Future Improvements

1. **Local Engine Integration**: Use Stockfish instead of Lichess API
2. **Batch Processing**: Process multiple images in parallel
3. **Advanced Scoring**: Use LLM-based scoring for subjective questions
4. **Visualization**: Generate comparison charts and graphs
5. **Statistical Analysis**: Confidence intervals, significance testing
6. **Multi-Model Comparison**: Test multiple VLMs simultaneously
7. **FEN Accuracy Tracking**: Track CLIP FEN prediction accuracy separately
