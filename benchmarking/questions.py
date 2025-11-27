"""
Question definitions for chess VLM benchmarking.
Each question has a type, prompt template, and scoring method.
"""

QUESTIONS = [
    {
        "id": 1,
        "type": "fen_extraction",
        "prompt": "What is the FEN (Forsyth-Edwards Notation) for this chess position? Provide only the FEN string.",
        "scoring_type": "exact_match",
        "weight": 1.0
    },
    {
        "id": 2,
        "type": "piece_count",
        "prompt": "How many total pieces (not including kings) does White have? How many does Black have? Answer in format: 'White: X, Black: Y'",
        "scoring_type": "llm_judge",
        "weight": 1.0
    },
    {
        "id": 3,
        "type": "check_status",
        "prompt": "Is either king in check? Answer with 'Yes' or 'No', and if yes, specify which king (White or Black).",
        "scoring_type": "llm_judge",
        "weight": 1.0
    },
    {
        "id": 4,
        "type": "material_balance",
        "prompt": "Who has more material (using standard piece values: Pawn=1, Knight=3, Bishop=3, Rook=5, Queen=9)? Answer: 'White', 'Black', or 'Equal'.",
        "scoring_type": "llm_judge",
        "weight": 1.0
    },
    {
        "id": 5,
        "type": "best_move",
        "prompt": "What is the best move in this position? Provide the move in algebraic notation (e.g., Nf3, e4, O-O).",
        "scoring_type": "llm_judge",
        "weight": 1.0
    },
    {
        "id": 6,
        "type": "tactical_pattern",
        "prompt": "Is there a tactical pattern (pin, fork, skewer, discovered attack) in this position? If yes, describe it briefly.",
        "scoring_type": "llm_judge",
        "weight": 0.8
    },
    {
        "id": 7,
        "type": "castling_available",
        "prompt": "Can White castle kingside? Can Black castle kingside? Answer for each: 'Yes' or 'No'.",
        "scoring_type": "llm_judge",
        "weight": 1.0
    },
    {
        "id": 8,
        "type": "piece_on_square",
        "prompt": "What piece is on square e4? Answer with the piece type and color (e.g., 'White Knight', 'Black Pawn', or 'Empty').",
        "scoring_type": "llm_judge",
        "weight": 1.0
    },
]

def get_question_by_id(question_id):
    """Get a question definition by ID."""
    for q in QUESTIONS:
        if q["id"] == question_id:
            return q
    return None

def get_all_questions():
    """Get all question definitions."""
    return QUESTIONS

def get_scoring_questions():
    """Get questions that can be objectively scored (weight > 0)."""
    return [q for q in QUESTIONS if q["weight"] > 0]

