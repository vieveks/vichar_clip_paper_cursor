"""
Enhanced question definitions with more material balance and similar questions.
These questions focus on material understanding which showed strong performance.
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
    # Original material balance
    {
        "id": 4,
        "type": "material_balance",
        "prompt": "Who has more material (using standard piece values: Pawn=1, Knight=3, Bishop=3, Rook=5, Queen=9)? Answer: 'White', 'Black', or 'Equal'.",
        "scoring_type": "llm_judge",
        "weight": 1.0
    },
    # Additional material balance questions
    {
        "id": 9,
        "type": "material_advantage",
        "prompt": "What is the material advantage? Calculate using standard values (Pawn=1, Knight=3, Bishop=3, Rook=5, Queen=9) and answer with the point difference (e.g., 'White +3', 'Black +2', or 'Equal').",
        "scoring_type": "llm_judge",
        "weight": 1.0
    },
    {
        "id": 10,
        "type": "material_count_white",
        "prompt": "What is White's total material value using standard piece values (Pawn=1, Knight=3, Bishop=3, Rook=5, Queen=9)? Answer with just the number.",
        "scoring_type": "llm_judge",
        "weight": 1.0
    },
    {
        "id": 11,
        "type": "material_count_black",
        "prompt": "What is Black's total material value using standard piece values (Pawn=1, Knight=3, Bishop=3, Rook=5, Queen=9)? Answer with just the number.",
        "scoring_type": "llm_judge",
        "weight": 1.0
    },
    {
        "id": 12,
        "type": "queen_count",
        "prompt": "How many queens does White have? How many queens does Black have? Answer in format: 'White: X, Black: Y'.",
        "scoring_type": "llm_judge",
        "weight": 1.0
    },
    {
        "id": 13,
        "type": "minor_piece_balance",
        "prompt": "Who has more minor pieces (knights and bishops combined)? Answer: 'White', 'Black', or 'Equal'.",
        "scoring_type": "llm_judge",
        "weight": 1.0
    },
    {
        "id": 14,
        "type": "rook_count",
        "prompt": "How many rooks does White have? How many rooks does Black have? Answer in format: 'White: X, Black: Y'.",
        "scoring_type": "llm_judge",
        "weight": 1.0
    },
    {
        "id": 15,
        "type": "pawn_advantage",
        "prompt": "Who has more pawns? Answer: 'White', 'Black', or 'Equal'.",
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

def get_material_questions():
    """Get all material-related questions (for focused evaluation)."""
    material_types = ["material_balance", "material_advantage", "material_count_white", 
                     "material_count_black", "queen_count", "minor_piece_balance", 
                     "rook_count", "pawn_advantage"]
    return [q for q in QUESTIONS if q["type"] in material_types]

