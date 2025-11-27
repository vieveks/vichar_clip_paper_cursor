"""
Question definitions for chess VLM benchmarking.
Each question has a type, prompt template, and scoring method.
"""

QUESTIONS = [
    {
        "id": 1,
        "type": "piece_location",
        "prompt": "Describe which piece is where on this chess board.",
        "scoring_type": "exact_match",  # Can be verified from FEN
        "weight": 1.0
    },
    {
        "id": 2,
        "type": "best_move",
        "prompt": "What is the best move in this position?",
        "scoring_type": "engine_analysis",  # From Lichess engine
        "weight": 1.0
    },
    {
        "id": 3,
        "type": "winning_assessment",
        "prompt": "Who seems to be winning in this position?",
        "scoring_type": "evaluation_score",  # From engine evaluation
        "weight": 0.0  # Subjective, hard to score objectively
    },
    {
        "id": 4,
        "type": "position_strength",
        "prompt": "How strong is the position for white?",
        "scoring_type": "evaluation_score",  # From engine evaluation
        "weight": 0.5
    },
    {
        "id": 5,
        "type": "previous_move_quality",
        "prompt": "How good was the previous move?",
        "scoring_type": "move_quality",  # Requires move history
        "weight": 1.0
    },
    {
        "id": 6,
        "type": "piece_attacks",
        "prompt": "Which piece is the knight attacking?",
        "scoring_type": "attack_analysis",  # From pychess
        "weight": 1.0
    },
    # Additional 4 questions
    {
        "id": 7,
        "type": "material_count",
        "prompt": "What is the material count for both sides?",
        "scoring_type": "material_count",  # From FEN
        "weight": 1.0
    },
    {
        "id": 8,
        "type": "check_status",
        "prompt": "Is either king in check?",
        "scoring_type": "check_status",  # From pychess
        "weight": 1.0
    },
    {
        "id": 9,
        "type": "castling_rights",
        "prompt": "What are the castling rights for both sides?",
        "scoring_type": "castling_rights",  # From FEN
        "weight": 1.0
    },
    {
        "id": 10,
        "type": "threats",
        "prompt": "What are the main threats in this position?",
        "scoring_type": "threat_analysis",  # From engine analysis
        "weight": 1.0
    }
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

