"""
Hybrid Reasoning Router for Experiment C.

Routes questions to appropriate solver (symbolic checker or VLM) based on question type.
"""

from typing import Dict, Optional, Any
from PIL import Image
import sys
from pathlib import Path

# Import symbolic checker
sys.path.insert(0, str(Path(__file__).parent))
from symbolic_checker import SymbolicChecker


class HybridRouter:
    """Routes questions to appropriate solver."""
    
    # Question types that should use symbolic checker
    SYMBOLIC_QUESTIONS = [
        'check_status',
        'castling_rights',
        'castling_available',
        'piece_location',
        'piece_on_square',
        'legal_moves'
    ]
    
    # Question types that should use VLM
    VLM_QUESTIONS = [
        'best_move',
        'tactical_pattern',
        'positional_advice',
        'explanation'
    ]
    
    # Question types that can use both (hybrid)
    HYBRID_QUESTIONS = [
        'material_balance',
        'material_count',
        'threat_assessment'
    ]
    
    def __init__(self, vlm_function=None):
        """
        Initialize hybrid router.
        
        Args:
            vlm_function: Function to call VLM (optional, can be set later)
        """
        self.symbolic_checker = SymbolicChecker()
        self.vlm_function = vlm_function
    
    def route_question(
        self,
        question_type: str,
        fen: str,
        image: Optional[Image.Image] = None,
        question_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Route question to appropriate solver.
        
        Args:
            question_type: Type of question (e.g., 'check_status', 'best_move')
            fen: FEN string of the position
            image: Optional image (for VLM questions)
            question_text: Optional question text (for VLM questions)
        
        Returns:
            Dict with answer and metadata
        """
        question_type = question_type.lower()
        
        if question_type in self.SYMBOLIC_QUESTIONS:
            return self._symbolic_answer(question_type, fen)
        
        elif question_type in self.VLM_QUESTIONS:
            return self._vlm_answer(question_type, fen, image, question_text)
        
        elif question_type in self.HYBRID_QUESTIONS:
            return self._hybrid_answer(question_type, fen, image, question_text)
        
        else:
            # Default to VLM if unknown
            return self._vlm_answer(question_type, fen, image, question_text)
    
    def _symbolic_answer(self, question_type: str, fen: str) -> Dict[str, Any]:
        """Get answer from symbolic checker."""
        if question_type == 'check_status':
            result = self.symbolic_checker.check_status(fen)
            answer = self._format_check_status(result)
        
        elif question_type in ['castling_rights', 'castling_available']:
            result = self.symbolic_checker.castling_rights(fen)
            answer = self._format_castling_rights(result)
        
        elif question_type in ['piece_location', 'piece_on_square']:
            # Extract square from question if needed
            square = 'e4'  # Default, should be extracted from question_text
            result = self.symbolic_checker.piece_location(fen, square)
            answer = result
        
        elif question_type == 'legal_moves':
            moves = self.symbolic_checker.legal_moves(fen)
            answer = f"Legal moves: {', '.join(moves[:10])}"  # Show first 10
        
        else:
            answer = "Unknown question type"
            result = {}
        
        return {
            'answer': answer,
            'method': 'symbolic',
            'raw_result': result
        }
    
    def _vlm_answer(
        self,
        question_type: str,
        fen: str,
        image: Optional[Image.Image],
        question_text: Optional[str]
    ) -> Dict[str, Any]:
        """Get answer from VLM."""
        if self.vlm_function is None:
            return {
                'answer': 'VLM function not available',
                'method': 'vlm',
                'error': 'No VLM function provided'
            }
        
        # Call VLM with FEN context
        try:
            answer = self.vlm_function(
                image=image,
                question=question_text or f"Answer this chess question: {question_type}",
                fen_context=fen
            )
            return {
                'answer': answer,
                'method': 'vlm',
                'fen_context': fen
            }
        except Exception as e:
            return {
                'answer': f'Error calling VLM: {e}',
                'method': 'vlm',
                'error': str(e)
            }
    
    def _hybrid_answer(
        self,
        question_type: str,
        fen: str,
        image: Optional[Image.Image],
        question_text: Optional[str]
    ) -> Dict[str, Any]:
        """Get answer from both symbolic checker and VLM, then combine."""
        # Get symbolic result
        symbolic_result = self._symbolic_answer(question_type, fen)
        
        # Get VLM result
        vlm_result = self._vlm_answer(question_type, fen, image, question_text)
        
        # Combine results
        if question_type == 'material_balance':
            # Use symbolic for exact count, VLM for explanation
            answer = f"{symbolic_result['answer']}. {vlm_result['answer']}"
        else:
            # Default combination
            answer = f"Symbolic: {symbolic_result['answer']}. VLM: {vlm_result['answer']}"
        
        return {
            'answer': answer,
            'method': 'hybrid',
            'symbolic': symbolic_result,
            'vlm': vlm_result
        }
    
    def _format_check_status(self, result: Dict[str, bool]) -> str:
        """Format check status result."""
        if result['is_check']:
            if result['white_in_check']:
                return "Yes, White king is in check"
            elif result['black_in_check']:
                return "Yes, Black king is in check"
            else:
                return "Yes, one of the kings is in check"
        else:
            return "No, neither king is in check"
    
    def _format_castling_rights(self, result: Dict[str, bool]) -> str:
        """Format castling rights result."""
        parts = []
        if result['white_kingside']:
            parts.append("White can castle kingside")
        if result['white_queenside']:
            parts.append("White can castle queenside")
        if result['black_kingside']:
            parts.append("Black can castle kingside")
        if result['black_queenside']:
            parts.append("Black can castle queenside")
        
        if not parts:
            return "No castling rights available"
        
        return ". ".join(parts)

