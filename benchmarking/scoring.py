"""
Scoring utilities for evaluating VLM responses against ground truth.
"""

import re
from typing import Dict, List, Optional, Any
from difflib import SequenceMatcher


class ResponseScorer:
    """Scores VLM responses against ground truth answers."""
    
    def __init__(self):
        """Initialize the scorer."""
        pass
    
    def similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (0-1)."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def extract_move_from_text(self, text: str) -> Optional[str]:
        """Extract chess move from text (e.g., 'e4', 'Nf3', 'e2e4')."""
        # Common move patterns
        patterns = [
            r'\b([a-h][1-8])\s*[-x]\s*([a-h][1-8])\b',  # e2-e4, e2xe4
            r'\b([a-h][1-8])\b',  # e4, d4
            r'\b([NBRQK][a-h]?[1-8]?[x-]?[a-h][1-8])\b',  # Nf3, Bxe5
            r'\b([Oo]-[Oo](-[Oo])?)\b',  # O-O, O-O-O
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).upper()
        
        return None
    
    def score_piece_locations(self, response: str, ground_truth: Dict[str, List[str]]) -> float:
        """
        Score piece location description.
        
        Args:
            response: VLM response text
            ground_truth: Dict mapping piece symbols to square names
            
        Returns:
            Score from 0 to 1
        """
        # Extract mentioned pieces and squares from response
        response_lower = response.lower()
        mentioned_pieces = []
        mentioned_squares = []
        
        # Check for piece mentions
        piece_names = {
            'pawn': 'P', 'rook': 'R', 'knight': 'N', 'bishop': 'B',
            'queen': 'Q', 'king': 'K'
        }
        
        for piece_name, symbol in piece_names.items():
            if piece_name in response_lower:
                mentioned_pieces.append(symbol)
        
        # Check for square mentions (a1-h8)
        square_pattern = r'\b([a-h][1-8])\b'
        squares = re.findall(square_pattern, response_lower)
        mentioned_squares.extend(squares)
        
        # Calculate coverage
        total_pieces = sum(len(squares) for squares in ground_truth.values())
        if total_pieces == 0:
            return 0.0
        
        # Simple scoring: check if key pieces and squares are mentioned
        score = 0.0
        for piece_symbol, squares in ground_truth.items():
            if piece_symbol.upper() in mentioned_pieces or piece_symbol.lower() in mentioned_pieces:
                score += 0.3
            for square in squares:
                if square.lower() in mentioned_squares:
                    score += 0.7 / len(squares)
        
        return min(score / total_pieces, 1.0)
    
    def score_best_move(self, response: str, ground_truth: str) -> float:
        """
        Score best move answer.
        
        Args:
            response: VLM response
            ground_truth: Best move in UCI notation (e.g., "e2e4")
            
        Returns:
            Score from 0 to 1
        """
        extracted_move = self.extract_move_from_text(response)
        if not extracted_move:
            return 0.0
        
        # Convert UCI to algebraic if needed
        ground_truth_algebraic = self._uci_to_algebraic(ground_truth)
        extracted_algebraic = self._uci_to_algebraic(extracted_move) if len(extracted_move) > 2 else extracted_move
        
        # Check exact match
        if extracted_move.upper() == ground_truth.upper():
            return 1.0
        
        # Check algebraic match
        if extracted_algebraic and ground_truth_algebraic:
            if extracted_algebraic.upper() == ground_truth_algebraic.upper():
                return 1.0
        
        # Check if move is mentioned in response
        if ground_truth.lower() in response.lower() or ground_truth_algebraic.lower() in response.lower():
            return 0.8
        
        return 0.0
    
    def score_evaluation(self, response: str, ground_truth: Dict) -> float:
        """
        Score position evaluation answer.
        
        Args:
            response: VLM response
            ground_truth: Dict with 'score' (centipawns) and optionally 'mate'
            
        Returns:
            Score from 0 to 1
        """
        # Extract numerical evaluation from response
        score_pattern = r'([+-]?\d+\.?\d*)\s*(centipawns?|cp|pawns?|points?)'
        match = re.search(score_pattern, response, re.IGNORECASE)
        
        if match:
            try:
                response_score = float(match.group(1))
                # Convert to centipawns if needed
                if 'pawn' in match.group(2).lower() and abs(response_score) < 100:
                    response_score *= 100
                
                ground_truth_score = ground_truth.get('score', 0)
                
                # Calculate error
                error = abs(response_score - ground_truth_score)
                # Normalize: full score if error < 50cp, partial if < 200cp
                if error < 50:
                    return 1.0
                elif error < 200:
                    return 0.5
                else:
                    return max(0.0, 1.0 - error / 1000.0)
            except:
                pass
        
        # Check for qualitative assessment
        response_lower = response.lower()
        ground_truth_score = ground_truth.get('score', 0)
        
        if ground_truth_score > 100:
            if any(word in response_lower for word in ['white', 'advantage', 'better', 'winning']):
                return 0.6
        elif ground_truth_score < -100:
            if any(word in response_lower for word in ['black', 'advantage', 'better', 'winning']):
                return 0.6
        else:
            if any(word in response_lower for word in ['equal', 'balanced', 'even']):
                return 0.6
        
        return 0.0
    
    def score_material_count(self, response: str, ground_truth: Dict[str, int]) -> float:
        """
        Score material count answer.
        
        Args:
            response: VLM response
            ground_truth: Dict with 'white' and 'black' material counts
            
        Returns:
            Score from 0 to 1
        """
        # Extract numbers from response
        numbers = re.findall(r'\d+', response)
        
        if len(numbers) >= 2:
            try:
                white_count = int(numbers[0])
                black_count = int(numbers[1])
                
                # Check if order matches (white first or black first)
                white_mentioned = 'white' in response.lower()
                black_mentioned = 'black' in response.lower()
                
                if white_mentioned and not black_mentioned:
                    # Assume first number is white
                    if abs(white_count - ground_truth['white']) <= 1:
                        return 0.5
                elif black_mentioned and not white_mentioned:
                    # Assume first number is black
                    if abs(black_count - ground_truth['black']) <= 1:
                        return 0.5
                else:
                    # Try both orders
                    if (abs(white_count - ground_truth['white']) <= 1 and 
                        abs(black_count - ground_truth['black']) <= 1):
                        return 1.0
                    elif (abs(white_count - ground_truth['black']) <= 1 and 
                          abs(black_count - ground_truth['white']) <= 1):
                        return 0.8
            except:
                pass
        
        # Partial credit for mentioning material
        if any(word in response.lower() for word in ['material', 'pieces', 'pawns', 'queen', 'rook']):
            return 0.3
        
        return 0.0
    
    def score_check_status(self, response: str, ground_truth: Dict[str, bool]) -> float:
        """
        Score check status answer.
        
        Args:
            response: VLM response
            ground_truth: Dict with check status information
            
        Returns:
            Score from 0 to 1
        """
        response_lower = response.lower()
        is_check = ground_truth.get('is_check', False)
        
        if is_check:
            if any(word in response_lower for word in ['check', 'in check', 'checked']):
                # Check if correct color mentioned
                if ground_truth.get('white_in_check'):
                    if 'white' in response_lower:
                        return 1.0
                    return 0.7
                elif ground_truth.get('black_in_check'):
                    if 'black' in response_lower:
                        return 1.0
                    return 0.7
                return 0.5
        else:
            if any(word in response_lower for word in ['no check', 'not in check', 'not checked']):
                return 1.0
            if 'check' not in response_lower:
                return 0.5
        
        return 0.0
    
    def score_castling_rights(self, response: str, ground_truth: Dict[str, bool]) -> float:
        """
        Score castling rights answer.
        
        Args:
            response: VLM response
            ground_truth: Dict with castling rights
            
        Returns:
            Score from 0 to 1
        """
        response_lower = response.lower()
        score = 0.0
        total_rights = 4
        
        # Check each castling right
        if ground_truth.get('white_kingside'):
            if 'white' in response_lower and ('kingside' in response_lower or 'o-o' in response_lower):
                score += 0.25
        if ground_truth.get('white_queenside'):
            if 'white' in response_lower and ('queenside' in response_lower or 'o-o-o' in response_lower):
                score += 0.25
        if ground_truth.get('black_kingside'):
            if 'black' in response_lower and ('kingside' in response_lower or 'o-o' in response_lower):
                score += 0.25
        if ground_truth.get('black_queenside'):
            if 'black' in response_lower and ('queenside' in response_lower or 'o-o-o' in response_lower):
                score += 0.25
        
        return score
    
    def score_knight_attacks(self, response: str, ground_truth: Dict[str, List[str]]) -> float:
        """
        Score knight attack answer.
        
        Args:
            response: VLM response
            ground_truth: Dict mapping knight squares to attacked squares
            
        Returns:
            Score from 0 to 1
        """
        # Extract squares mentioned in response
        square_pattern = r'\b([a-h][1-8])\b'
        mentioned_squares = re.findall(square_pattern, response.lower())
        
        if not mentioned_squares:
            return 0.0
        
        # Check if any attacked squares are mentioned
        all_attacked = []
        for knight_sq, attacked_sqs in ground_truth.items():
            all_attacked.extend(attacked_sqs)
        
        if not all_attacked:
            return 0.0
        
        # Calculate overlap
        correct_mentions = sum(1 for sq in mentioned_squares if sq.upper() in [s.upper() for s in all_attacked])
        total_attacked = len(all_attacked)
        
        return min(correct_mentions / max(total_attacked, 1), 1.0)
    
    def _uci_to_algebraic(self, uci_move: str) -> Optional[str]:
        """Convert UCI move to algebraic notation (simplified)."""
        if len(uci_move) < 4:
            return None
        
        try:
            from_square = uci_move[:2]
            to_square = uci_move[2:4]
            
            # Simple conversion (doesn't handle all cases)
            return f"{from_square}-{to_square}"
        except:
            return None

