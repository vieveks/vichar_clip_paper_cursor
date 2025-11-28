
class FENTokenizer:
    """
    Tokenizer for Chess FEN strings (Board placement part).
    Uses expanded FEN format where numbers are replaced with '1' tokens.
    Vocabulary: p, n, b, r, q, k, P, N, B, R, Q, K, 1 (empty), /, w, b, -, <SOS>, <EOS>, <PAD>
    Note: Digits 2-8 are removed since we use expanded FEN format (e.g., '3' -> '111')
    """
    def __init__(self):
        self.vocab = [
            "<PAD>", "<SOS>", "<EOS>",
            "p", "n", "b", "r", "q", "k",
            "P", "N", "B", "R", "Q", "K",
            "1",  # Empty square token (replaces all digits 1-8 in expanded FEN)
            "/", "w", "b", "-"
        ]
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        
        self.pad_token_id = self.token_to_id["<PAD>"]
        self.sos_token_id = self.token_to_id["<SOS>"]
        self.eos_token_id = self.token_to_id["<EOS>"]
        
    def encode(self, fen):
        """
        Encode a FEN string into a list of token IDs.
        Focuses on the board placement part (first field of FEN).
        """
        # Take only the board placement part if full FEN is provided
        board_fen = fen.split()[0]
        
        tokens = [self.sos_token_id]
        for char in board_fen:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                # Handle unknown characters if any (shouldn't happen for valid FEN)
                pass
        tokens.append(self.eos_token_id)
        return tokens
    
    def decode(self, token_ids):
        """
        Decode a list of token IDs back into a FEN string.
        """
        tokens = []
        for tid in token_ids:
            if tid == self.eos_token_id:
                break
            if tid == self.sos_token_id or tid == self.pad_token_id:
                continue
            tokens.append(self.id_to_token.get(tid, ""))
        return "".join(tokens)
    
    def __len__(self):
        return len(self.vocab)
