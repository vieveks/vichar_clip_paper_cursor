import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from FEN_generator.tokenizer import FENTokenizer
from pathlib import Path

def analyze_data():
    tokenizer = FENTokenizer()
    data_dir = Path("data/hf_chess_puzzles")
    
    # Check train, val, and test
    for split in ["train", "validation", "test"]:
        csv_file = data_dir / f"{split}.csv"
        if not csv_file.exists():
            print(f"{split}.csv not found!")
            continue
        
        df = pd.read_csv(csv_file)
        print(f"\n=== {split.upper()} SET ===")
        print(f"Total samples: {len(df)}")
        
        # Sample some FENs
        sample_fens = df['fen'].head(10).tolist()
        
        print("\nSample FENs (first 10):")
        for i, fen in enumerate(sample_fens):
            # Get board placement part
            board_fen = fen.split()[0] if ' ' in fen else fen
            
            # Tokenize
            tokens = tokenizer.encode(fen)
            
            print(f"{i+1}. FEN: {fen}")
            print(f"   Board part: {board_fen}")
            print(f"   Token length (with SOS/EOS): {len(tokens)}")
            print(f"   Decoded: {tokenizer.decode(tokens)}")
            print()
        
        # Statistics on all FENs
        print("\nLength Statistics:")
        all_token_lengths = []
        all_board_lengths = []
        
        for fen in df['fen']:
            board_fen = fen.split()[0] if ' ' in fen else fen
            tokens = tokenizer.encode(fen)
            all_token_lengths.append(len(tokens))
            all_board_lengths.append(len(board_fen))
        
        print(f"Token lengths (with SOS/EOS):")
        print(f"  Min: {min(all_token_lengths)}")
        print(f"  Max: {max(all_token_lengths)}")
        print(f"  Mean: {sum(all_token_lengths) / len(all_token_lengths):.2f}")
        
        print(f"Board FEN character lengths:")
        print(f"  Min: {min(all_board_lengths)}")
        print(f"  Max: {max(all_board_lengths)}")
        print(f"  Mean: {sum(all_board_lengths) / len(all_board_lengths):.2f}")
        
        # Check if FENs are complete (should have 8 ranks with / separators)
        incomplete_count = 0
        for fen in df['fen']:
            board_fen = fen.split()[0] if ' ' in fen else fen
            if board_fen.count('/') != 7:  # Should have 7 slashes for 8 ranks
                incomplete_count += 1
        
        print(f"\nIncomplete FENs (missing ranks): {incomplete_count} / {len(df)}")

if __name__ == "__main__":
    analyze_data()
