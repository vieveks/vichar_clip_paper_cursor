#!/usr/bin/env python3
"""
Check if there's overlap between training data and independent test set.
"""

import chess.pgn
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

def extract_fens_from_pgn(pgn_path, max_games=None):
    """Extract all FEN positions from a PGN file."""
    fens = set()
    game_count = 0
    
    with open(pgn_path, 'r', encoding='utf-8') as f:
        while True:
            if max_games and game_count >= max_games:
                break
                
            game = chess.pgn.read_game(f)
            if game is None:
                break
            
            board = game.board()
            for move in game.mainline_moves():
                fen = board.fen()
                # Normalize FEN (remove move counters and en passant info for comparison)
                fen_parts = fen.split(' ')
                normalized_fen = ' '.join(fen_parts[:4])  # Just position, turn, castling, en passant
                fens.add(normalized_fen)
                board.push(move)
            
            game_count += 1
    
    return fens, game_count

def extract_fens_from_dataset(data_dir):
    """Extract FENs from the training dataset."""
    data_path = Path(data_dir)
    text_dir = data_path / "texts"
    
    fens = set()
    
    if not text_dir.exists():
        logging.warning(f"Text directory not found: {text_dir}")
        return fens
    
    text_files = list(text_dir.glob("*.txt"))
    logging.info(f"Reading {len(text_files)} FEN files from dataset...")
    
    for txt_file in tqdm(text_files, desc="Reading dataset FENs"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                fen = f.read().strip()
                # Normalize FEN
                fen_parts = fen.split(' ')
                normalized_fen = ' '.join(fen_parts[:4])
                fens.add(normalized_fen)
        except Exception as e:
            logging.warning(f"Error reading {txt_file}: {e}")
    
    return fens

def main():
    # Paths
    training_data_dir = "../Notebooks/large_datasets/fen_only"
    anand_pgn = "../data/pgn_files/anand_pgns/Anand.pgn"
    
    logging.info("Extracting FENs from training dataset...")
    training_fens = extract_fens_from_dataset(training_data_dir)
    logging.info(f"Training dataset: {len(training_fens)} unique positions")
    
    logging.info(f"Extracting FENs from {anand_pgn}...")
    anand_fens, game_count = extract_fens_from_pgn(anand_pgn)
    logging.info(f"Anand.pgn: {len(anand_fens)} unique positions from {game_count} games")
    
    # Find overlap
    overlap = training_fens & anand_fens
    overlap_percent = (len(overlap) / len(anand_fens) * 100) if anand_fens else 0
    
    print("\n" + "="*80)
    print("DATA INDEPENDENCE CHECK")
    print("="*80)
    print(f"Training dataset positions: {len(training_fens):,}")
    print(f"Anand.pgn positions: {len(anand_fens):,}")
    print(f"Overlapping positions: {len(overlap):,} ({overlap_percent:.2f}%)")
    print("="*80)
    
    if overlap_percent < 5:
        print("GOOD: Very little overlap - Anand.pgn is a good independent test set!")
    elif overlap_percent < 20:
        print("ACCEPTABLE: Some overlap, but still mostly independent.")
    else:
        print("WARNING: Significant overlap - may not be truly independent!")
    
    # Show some examples of overlap if any
    if overlap:
        print(f"\nSample overlapping positions (first 5):")
        for i, fen in enumerate(list(overlap)[:5]):
            print(f"  {i+1}. {fen}")

if __name__ == "__main__":
    main()

