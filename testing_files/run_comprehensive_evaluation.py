#!/usr/bin/env python3
"""
Helper script to run comprehensive evaluation on both models.
This will test for overfitting and evaluate on independent datasets.
"""

import subprocess
import sys
from pathlib import Path

def run_evaluation(model_path, model_name, data_dir, independent_pgn=None):
    """Run comprehensive evaluation for a model."""
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name}")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable,
        "comprehensive_evaluation.py",
        "--model_path", model_path,
        "--data_dir", data_dir,
        "--batch_size", "64",
        "--train_ratio", "0.8",
        "--val_ratio", "0.1",
        "--test_ratio", "0.1",
        "--num_independent", "1000"
    ]
    
    if independent_pgn:
        cmd.extend(["--independent_pgn", independent_pgn])
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0

def main():
    # Model paths
    fen_only_model = "checkpoints/large_1000/fen_only_model/clip_chess_epoch_5.pt"
    fen_move_model = "checkpoints/large_1000/fen_move_model/clip_chess_epoch_5.pt"
    
    # Dataset paths
    fen_only_data = "large_datasets/fen_only"
    fen_move_data = "large_datasets/fen_move"
    
    # Check for independent PGN (different time period)
    # Try to find a PGN file from a different year/month
    independent_pgn = None
    possible_pgns = [
        "lichess_games_2014-01.pgn",
        "lichess_games_2013-02.pgn",
        "lichess_games_2012-01.pgn",
        "../lichess_games_2013-01.pgn"  # If in parent directory
    ]
    
    for pgn in possible_pgns:
        pgn_path = Path(__file__).parent / pgn
        if pgn_path.exists():
            independent_pgn = str(pgn_path)
            print(f"✅ Found independent PGN: {pgn_path}")
            break
    
    if not independent_pgn:
        print("⚠️  No independent PGN file found. Will only test on training data splits.")
        print("   To test on independent data, download a PGN from a different time period.")
    
    # Run evaluations
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION - OVERFITTING ANALYSIS")
    print("="*80)
    
    # Evaluate FEN-only model
    success1 = run_evaluation(
        fen_only_model,
        "FEN-Only Model",
        fen_only_data,
        independent_pgn
    )
    
    # Evaluate FEN+Move model
    success2 = run_evaluation(
        fen_move_model,
        "FEN+Move Model",
        fen_move_data,
        independent_pgn
    )
    
    if success1 and success2:
        print("\n✅ All evaluations completed successfully!")
        print("\n📊 Check the following files for results:")
        print("   - comprehensive_evaluation_results.json")
        print("   - train_val_test_comparison.csv")
    else:
        print("\n⚠️  Some evaluations may have failed. Check the output above.")

if __name__ == "__main__":
    main()

