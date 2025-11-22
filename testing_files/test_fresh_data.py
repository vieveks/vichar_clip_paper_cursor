#!/usr/bin/env python3
"""
Test models on completely fresh chess data not used in training.
Downloads new PGN data and creates independent test set.
"""

import torch
import clip
from PIL import Image
import chess
import chess.pgn
import random
from pathlib import Path
import logging
from dataset_prep_simple import create_chess_board_image
import pandas as pd
from tqdm import tqdm
import argparse
import requests
import tempfile
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_fresh_pgn_data(year: int = 2024, month: int = 1):
    """Download fresh PGN data from Lichess that wasn't used in training."""
    url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"
    
    print(f"🌐 Attempting to download fresh data from {year}-{month:02d}...")
    print(f"URL: {url}")
    
    try:
        response = requests.head(url, timeout=10)
        if response.status_code == 200:
            print(f"✅ Fresh data available for {year}-{month:02d}")
            return url
        else:
            print(f"❌ No data available for {year}-{month:02d} (status: {response.status_code})")
            return None
    except Exception as e:
        print(f"❌ Error checking data availability: {e}")
        return None

def generate_positions_from_famous_games():
    """Generate positions from famous historical chess games (not in training data)."""
    
    # Famous games PGN data (these shouldn't be in the 2013 Lichess database)
    famous_games_pgn = """
[Event "World Championship"]
[Site "New York"]
[Date "1972.07.11"]
[Round "6"]
[White "Fischer, Robert James"]
[Black "Spassky, Boris V"]
[Result "1-0"]

1. c4 e6 2. Nf3 d5 3. d4 Nf6 4. Nc3 Be7 5. Bg5 O-O 6. e3 h6 7. Bh4 b6 8. cxd5 Nxd5 9. Bxe7 Qxe7 10. Nxd5 exd5 11. Rc1 Be6 12. Qa4 c5 13. Qa3 Rc8 14. Bb5 a6 15. dxc5 bxc5 16. O-O Ra7 17. Be2 Nd7 18. Nd4 Qf8 19. Nxe6 fxe6 20. e4 d4 21. f4 Qf6 22. e5 Qf5 23. Bc4 Kh8 24. Qh3 Nf8 25. b3 a5 26. f5 exf5 27. Rxf5 Nh7 28. Rcf1 Qd7 29. Qg3 Re8 30. h4 Re7 31. e6 Rxe6 32. Qxg7+ Qxg7 33. Rf8# 1-0

[Event "Immortal Game"]
[Site "London"]
[Date "1851.06.21"]
[Round "?"]
[White "Anderssen, Adolf"]
[Black "Kieseritzky, Lionel"]
[Result "1-0"]

1. e4 e5 2. f4 exf4 3. Bc4 Qh4+ 4. Kf1 b5 5. Bxb5 Nf6 6. Nf3 Qh6 7. d3 Nh5 8. Nh4 Qg5 9. Nf5 c6 10. g4 Nf6 11. Rg1 cxb5 12. h4 Qg6 13. h5 Qg5 14. Qf3 Ng8 15. Bxf4 Qf6 16. Nc3 Bc5 17. Nd5 Qxb2 18. Bd6 Bxg1 19. e5 Qxa1+ 20. Ke2 Na6 21. Nxg7+ Kd8 22. Qf6+ Nxf6 23. Be7# 1-0

[Event "Evergreen Game"]
[Site "Berlin"]
[Date "1852.??.??"]
[Round "?"]
[White "Anderssen, Adolf"]
[Black "Dufresne, Jean"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. b4 Bxb4 5. c3 Ba5 6. d4 exd4 7. O-O d3 8. Qb3 Qf6 9. e5 Qg6 10. Re1 Nge7 11. Ba3 b5 12. Qxb5 Rb8 13. Qa4 Bb6 14. Nbd2 Bb7 15. Ne4 Qf5 16. Bxd3 Qh5 17. Nf6+ gxf6 18. exf6 Rg8 19. Rad1 Qxf3 20. Rxe7+ Nxe7 21. Qxd7+ Kxd7 22. Bf5+ Ke8 23. Bd7+ Kf8 24. Bxe7# 1-0
"""
    
    positions = []
    games_processed = 0
    
    print("🏛️ Extracting positions from famous historical games...")
    
    # Parse the famous games
    pgn_io = io.StringIO(famous_games_pgn)
    
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
            
        board = game.board()
        move_count = 0
        
        for move in game.mainline_moves():
            board.push(move)
            move_count += 1
            
            # Sample positions at different points in the game
            if move_count in [5, 10, 15, 20, 25, 30]:
                positions.append({
                    'fen': board.fen(),
                    'source': f"Famous Game {games_processed + 1}, Move {move_count}",
                    'game_info': f"{game.headers.get('White', 'Unknown')} vs {game.headers.get('Black', 'Unknown')}"
                })
        
        games_processed += 1
    
    print(f"✅ Extracted {len(positions)} positions from {games_processed} famous games")
    return positions

def generate_puzzle_positions():
    """Generate tactical puzzle-like positions."""
    print("🧩 Generating tactical puzzle positions...")
    
    positions = []
    
    # Create some tactical positions manually
    tactical_fens = [
        # Fork positions
        "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4",
        # Pin positions  
        "rnbqkbnr/ppp2ppp/3p4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
        # Discovered attack
        "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 3",
        # Back rank mate threats
        "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
        # Queen vs Rook endgame
        "8/8/8/8/8/2k5/8/1Q4K1 w - - 0 1",
        # King and pawn endgame
        "8/8/8/3k4/3P4/3K4/8/8 w - - 0 1",
        # Rook endgame
        "8/8/8/8/8/2k5/2r5/2K5 w - - 0 1",
        # Bishop vs Knight
        "8/8/8/3k1n2/8/3K1B2/8/8 w - - 0 1",
        # Complex middlegame
        "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 8",
        # Sicilian Dragon
        "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"
    ]
    
    for i, fen in enumerate(tactical_fens):
        positions.append({
            'fen': fen,
            'source': f"Tactical Position {i+1}",
            'game_info': "Generated tactical position"
        })
    
    print(f"✅ Generated {len(positions)} tactical positions")
    return positions

def create_fresh_test_set(num_positions: int = 50):
    """Create a test set from completely fresh data."""
    print(f"🆕 Creating fresh test set with {num_positions} positions...")
    
    positions = []
    
    # Get positions from famous games
    famous_positions = generate_positions_from_famous_games()
    positions.extend(famous_positions)
    
    # Get tactical positions
    tactical_positions = generate_puzzle_positions()
    positions.extend(tactical_positions)
    
    # Generate additional random positions if needed
    while len(positions) < num_positions:
        # Create random position
        board = chess.Board()
        num_moves = random.randint(8, 25)  # More mature positions
        
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            board.push(move)
        
        positions.append({
            'fen': board.fen(),
            'source': f"Random Position {len(positions) + 1}",
            'game_info': f"Random game, {num_moves} moves"
        })
    
    # Limit to requested number
    positions = positions[:num_positions]
    
    print(f"✅ Created {len(positions)} fresh test positions")
    return positions

def create_test_images_from_positions(positions: list, output_dir: str = "fresh_test_images"):
    """Create images from the fresh positions."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"🖼️ Creating test images for {len(positions)} positions...")
    
    for i, pos_data in enumerate(tqdm(positions, desc="Creating fresh test images")):
        fen = pos_data['fen']
        
        # Create image
        png_bytes = create_chess_board_image(fen, size=350)
        
        # Save image
        img_path = output_path / f"fresh_{i:03d}.png"
        with open(img_path, "wb") as f:
            f.write(png_bytes)
        
        # Save metadata
        meta_path = output_path / f"fresh_{i:03d}.txt"
        with open(meta_path, "w") as f:
            f.write(f"FEN: {fen}\n")
            f.write(f"Source: {pos_data['source']}\n")
            f.write(f"Game: {pos_data['game_info']}\n")
    
    print(f"✅ Created {len(positions)} fresh test images in {output_dir}/")
    return output_path

def test_models_on_fresh_data(positions: list, test_dir: Path, model_paths: dict, num_distractors: int = 49):
    """Test both models on completely fresh data."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = {}
    
    for model_name, model_path in model_paths.items():
        print(f"\n🧠 Testing {model_name} on fresh data...")
        
        # Load model
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        model_results = []
        
        for i, pos_data in enumerate(tqdm(positions, desc=f"Testing {model_name}")):
            true_fen = pos_data['fen']
            img_path = test_dir / f"fresh_{i:03d}.png"
            
            # Load image
            image = Image.open(img_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            # Create candidates: true FEN + random distractors
            candidates = [true_fen]
            
            # Generate fresh random distractors (not from training data)
            for _ in range(num_distractors):
                distractor_board = chess.Board()
                distractor_moves = random.randint(5, 30)
                
                for _ in range(distractor_moves):
                    legal_moves = list(distractor_board.legal_moves)
                    if not legal_moves:
                        break
                    move = random.choice(legal_moves)
                    distractor_board.push(move)
                
                candidates.append(distractor_board.fen())
            
            # Shuffle candidates
            random.shuffle(candidates)
            true_idx = candidates.index(true_fen)
            
            # Test model
            text_inputs = clip.tokenize(candidates, truncate=True).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)
                
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                scores = (image_features @ text_features.T)[0]
                sorted_indices = torch.argsort(scores, descending=True)
                
                true_rank = (sorted_indices == true_idx).nonzero().item() + 1
                
                model_results.append({
                    'position_id': i,
                    'source': pos_data['source'],
                    'true_rank': true_rank,
                    'top1_hit': true_rank == 1,
                    'top5_hit': true_rank <= 5,
                    'top10_hit': true_rank <= 10,
                    'confidence': scores[true_idx].item(),
                    'max_confidence': scores.max().item()
                })
        
        results[model_name] = model_results
    
    return results

def analyze_fresh_results(results: dict, num_candidates: int):
    """Analyze results from fresh data testing."""
    
    print(f"\n" + "="*80)
    print("🆕 FRESH DATA COMPARATIVE ANALYSIS")
    print("="*80)
    print(f"Test Set: {len(list(results.values())[0])} fresh positions vs {num_candidates-1} distractors each")
    print(f"Data Source: Famous games + tactical positions + random positions")
    print("-"*80)
    
    # Calculate metrics
    metrics_summary = {}
    for model_name, model_results in results.items():
        total = len(model_results)
        metrics = {
            'Top-1 Accuracy (%)': sum(r['top1_hit'] for r in model_results) / total * 100,
            'Top-5 Accuracy (%)': sum(r['top5_hit'] for r in model_results) / total * 100,
            'Top-10 Accuracy (%)': sum(r['top10_hit'] for r in model_results) / total * 100,
            'Average Rank': sum(r['true_rank'] for r in model_results) / total,
            'Average Confidence': sum(r['confidence'] for r in model_results) / total,
            'Median Rank': sorted([r['true_rank'] for r in model_results])[total//2]
        }
        metrics_summary[model_name] = metrics
    
    # Create comparison table
    df = pd.DataFrame(metrics_summary).T
    print(df.to_string(float_format="%.2f"))
    
    # Performance by source type
    print(f"\n📊 Performance by Position Type:")
    model_names = list(results.keys())
    
    # Group by source type
    source_types = {}
    for model_results in results.values():
        for result in model_results:
            source = result['source'].split()[0]  # First word of source
            if source not in source_types:
                source_types[source] = []
    
    for source_type in source_types.keys():
        print(f"\n{source_type} Positions:")
        for model_name in model_names:
            model_results = results[model_name]
            source_results = [r for r in model_results if r['source'].startswith(source_type)]
            if source_results:
                top1_acc = sum(r['top1_hit'] for r in source_results) / len(source_results) * 100
                avg_rank = sum(r['true_rank'] for r in source_results) / len(source_results)
                print(f"  {model_name}: {top1_acc:.1f}% top-1, avg rank {avg_rank:.1f} ({len(source_results)} positions)")
    
    return df, metrics_summary

def main():
    parser = argparse.ArgumentParser(description="Test models on completely fresh chess data")
    parser.add_argument("--model_a", default="checkpoints/large_1000/fen_only_model/clip_chess_epoch_5.pt")
    parser.add_argument("--model_b", default="checkpoints/large_1000/fen_move_model/clip_chess_epoch_5.pt")
    parser.add_argument("--num_positions", type=int, default=50, help="Number of test positions")
    parser.add_argument("--num_distractors", type=int, default=49, help="Number of distractor positions")
    
    args = parser.parse_args()
    
    # Create fresh test set
    positions = create_fresh_test_set(args.num_positions)
    test_dir = create_test_images_from_positions(positions)
    
    # Test models
    model_paths = {
        "FEN Only": args.model_a,
        "FEN + Move": args.model_b
    }
    
    results = test_models_on_fresh_data(positions, test_dir, model_paths, args.num_distractors)
    
    # Analyze results
    df, metrics = analyze_fresh_results(results, args.num_distractors + 1)
    
    # Save results
    df.to_csv("fresh_data_results.csv")
    print(f"\n💾 Results saved to fresh_data_results.csv")
    
    print("\n🎉 Fresh data testing complete!")

if __name__ == "__main__":
    main()
