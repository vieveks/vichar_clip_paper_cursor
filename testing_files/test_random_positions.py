#!/usr/bin/env python3
"""
Generate random chess positions and test both models for comparative analysis.
"""

import torch
import clip
from PIL import Image
import chess
import random
from pathlib import Path
import logging
from dataset_prep_simple import create_chess_board_image
import pandas as pd
from tqdm import tqdm
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_random_chess_position():
    """Generate a random but legal chess position."""
    board = chess.Board()
    
    # Make 5-15 random legal moves to get interesting positions
    num_moves = random.randint(5, 15)
    
    for _ in range(num_moves):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        board.push(move)
    
    return board.fen()

def create_test_images(num_images: int = 20, output_dir: str = "random_test_images"):
    """Create random chess board images for testing."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    positions = []
    
    print(f"🎲 Generating {num_images} random chess positions...")
    for i in tqdm(range(num_images), desc="Creating test images"):
        # Generate random position
        fen = generate_random_chess_position()
        positions.append(fen)
        
        # Create image
        png_bytes = create_chess_board_image(fen, size=350)
        
        # Save image
        img_path = output_path / f"random_{i:02d}.png"
        with open(img_path, "wb") as f:
            f.write(png_bytes)
        
        # Save corresponding FEN
        fen_path = output_path / f"random_{i:02d}.txt"
        with open(fen_path, "w") as f:
            f.write(fen)
    
    print(f"✅ Created {num_images} test images in {output_dir}/")
    return positions, output_path

def test_model_on_images(model_path: str, test_images_dir: Path, positions: list, model_name: str):
    """Test a single model on the random images."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    results = []
    
    print(f"🧠 Testing {model_name} model...")
    
    for i, true_fen in enumerate(tqdm(positions, desc=f"Testing {model_name}")):
        img_path = test_images_dir / f"random_{i:02d}.png"
        
        # Load and preprocess image
        image = Image.open(img_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Create candidate FENs (true FEN + some random distractors)
        candidates = [true_fen]
        # Add 19 random distractors
        for _ in range(19):
            distractor_fen = generate_random_chess_position()
            candidates.append(distractor_fen)
        
        # Shuffle candidates so true FEN isn't always first
        random.shuffle(candidates)
        true_idx = candidates.index(true_fen)
        
        # Tokenize candidates
        text_inputs = clip.tokenize(candidates, truncate=True).to(device)
        
        with torch.no_grad():
            # Encode image and texts
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity scores
            scores = (image_features @ text_features.T)[0]
            
            # Get rankings
            sorted_indices = torch.argsort(scores, descending=True)
            
            # Find where the true FEN ranked
            true_rank = (sorted_indices == true_idx).nonzero().item() + 1
            top5_hit = true_rank <= 5
            top1_hit = true_rank == 1
        
        results.append({
            'image_id': i,
            'true_rank': true_rank,
            'top1_hit': top1_hit,
            'top5_hit': top5_hit,
            'confidence': scores[true_idx].item(),
            'max_confidence': scores.max().item()
        })
    
    return results

def analyze_results(results_a: list, results_b: list, model_a_name: str, model_b_name: str):
    """Analyze and compare results from both models."""
    
    # Calculate metrics for both models
    def calc_metrics(results):
        total = len(results)
        top1_acc = sum(r['top1_hit'] for r in results) / total * 100
        top5_acc = sum(r['top5_hit'] for r in results) / total * 100
        avg_rank = sum(r['true_rank'] for r in results) / total
        avg_confidence = sum(r['confidence'] for r in results) / total
        return {
            'Top-1 Accuracy (%)': top1_acc,
            'Top-5 Accuracy (%)': top5_acc,
            'Average Rank': avg_rank,
            'Average Confidence': avg_confidence
        }
    
    metrics_a = calc_metrics(results_a)
    metrics_b = calc_metrics(results_b)
    
    # Create comparison table
    df = pd.DataFrame([metrics_a, metrics_b], index=[model_a_name, model_b_name])
    
    print("\n" + "="*80)
    print("🏆 RANDOM CHESS POSITIONS COMPARATIVE ANALYSIS")
    print("="*80)
    print(f"Test Set: 20 random chess positions vs 19 distractors each")
    print("-"*80)
    print(df.to_string(float_format="%.2f"))
    
    # Detailed comparison
    print(f"\n🎯 Performance Comparison:")
    for metric in metrics_a.keys():
        diff = metrics_a[metric] - metrics_b[metric]
        if diff > 0:
            winner = model_a_name
            print(f"  {metric}: {winner} wins by {abs(diff):.2f}")
        elif diff < 0:
            winner = model_b_name
            print(f"  {metric}: {winner} wins by {abs(diff):.2f}")
        else:
            print(f"  {metric}: Tie")
    
    # Individual image analysis
    print(f"\n📊 Per-Image Analysis:")
    print(f"{'Image':<8} {'Model A Rank':<12} {'Model B Rank':<12} {'Winner':<15}")
    print("-"*50)
    
    a_wins = 0
    b_wins = 0
    ties = 0
    
    for i in range(len(results_a)):
        rank_a = results_a[i]['true_rank']
        rank_b = results_b[i]['true_rank']
        
        if rank_a < rank_b:
            winner = model_a_name
            a_wins += 1
        elif rank_b < rank_a:
            winner = model_b_name
            b_wins += 1
        else:
            winner = "Tie"
            ties += 1
        
        print(f"Img {i:02d}   {rank_a:<12} {rank_b:<12} {winner:<15}")
    
    print("-"*50)
    print(f"Head-to-Head: {model_a_name}: {a_wins}, {model_b_name}: {b_wins}, Ties: {ties}")
    
    return df, metrics_a, metrics_b

def main():
    parser = argparse.ArgumentParser(description="Test both models on random chess positions")
    parser.add_argument("--model_a", default="checkpoints/large_1000/fen_only_model/clip_chess_epoch_5.pt", 
                       help="Path to first model")
    parser.add_argument("--model_b", default="checkpoints/large_1000/fen_move_model/clip_chess_epoch_5.pt", 
                       help="Path to second model")
    parser.add_argument("--model_a_name", default="FEN Only", help="Name for first model")
    parser.add_argument("--model_b_name", default="FEN + Move", help="Name for second model")
    parser.add_argument("--num_images", type=int, default=20, help="Number of random test images")
    parser.add_argument("--output_dir", default="random_test_images", help="Directory for test images")
    
    args = parser.parse_args()
    
    # Create random test images
    positions, test_dir = create_test_images(args.num_images, args.output_dir)
    
    # Test both models
    results_a = test_model_on_images(args.model_a, test_dir, positions, args.model_a_name)
    results_b = test_model_on_images(args.model_b, test_dir, positions, args.model_b_name)
    
    # Analyze and compare
    df, metrics_a, metrics_b = analyze_results(results_a, results_b, args.model_a_name, args.model_b_name)
    
    # Save results
    results_file = f"random_test_results.csv"
    df.to_csv(results_file)
    print(f"\n💾 Results saved to {results_file}")
    
    print("\n🎉 Random position testing complete!")

if __name__ == "__main__":
    main()
