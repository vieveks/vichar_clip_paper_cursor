"""
Evaluation script for grid predictor model.
"""

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np

from .model import SpatialGridPredictor
from .dataset import GridPredictionDataset, create_transforms
from ..data_processing.representations import fen_to_grid


def evaluate_model(model, dataloader, device, output_path=None):
    """
    Evaluate grid predictor model.
    
    Returns:
        Dictionary with metrics:
        - per_square_accuracy: Accuracy per square
        - exact_board_match: Percentage of boards with all squares correct
        - per_piece_type_accuracy: Accuracy broken down by piece type
    """
    model.eval()
    
    correct_squares = 0
    total_squares = 0
    exact_matches = 0
    total_boards = 0
    
    # Per-piece-type accuracy
    piece_type_correct = {i: 0 for i in range(13)}
    piece_type_total = {i: 0 for i in range(13)}
    
    # Piece type names for reporting
    piece_type_names = {
        0: 'empty',
        1: 'white_pawn', 2: 'white_knight', 3: 'white_bishop',
        4: 'white_rook', 5: 'white_queen', 6: 'white_king',
        7: 'black_pawn', 8: 'black_knight', 9: 'black_bishop',
        10: 'black_rook', 11: 'black_queen', 12: 'black_king'
    }
    
    all_results = []
    
    with torch.no_grad():
        for images, grid_labels, fens in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            grid_labels = grid_labels.to(device)  # [B, 64]
            
            # Predict
            grid_logits = model(images)  # [B, 64, 13]
            predictions = grid_logits.argmax(dim=-1)  # [B, 64]
            
            # Compute per-square accuracy
            correct = (predictions == grid_labels)
            correct_squares += correct.sum().item()
            total_squares += grid_labels.numel()
            
            # Compute exact board matches
            batch_size = grid_labels.size(0)
            for i in range(batch_size):
                exact_match = correct[i].all().item()
                if exact_match:
                    exact_matches += 1
                total_boards += 1
                
                # Store result
                all_results.append({
                    'fen': fens[i],
                    'exact_match': exact_match,
                    'per_square_accuracy': correct[i].float().mean().item()
                })
            
            # Per-piece-type accuracy
            for piece_type in range(13):
                mask = (grid_labels == piece_type)
                if mask.any():
                    piece_type_total[piece_type] += mask.sum().item()
                    piece_type_correct[piece_type] += (predictions[mask] == piece_type).sum().item()
    
    # Compute metrics
    per_square_accuracy = correct_squares / total_squares if total_squares > 0 else 0.0
    exact_board_match = exact_matches / total_boards if total_boards > 0 else 0.0
    
    per_piece_type_accuracy = {}
    for piece_type in range(13):
        if piece_type_total[piece_type] > 0:
            acc = piece_type_correct[piece_type] / piece_type_total[piece_type]
            per_piece_type_accuracy[piece_type_names[piece_type]] = acc
    
    results = {
        'per_square_accuracy': per_square_accuracy,
        'exact_board_match': exact_board_match,
        'per_piece_type_accuracy': per_piece_type_accuracy,
        'total_boards': total_boards,
        'total_squares': total_squares,
        'correct_squares': correct_squares,
        'exact_matches': exact_matches
    }
    
    # Save detailed results if requested
    if output_path:
        with open(output_path, 'w') as f:
            json.dump({
                'summary': results,
                'detailed_results': all_results[:1000]  # Save first 1000 for space
            }, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Grid Predictor")
    parser.add_argument("--test_json", type=str, required=True,
                       help="Path to enriched test JSON file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save evaluation results JSON")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to evaluate (for testing)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = SpatialGridPredictor().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create dataset
    transform = create_transforms()
    test_dataset = GridPredictionDataset(args.test_json, transform=transform)
    
    if args.max_samples:
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, range(min(args.max_samples, len(test_dataset))))
        print(f"Limited to {args.max_samples} samples")
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    print(f"Evaluating on {len(test_dataset)} samples...")
    
    # Evaluate
    results = evaluate_model(model, test_loader, device, args.output)
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Per-Square Accuracy: {results['per_square_accuracy']:.4f} ({results['per_square_accuracy']*100:.2f}%)")
    print(f"Exact Board Match: {results['exact_board_match']:.4f} ({results['exact_board_match']*100:.2f}%)")
    print(f"\nPer-Piece-Type Accuracy:")
    for piece_type, acc in results['per_piece_type_accuracy'].items():
        print(f"  {piece_type}: {acc:.4f} ({acc*100:.2f}%)")
    print("="*50)
    
    if args.output:
        print(f"\nDetailed results saved to {args.output}")


if __name__ == '__main__':
    main()

