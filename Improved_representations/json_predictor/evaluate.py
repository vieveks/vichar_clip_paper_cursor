"""
Evaluation script for JSON Predictor model.

Evaluates the model on test set and generates detailed metrics:
- Per-square accuracy
- Exact board match
- Per-piece-type accuracy
- FEN reconstruction accuracy
- JSON validity
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from json_predictor.model import JSONPredictorModel
from json_predictor.dataset import JSONDataset, PIECE_TO_IDX, IDX_TO_PIECE, grid_to_json
from data_processing.converters import json_to_fen, validate_json_position


def evaluate_model(model, dataloader, device):
    """
    Comprehensive evaluation of the model.
    
    Returns detailed metrics including:
    - Overall accuracy
    - Per-piece-type accuracy
    - FEN reconstruction accuracy
    """
    model.eval()
    
    # Accumulators
    total_squares = 0
    correct_squares = 0
    total_boards = 0
    exact_matches = 0
    
    # Per-piece-type metrics
    piece_type_correct = defaultdict(int)
    piece_type_total = defaultdict(int)
    
    # Confusion matrix for piece types
    confusion = torch.zeros(13, 13, dtype=torch.long)
    
    # FEN reconstruction
    fen_exact_matches = 0
    valid_json_count = 0
    
    # Metadata accuracy
    to_move_correct = 0
    castling_correct = 0
    
    # Sample predictions for analysis
    sample_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = batch['image'].to(device)
            grid_targets = batch['grid_target'].to(device)
            to_move_targets = batch['to_move'].to(device)
            castling_targets = batch['castling'].to(device)
            fens = batch['fen']
            
            # Forward pass
            predictions = model.predict(images)
            grid_preds = predictions['grid_preds']  # [B, 64]
            to_move_preds = predictions['to_move']  # [B]
            castling_preds = predictions['castling']  # [B, 4]
            
            B = images.shape[0]
            
            for i in range(B):
                # Per-square accuracy
                pred = grid_preds[i]
                target = grid_targets[i]
                
                correct = (pred == target)
                correct_squares += correct.sum().item()
                total_squares += 64
                
                # Exact board match
                if correct.all():
                    exact_matches += 1
                total_boards += 1
                
                # Per-piece-type accuracy
                for sq in range(64):
                    true_piece = target[sq].item()
                    pred_piece = pred[sq].item()
                    
                    piece_type_total[true_piece] += 1
                    if true_piece == pred_piece:
                        piece_type_correct[true_piece] += 1
                    
                    confusion[true_piece, pred_piece] += 1
                
                # Metadata accuracy
                if to_move_preds[i].item() == to_move_targets[i].item():
                    to_move_correct += 1
                
                castling_match = (castling_preds[i] == castling_targets[i]).all()
                if castling_match:
                    castling_correct += 1
                
                # JSON validity and FEN reconstruction
                try:
                    pred_json = grid_to_json(
                        pred,
                        to_move=to_move_preds[i].item(),
                        castling=castling_preds[i]
                    )
                    
                    is_valid, errors = validate_json_position(pred_json)
                    if is_valid:
                        valid_json_count += 1
                    
                    # Reconstruct FEN
                    pred_fen = json_to_fen(pred_json)
                    true_fen_board = fens[i].split()[0]
                    pred_fen_board = pred_fen.split()[0]
                    
                    if true_fen_board == pred_fen_board:
                        fen_exact_matches += 1
                    
                except Exception as e:
                    pass
                
                # Save sample predictions
                if batch_idx == 0 and i < 5:
                    sample_predictions.append({
                        'true_fen': fens[i],
                        'pred_grid': pred.cpu().tolist(),
                        'target_grid': target.cpu().tolist(),
                        'correct_squares': correct.sum().item()
                    })
    
    # Compute metrics
    metrics = {
        'per_square_accuracy': correct_squares / total_squares,
        'exact_board_match': exact_matches / total_boards,
        'fen_reconstruction_accuracy': fen_exact_matches / total_boards,
        'valid_json_rate': valid_json_count / total_boards,
        'to_move_accuracy': to_move_correct / total_boards,
        'castling_accuracy': castling_correct / total_boards,
        'total_samples': total_boards
    }
    
    # Per-piece-type accuracy
    piece_type_acc = {}
    for piece_idx in range(13):
        total = piece_type_total[piece_idx]
        if total > 0:
            acc = piece_type_correct[piece_idx] / total
            piece_name = IDX_TO_PIECE[piece_idx]
            piece_type_acc[piece_name] = {
                'accuracy': acc,
                'total': total,
                'correct': piece_type_correct[piece_idx]
            }
    
    metrics['per_piece_type'] = piece_type_acc
    
    # Confusion matrix as list
    metrics['confusion_matrix'] = confusion.tolist()
    metrics['piece_labels'] = [IDX_TO_PIECE[i] for i in range(13)]
    
    # Sample predictions
    metrics['sample_predictions'] = sample_predictions
    
    return metrics


def print_results(metrics):
    """Pretty print evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nOverall Metrics:")
    print(f"  Per-square accuracy:       {metrics['per_square_accuracy']*100:.2f}%")
    print(f"  Exact board match:         {metrics['exact_board_match']*100:.2f}%")
    print(f"  FEN reconstruction:        {metrics['fen_reconstruction_accuracy']*100:.2f}%")
    print(f"  Valid JSON rate:           {metrics['valid_json_rate']*100:.2f}%")
    print(f"  To-move accuracy:          {metrics['to_move_accuracy']*100:.2f}%")
    print(f"  Castling accuracy:         {metrics['castling_accuracy']*100:.2f}%")
    print(f"  Total samples:             {metrics['total_samples']}")
    
    print(f"\nPer-Piece-Type Accuracy:")
    for piece_name, stats in sorted(metrics['per_piece_type'].items()):
        print(f"  {piece_name:15s}: {stats['accuracy']*100:6.2f}% ({stats['correct']}/{stats['total']})")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate JSON Predictor')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Data arguments
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test.json')
    parser.add_argument('--image_base_dir', type=str, default=None,
                       help='Base directory for image paths')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='results/json_results.json',
                       help='Path to save results JSON')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get model args from checkpoint
    model_args = checkpoint.get('args', {})
    
    # Create model
    model = JSONPredictorModel(
        hidden_dim=model_args.get('hidden_dim', 512),
        freeze_encoder=True  # Always freeze for evaluation
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Determine image base directory
    if args.image_base_dir is None:
        args.image_base_dir = str(Path(args.test_data).parent.parent.parent)
    
    # Create test dataset
    print(f"Loading test data from {args.test_data}")
    test_dataset = JSONDataset(
        args.test_data,
        image_base_dir=args.image_base_dir
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Evaluate
    print(f"\nEvaluating on {len(test_dataset)} samples...")
    metrics = evaluate_model(model, test_loader, device)
    
    # Print results
    print_results(metrics)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()

