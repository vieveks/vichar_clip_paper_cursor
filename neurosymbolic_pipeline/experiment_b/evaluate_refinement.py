"""
Evaluation script for Experiment B: Symbolic Refinement

Compares model performance with and without symbolic refinement.
Target: Improve exact match from 0.008% to 8.3% (+103x improvement).
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add paths for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
IMPROVED_REP_PATH = PROJECT_ROOT / "Improved_representations"
sys.path.insert(0, str(IMPROVED_REP_PATH))
sys.path.insert(0, str(IMPROVED_REP_PATH / "json_predictor"))
sys.path.insert(0, str(IMPROVED_REP_PATH / "data_processing"))

# Import existing code (read-only)
try:
    from json_predictor.model import JSONPredictorModel
    from json_predictor.dataset import JSONDataset, PIECE_TO_IDX, IDX_TO_PIECE, grid_to_json
    from data_processing.converters import json_to_fen, validate_json_position
except ImportError as e:
    # Try alternative import paths
    import importlib.util
    model_path = IMPROVED_REP_PATH / "json_predictor" / "model.py"
    dataset_path = IMPROVED_REP_PATH / "json_predictor" / "dataset.py"
    converters_path = IMPROVED_REP_PATH / "data_processing" / "converters.py"
    
    if model_path.exists():
        spec = importlib.util.spec_from_file_location("json_predictor.model", model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        JSONPredictorModel = model_module.JSONPredictorModel
    
    if dataset_path.exists():
        spec = importlib.util.spec_from_file_location("json_predictor.dataset", dataset_path)
        dataset_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset_module)
        JSONDataset = dataset_module.JSONDataset
        PIECE_TO_IDX = dataset_module.PIECE_TO_IDX
        IDX_TO_PIECE = dataset_module.IDX_TO_PIECE
        grid_to_json = dataset_module.grid_to_json
    
    if converters_path.exists():
        spec = importlib.util.spec_from_file_location("converters", converters_path)
        converters_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converters_module)
        json_to_fen = converters_module.json_to_fen
        validate_json_position = converters_module.validate_json_position

# Import refinement module
sys.path.insert(0, str(Path(__file__).parent))
from refinement import refine_json_prediction


def evaluate_with_refinement(
    model: JSONPredictorModel,
    dataloader: DataLoader,
    device: str,
    use_refinement: bool = True,
    confidence_threshold: float = 0.5
) -> Dict:
    """
    Evaluate model with or without refinement.
    
    Args:
        model: Trained JSON predictor model
        dataloader: Test dataloader
        device: Device to run on
        use_refinement: Whether to apply symbolic refinement
        confidence_threshold: Confidence threshold for refinement
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    # Accumulators
    total_squares = 0
    correct_squares = 0
    total_boards = 0
    exact_matches = 0
    fen_exact_matches = 0
    valid_json_count = 0
    
    # Metadata accuracy
    to_move_correct = 0
    castling_correct = 0
    
    # Per-piece-type metrics
    piece_type_correct = defaultdict(int)
    piece_type_total = defaultdict(int)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {'with' if use_refinement else 'without'} refinement")):
            images = batch['image'].to(device)
            grid_targets = batch['grid_target'].to(device)
            to_move_targets = batch['to_move'].to(device)
            castling_targets = batch['castling'].to(device)
            fens = batch['fen']
            
            # Forward pass
            predictions = model.predict(images)
            grid_preds = predictions['grid_preds']  # [B, 64]
            grid_probs = predictions['grid_probs']  # [B, 64, 13]
            to_move_preds = predictions['to_move']  # [B]
            castling_preds = predictions['castling']  # [B, 4]
            
            batch_size = grid_preds.size(0)
            
            for i in range(batch_size):
                pred = grid_preds[i]  # [64]
                target = grid_targets[i]  # [64]
                prob = grid_probs[i]  # [64, 13]
                
                # Apply refinement if requested
                if use_refinement:
                    # Convert to JSON
                    pred_json = grid_to_json(
                        pred,
                        to_move=to_move_preds[i].item(),
                        castling=castling_preds[i]
                    )
                    
                    # Apply refinement
                    refined_json = refine_json_prediction(
                        pred_json,
                        grid_probs=prob,
                        confidence_threshold=confidence_threshold
                    )
                    
                    # Convert refined JSON back to grid for comparison
                    # (We'll compare FEN instead for exact match)
                    pred_fen = json_to_fen(refined_json)
                else:
                    # No refinement - use original prediction
                    pred_json = grid_to_json(
                        pred,
                        to_move=to_move_preds[i].item(),
                        castling=castling_preds[i]
                    )
                    pred_fen = json_to_fen(pred_json)
                
                # Compare with ground truth
                true_fen_board = fens[i].split()[0] if ' ' in fens[i] else fens[i]
                pred_fen_board = pred_fen.split()[0] if ' ' in pred_fen else pred_fen
                
                # Exact board match (FEN comparison)
                if true_fen_board == pred_fen_board:
                    fen_exact_matches += 1
                    exact_matches += 1
                
                # Per-square accuracy (using original grid predictions for consistency)
                correct = (pred == target)
                correct_squares += correct.sum().item()
                total_squares += 64
                
                # Exact board match (grid comparison)
                if correct.all().item():
                    exact_matches += 1
                
                # JSON validity
                try:
                    is_valid, errors = validate_json_position(pred_json if not use_refinement else refined_json)
                    if is_valid:
                        valid_json_count += 1
                except:
                    pass
                
                # Metadata accuracy
                if to_move_preds[i].item() == to_move_targets[i].item():
                    to_move_correct += 1
                
                castling_match = (castling_preds[i] == castling_targets[i]).all()
                if castling_match:
                    castling_correct += 1
                
                # Per-piece-type accuracy
                for sq in range(64):
                    true_piece = target[sq].item()
                    pred_piece = pred[sq].item()
                    
                    piece_type_total[true_piece] += 1
                    if true_piece == pred_piece:
                        piece_type_correct[true_piece] += 1
                
                total_boards += 1
    
    # Compute metrics
    metrics = {
        'per_square_accuracy': correct_squares / total_squares if total_squares > 0 else 0.0,
        'exact_board_match': exact_matches / total_boards if total_boards > 0 else 0.0,
        'fen_exact_match': fen_exact_matches / total_boards if total_boards > 0 else 0.0,
        'valid_json_rate': valid_json_count / total_boards if total_boards > 0 else 0.0,
        'to_move_accuracy': to_move_correct / total_boards if total_boards > 0 else 0.0,
        'castling_accuracy': castling_correct / total_boards if total_boards > 0 else 0.0,
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
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate symbolic refinement (Experiment B)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (e.g., Improved_representations/checkpoints/exp1b_finetuned_frozen/best_model.pt)')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test JSON dataset')
    parser.add_argument('--image_base_dir', type=str, default=None,
                        help='Base directory for images')
    parser.add_argument('--output', type=str, default='neurosymbolic_pipeline/results/exp_b/refinement_comparison.json',
                        help='Output path for results')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Confidence threshold for refinement')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to evaluate (for testing)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = JSONPredictorModel()
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.to(args.device)
    model.eval()
    
    # Create dataloader
    print(f"Loading test data from {args.test_data}...")
    # JSONL paths are relative to project root (e.g., "data/hf_chess_puzzles/test/images/...")
    # So image_base_dir should always be project root
    image_base_dir = PROJECT_ROOT.resolve()
    
    if args.image_base_dir:
        # If user provides a path, warn but use project root
        print(f"Warning: image_base_dir argument ignored. Using project root: {image_base_dir}")
    
    print(f"Using image base directory: {image_base_dir}")
    
    dataset = JSONDataset(
        args.test_data,
        image_base_dir=str(image_base_dir),
        transform=None,  # Will use default transforms
        max_samples=args.max_samples
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Evaluate without refinement
    print("\n" + "="*60)
    print("Evaluating WITHOUT refinement...")
    print("="*60)
    metrics_before = evaluate_with_refinement(
        model, dataloader, args.device,
        use_refinement=False
    )
    
    # Evaluate with refinement
    print("\n" + "="*60)
    print("Evaluating WITH refinement...")
    print("="*60)
    metrics_after = evaluate_with_refinement(
        model, dataloader, args.device,
        use_refinement=True,
        confidence_threshold=args.confidence_threshold
    )
    
    # Compute improvements
    improvements = {
        'exact_match_improvement': metrics_after['exact_board_match'] - metrics_before['exact_board_match'],
        'exact_match_improvement_x': metrics_after['exact_board_match'] / metrics_before['exact_board_match'] if metrics_before['exact_board_match'] > 0 else float('inf'),
        'fen_exact_match_improvement': metrics_after['fen_exact_match'] - metrics_before['fen_exact_match'],
        'fen_exact_match_improvement_x': metrics_after['fen_exact_match'] / metrics_before['fen_exact_match'] if metrics_before['fen_exact_match'] > 0 else float('inf'),
        'per_square_improvement': metrics_after['per_square_accuracy'] - metrics_before['per_square_accuracy'],
        'valid_json_improvement': metrics_after['valid_json_rate'] - metrics_before['valid_json_rate'],
    }
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\nExact Board Match:")
    print(f"  Before: {metrics_before['exact_board_match']*100:.4f}%")
    print(f"  After:  {metrics_after['exact_board_match']*100:.4f}%")
    print(f"  Improvement: {improvements['exact_match_improvement']*100:.4f}% ({improvements['exact_match_improvement_x']:.2f}x)")
    
    print(f"\nFEN Exact Match:")
    print(f"  Before: {metrics_before['fen_exact_match']*100:.4f}%")
    print(f"  After:  {metrics_after['fen_exact_match']*100:.4f}%")
    print(f"  Improvement: {improvements['fen_exact_match_improvement']*100:.4f}% ({improvements['fen_exact_match_improvement_x']:.2f}x)")
    
    print(f"\nPer-Square Accuracy:")
    print(f"  Before: {metrics_before['per_square_accuracy']*100:.2f}%")
    print(f"  After:  {metrics_after['per_square_accuracy']*100:.2f}%")
    print(f"  Improvement: {improvements['per_square_improvement']*100:.2f}%")
    
    print(f"\nValid JSON Rate:")
    print(f"  Before: {metrics_before['valid_json_rate']*100:.2f}%")
    print(f"  After:  {metrics_after['valid_json_rate']*100:.2f}%")
    print(f"  Improvement: {improvements['valid_json_improvement']*100:.2f}%")
    
    # Save results
    results = {
        'experiment': 'Experiment B: Symbolic Refinement',
        'checkpoint': args.checkpoint,
        'test_data': args.test_data,
        'confidence_threshold': args.confidence_threshold,
        'before_refinement': metrics_before,
        'after_refinement': metrics_after,
        'improvements': improvements
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == '__main__':
    main()

