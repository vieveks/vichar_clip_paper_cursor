"""
Evaluation script for Experiment A: Stockfish CP Loss Validation

Evaluates predicted FENs from Exp 1B and calculates CP loss vs ground truth.
Target: Mean CP loss < 150 (expected ~127 ± 89).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm

# Add paths for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
IMPROVED_REP_PATH = PROJECT_ROOT / "Improved_representations"
sys.path.insert(0, str(IMPROVED_REP_PATH))
sys.path.insert(0, str(IMPROVED_REP_PATH / "data_processing"))

# Import existing code (read-only) - use shared utils
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from utils import json_to_fen

# Import evaluator
sys.path.insert(0, str(Path(__file__).parent))
from stockfish_evaluator import StockfishEvaluator


def load_predictions(predictions_path: str) -> List[Dict]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                pred = json.loads(line)
                # Handle different JSONL formats
                # Format 1: {"predicted_json": {...}, "predicted_fen": "..."}
                # Format 2: {"pred_json": {...}, "true_fen": "..."}
                # Format 3: {"json_pred": {...}, "fen": "..."}
                if 'predicted_json' in pred:
                    pred['pred_json'] = pred['predicted_json']
                    pred['true_fen'] = pred.get('predicted_fen', '')
                elif 'json_pred' in pred:
                    pred['pred_json'] = pred['json_pred']
                    pred['true_fen'] = pred.get('fen', '')
                predictions.append(pred)
    return predictions


def evaluate_cp_loss(
    predictions: List[Dict],
    evaluator: StockfishEvaluator,
    max_samples: Optional[int] = None
) -> Dict:
    """
    Evaluate CP loss for all predictions.
    
    Args:
        predictions: List of prediction dicts with 'pred_json' and 'true_fen'
        evaluator: StockfishEvaluator instance
        max_samples: Maximum samples to evaluate (None = all)
    
    Returns:
        Dictionary with CP loss statistics
    """
    if max_samples:
        predictions = predictions[:max_samples]
    
    cp_losses = []
    failed_evaluations = 0
    
    print(f"Evaluating {len(predictions)} positions...")
    
    for pred in tqdm(predictions, desc="Calculating CP loss"):
        # Get predicted FEN from JSON
        try:
            pred_json = pred.get('pred_json') or pred.get('json_pred') or pred.get('predicted_json')
            if not pred_json:
                # Skip if no JSON prediction available
                failed_evaluations += 1
                continue
            
            pred_fen = json_to_fen(pred_json)
            true_fen = pred.get('true_fen') or pred.get('fen') or pred.get('predicted_fen')
            
            if not true_fen:
                # Try to get from json_repr if available
                if 'json_repr' in pred:
                    # This is ground truth JSON, skip (we need predicted JSON)
                    failed_evaluations += 1
                    continue
                failed_evaluations += 1
                continue
            
            # Calculate CP loss
            cp_loss = evaluator.calculate_cp_loss(pred_fen, true_fen)
            
            if cp_loss is not None:
                cp_losses.append(cp_loss)
            else:
                failed_evaluations += 1
        
        except Exception as e:
            failed_evaluations += 1
            continue
    
    if not cp_losses:
        return {
            'mean_cp_loss': None,
            'std_cp_loss': None,
            'median_cp_loss': None,
            'min_cp_loss': None,
            'max_cp_loss': None,
            'total_samples': len(predictions),
            'successful_evaluations': 0,
            'failed_evaluations': failed_evaluations
        }
    
    cp_losses = np.array(cp_losses)
    
    return {
        'mean_cp_loss': float(np.mean(cp_losses)),
        'std_cp_loss': float(np.std(cp_losses)),
        'median_cp_loss': float(np.median(cp_losses)),
        'min_cp_loss': float(np.min(cp_losses)),
        'max_cp_loss': float(np.max(cp_losses)),
        'percentiles': {
            '25th': float(np.percentile(cp_losses, 25)),
            '50th': float(np.percentile(cp_losses, 50)),
            '75th': float(np.percentile(cp_losses, 75)),
            '90th': float(np.percentile(cp_losses, 90)),
            '95th': float(np.percentile(cp_losses, 95)),
            '99th': float(np.percentile(cp_losses, 99))
        },
        'total_samples': len(predictions),
        'successful_evaluations': len(cp_losses),
        'failed_evaluations': failed_evaluations
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Stockfish CP Loss (Experiment A)')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions JSONL file (e.g., Improved_representations/results/predictions_clip_exp1b.jsonl)')
    parser.add_argument('--output', type=str, default='neurosymbolic_pipeline/results/exp_a/cp_loss_results.json',
                        help='Output path for results')
    parser.add_argument('--use_lichess_api', action='store_true', default=True,
                        help='Use Lichess Cloud Evaluation API (default: True, recommended)')
    parser.add_argument('--no_lichess_api', dest='use_lichess_api', action='store_false',
                        help='Disable Lichess API and use python-chess simple evaluation')
    parser.add_argument('--depth', type=int, default=15,
                        help='Search depth for Stockfish evaluation (Lichess API)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to evaluate (None = all)')
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from {args.predictions}...")
    predictions = load_predictions(args.predictions)
    print(f"Loaded {len(predictions)} predictions")
    
    # Initialize evaluator
    print(f"Initializing Stockfish evaluator (depth={args.depth}, use_lichess_api={args.use_lichess_api})...")
    evaluator = StockfishEvaluator(use_lichess_api=args.use_lichess_api, depth=args.depth)
    
    try:
        # Evaluate CP loss
        results = evaluate_cp_loss(predictions, evaluator, max_samples=args.max_samples)
        
        # Print results
        print("\n" + "="*60)
        print("CP LOSS RESULTS")
        print("="*60)
        print(f"\nMean CP Loss: {results['mean_cp_loss']:.2f} ± {results['std_cp_loss']:.2f}")
        print(f"Median CP Loss: {results['median_cp_loss']:.2f}")
        print(f"Min/Max CP Loss: {results['min_cp_loss']:.2f} / {results['max_cp_loss']:.2f}")
        print(f"\nPercentiles:")
        for percentile, value in results['percentiles'].items():
            print(f"  {percentile}: {value:.2f}")
        print(f"\nSuccessful evaluations: {results['successful_evaluations']}/{results['total_samples']}")
        
        # Check target
    if results['mean_cp_loss'] is not None:
        target = 150
        mean_loss = results['mean_cp_loss']
        std_loss = results['std_cp_loss'] if results['std_cp_loss'] is not None else 0.0
        print(f"\nMean CP Loss: {mean_loss:.2f} ± {std_loss:.2f}")
        print(f"Median CP Loss: {mean_loss:.2f}")
        print(f"Min/Max CP Loss: {results['min_cp_loss']:.2f} / {results['max_cp_loss']:.2f}")
        if mean_loss < target:
            print(f"\nSUCCESS: Mean CP loss ({mean_loss:.2f}) < target ({target})")
        else:
            print(f"\nWARNING: Mean CP loss ({mean_loss:.2f}) >= target ({target})")
    else:
        print("\nWARNING: No valid CP loss calculations (all evaluations failed)")
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    finally:
        evaluator.close()
    
    return results


if __name__ == '__main__':
    main()

