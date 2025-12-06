"""Extract metrics from checkpoint without importing torch."""
import pickle
import json
from pathlib import Path

def extract_checkpoint_metrics(checkpoint_path):
    """Extract metrics from PyTorch checkpoint using pickle."""
    with open(checkpoint_path, 'rb') as f:
        # Skip the first few bytes (PyTorch version info)
        # PyTorch saves: protocol version, then the actual data
        try:
            # Try to load directly
            data = pickle.load(f)
            if isinstance(data, dict) and 'val_metrics' in data:
                return data['val_metrics'], data.get('epoch', 'unknown')
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None, None
    return None, None

# Extract Exp 1D metrics
checkpoint_path = Path('checkpoints/exp1d_base_unfrozen/best_model.pt')
if checkpoint_path.exists():
    val_metrics, epoch = extract_checkpoint_metrics(checkpoint_path)
    if val_metrics:
        results_1d = {
            'experiment': 'Exp 1D: Base CLIP, Unfrozen Encoder',
            'best_epoch': epoch,
            'per_square_accuracy': float(val_metrics['per_square_acc']),
            'exact_board_match': float(val_metrics['exact_board_match']),
            'to_move_accuracy': float(val_metrics['to_move_acc']),
            'castling_accuracy': float(val_metrics['castling_acc']),
            'loss': float(val_metrics['loss'])
        }
        
        # Load previous results
        with open('results/exp1a_results.json') as f:
            results_1a = json.load(f)
        with open('results/exp1b_results.json') as f:
            results_1b = json.load(f)
        
        print('='*70)
        print('ABLATION STUDY RESULTS - ALL EXPERIMENTS')
        print('='*70)
        print()
        print('Exp 1A: Base CLIP, Frozen Encoder')
        print(f'  Per-square accuracy: {results_1a["per_square_accuracy"]:.4f} ({results_1a["per_square_accuracy"]*100:.2f}%)')
        print(f'  Exact board match: {results_1a["exact_board_match"]:.4f}')
        print(f'  Best epoch: {results_1a["best_epoch"]}')
        print()
        print('Exp 1B: Fine-tuned CLIP, Frozen Encoder')
        print(f'  Per-square accuracy: {results_1b["per_square_accuracy"]:.4f} ({results_1b["per_square_accuracy"]*100:.2f}%)')
        print(f'  Exact board match: {results_1b["exact_board_match"]:.4f}')
        print(f'  Best epoch: {results_1b["best_epoch"]}')
        print()
        print('Exp 1D: Base CLIP, Unfrozen Encoder')
        print(f'  Per-square accuracy: {results_1d["per_square_accuracy"]:.4f} ({results_1d["per_square_accuracy"]*100:.2f}%)')
        print(f'  Exact board match: {results_1d["exact_board_match"]:.4f}')
        print(f'  Best epoch: {results_1d["best_epoch"]}')
        print()
        print('='*70)
        print('KEY FINDINGS:')
        print('='*70)
        
        diff_1a_1b = results_1b['per_square_accuracy'] - results_1a['per_square_accuracy']
        print(f'1A vs 1B (Frozen): Chess fine-tuning benefit = {diff_1a_1b:+.4f} ({diff_1a_1b*100:+.2f}%)')
        print('   -> Minimal benefit when encoder is frozen')
        print()
        
        diff_1a_1d = results_1d['per_square_accuracy'] - results_1a['per_square_accuracy']
        print(f'1A vs 1D (Unfrozen): End-to-end training benefit = {diff_1a_1d:+.4f} ({diff_1a_1d*100:+.2f}%)')
        if diff_1a_1d > 0:
            print('   -> End-to-end training helps!')
        else:
            print('   -> End-to-end training does not help (or hurts)')
        
        # Save
        Path('results').mkdir(exist_ok=True, parents=True)
        with open('results/exp1d_results.json', 'w') as f:
            json.dump(results_1d, f, indent=2)
        
        # Save full comparison
        comparison = {
            'exp1a': results_1a,
            'exp1b': results_1b,
            'exp1d': results_1d,
            'findings': {
                'frozen_encoder': {
                    'chess_finetuning_benefit': float(diff_1a_1b),
                    'conclusion': 'Minimal benefit from chess fine-tuning when encoder is frozen'
                },
                'unfrozen_encoder': {
                    'end_to_end_benefit': float(diff_1a_1d),
                    'conclusion': 'End-to-end training helps' if diff_1a_1d > 0 else 'End-to-end training does not help'
                }
            }
        }
        with open('results/full_ablation_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print()
        print('Results saved to results/')
    else:
        print("Could not extract metrics from checkpoint")
else:
    print(f"Checkpoint not found: {checkpoint_path}")

