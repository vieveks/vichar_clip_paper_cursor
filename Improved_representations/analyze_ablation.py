"""Analyze all ablation experiment results."""
import torch
import json
from pathlib import Path
import os

# Get script directory
script_dir = Path(__file__).parent
base_dir = script_dir.parent if script_dir.name == 'Improved_representations' else script_dir

# Load Exp 1D checkpoint
checkpoint_path = base_dir / 'Improved_representations' / 'checkpoints' / 'exp1d_base_unfrozen' / 'best_model.pt'
checkpoint_1d = torch.load(checkpoint_path, map_location='cpu')
val_metrics_1d = checkpoint_1d['val_metrics']
epoch_1d = checkpoint_1d['epoch']

results_1d = {
    'experiment': 'Exp 1D: Base CLIP, Unfrozen Encoder',
    'best_epoch': epoch_1d,
    'per_square_accuracy': val_metrics_1d['per_square_acc'],
    'exact_board_match': val_metrics_1d['exact_board_match'],
    'to_move_accuracy': val_metrics_1d['to_move_acc'],
    'castling_accuracy': val_metrics_1d['castling_acc'],
    'loss': val_metrics_1d['loss']
}

# Load previous results
results_dir = base_dir / 'Improved_representations' / 'results'
with open(results_dir / 'exp1a_results.json') as f:
    results_1a = json.load(f)
with open(results_dir / 'exp1b_results.json') as f:
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
results_dir.mkdir(exist_ok=True, parents=True)
with open(results_dir / 'exp1d_results.json', 'w') as f:
    json.dump(results_1d, f, indent=2)

# Save full comparison
comparison = {
    'exp1a': results_1a,
    'exp1b': results_1b,
    'exp1d': results_1d,
    'findings': {
        'frozen_encoder': {
            'chess_finetuning_benefit': diff_1a_1b,
            'conclusion': 'Minimal benefit from chess fine-tuning when encoder is frozen'
        },
        'unfrozen_encoder': {
            'end_to_end_benefit': diff_1a_1d,
            'conclusion': 'End-to-end training helps' if diff_1a_1d > 0 else 'End-to-end training does not help'
        }
    }
}
with open(results_dir / 'full_ablation_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

print()
print(f'Results saved to {results_dir}')


