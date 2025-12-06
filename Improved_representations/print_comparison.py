"""Print ablation study comparison."""
import json
from pathlib import Path

# Get script directory
script_dir = Path(__file__).parent
results_dir = script_dir / 'results'

# Load results
with open(results_dir / 'exp1a_results.json') as f:
    exp1a = json.load(f)
with open(results_dir / 'exp1b_results.json') as f:
    exp1b = json.load(f)
with open(results_dir / 'exp1d_results.json') as f:
    exp1d = json.load(f)

print('='*70)
print('ABLATION STUDY RESULTS - ALL EXPERIMENTS')
print('='*70)
print()
print('Exp 1A: Base CLIP, Frozen Encoder')
print(f'  Per-square accuracy: {exp1a["per_square_accuracy"]:.4f} ({exp1a["per_square_accuracy"]*100:.2f}%)')
print(f'  Exact board match: {exp1a["exact_board_match"]:.6f}')
print(f'  Best epoch: {exp1a["best_epoch"]}')
print()
print('Exp 1B: Fine-tuned CLIP, Frozen Encoder')
print(f'  Per-square accuracy: {exp1b["per_square_accuracy"]:.4f} ({exp1b["per_square_accuracy"]*100:.2f}%)')
print(f'  Exact board match: {exp1b["exact_board_match"]:.6f}')
print(f'  Best epoch: {exp1b["best_epoch"]}')
print()
print('Exp 1D: Base CLIP, Unfrozen Encoder')
print(f'  Per-square accuracy: {exp1d["per_square_accuracy"]:.4f} ({exp1d["per_square_accuracy"]*100:.2f}%)')
print(f'  Exact board match: {exp1d["exact_board_match"]:.6f}')
print(f'  Best epoch: {exp1d["best_epoch"]}')
print()
print('='*70)
print('KEY FINDINGS:')
print('='*70)
diff_1a_1b = exp1b['per_square_accuracy'] - exp1a['per_square_accuracy']
print(f'1A vs 1B (Frozen): Chess fine-tuning benefit = {diff_1a_1b:+.4f} ({diff_1a_1b*100:+.2f}%)')
print('   -> Minimal benefit when encoder is frozen')
print()
diff_1a_1d = exp1d['per_square_accuracy'] - exp1a['per_square_accuracy']
print(f'1A vs 1D (Unfrozen): End-to-end training benefit = {diff_1a_1d:+.4f} ({diff_1a_1d*100:+.2f}%)')
print('   -> End-to-end training does not help (slightly worse performance)')
print()
print('Conclusion: Freezing the encoder provides better results than unfreezing it.')

