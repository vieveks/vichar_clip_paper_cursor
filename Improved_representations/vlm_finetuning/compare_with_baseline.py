"""Compare Qwen2-VL fine-tuned results with CLIP-based JSON predictor baseline."""
import json
from pathlib import Path

# Load Qwen2-VL results
qwen_results_path = Path('Improved_representations/results/qwen2vl_eval.json')
if qwen_results_path.exists():
    with open(qwen_results_path) as f:
        qwen_results = json.load(f)
else:
    print("Qwen2-VL evaluation results not found. Run evaluation first.")
    qwen_results = None

# Load CLIP baseline results
clip_results_path = Path('Improved_representations/results/exp1b_results.json')
if clip_results_path.exists():
    with open(clip_results_path) as f:
        clip_results = json.load(f)
else:
    print("CLIP baseline results not found.")
    clip_results = None

print("="*70)
print("COMPARISON: Qwen2-VL-2B Fine-tuned vs CLIP-based JSON Predictor")
print("="*70)
print()

if clip_results:
    print("CLIP-based JSON Predictor (Exp 1B):")
    print(f"  Per-square accuracy: {clip_results['per_square_accuracy']:.4f} ({clip_results['per_square_accuracy']*100:.2f}%)")
    print(f"  Exact board match: {clip_results['exact_board_match']:.6f} ({clip_results['exact_board_match']*100:.4f}%)")
    print(f"  To-move accuracy: {clip_results['to_move_accuracy']:.4f}")
    print(f"  Castling accuracy: {clip_results['castling_accuracy']:.4f}")
    print()

if qwen_results:
    print("Qwen2-VL-2B Fine-tuned:")
    print(f"  Valid JSON rate: {qwen_results.get('valid_json_rate', 0):.4f} ({qwen_results.get('valid_json_rate', 0)*100:.2f}%)")
    print(f"  Valid position rate: {qwen_results.get('valid_position_rate', 0):.4f} ({qwen_results.get('valid_position_rate', 0)*100:.2f}%)")
    print(f"  Exact JSON match: {qwen_results.get('exact_json_match_rate', 0):.4f} ({qwen_results.get('exact_json_match_rate', 0)*100:.2f}%)")
    print(f"  FEN accuracy: {qwen_results.get('fen_accuracy_rate', 0):.4f} ({qwen_results.get('fen_accuracy_rate', 0)*100:.2f}%)")
    print(f"  Per-square accuracy: {qwen_results.get('avg_per_square_accuracy', 0):.4f} ({qwen_results.get('avg_per_square_accuracy', 0)*100:.2f}%)")
    print(f"  Exact board match: {qwen_results.get('exact_board_match_rate', 0):.4f} ({qwen_results.get('exact_board_match_rate', 0)*100:.4f}%)")
    print()
    
    if clip_results:
        print("="*70)
        print("KEY COMPARISONS:")
        print("="*70)
        qwen_square_acc = qwen_results.get('avg_per_square_accuracy', 0)
        clip_square_acc = clip_results['per_square_accuracy']
        diff = qwen_square_acc - clip_square_acc
        print(f"Per-square accuracy: Qwen2-VL ({qwen_square_acc*100:.2f}%) vs CLIP ({clip_square_acc*100:.2f}%)")
        print(f"  Difference: {diff:+.4f} ({diff*100:+.2f}%)")
        print()
        
        qwen_exact = qwen_results.get('exact_board_match_rate', 0)
        clip_exact = clip_results['exact_board_match']
        diff_exact = qwen_exact - clip_exact
        print(f"Exact board match: Qwen2-VL ({qwen_exact*100:.4f}%) vs CLIP ({clip_exact*100:.4f}%)")
        print(f"  Difference: {diff_exact:+.6f} ({diff_exact*100:+.4f}%)")
        print()
        
        print("="*70)
        print("CONCLUSION:")
        print("="*70)
        if qwen_square_acc > clip_square_acc:
            print("✅ Qwen2-VL fine-tuned model outperforms CLIP-based predictor")
        elif qwen_square_acc < clip_square_acc:
            print("⚠️  CLIP-based predictor outperforms Qwen2-VL fine-tuned model")
        else:
            print("➖ Both models perform similarly")
else:
    print("⚠️  Qwen2-VL evaluation not yet complete. Please wait for evaluation to finish.")

print("="*70)

