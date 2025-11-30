"""
Script to combine separate baseline and chess-CLIP evaluation results
and generate a comparison summary.
"""

import argparse
import json
import os
from pathlib import Path


def load_results(file_path):
    """Load results from JSON file."""
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as f:
        return json.load(f)


def combine_and_compare(baseline_path, chess_path, output_dir):
    """Combine results and generate comparison."""
    
    # Load results
    baseline_data = load_results(baseline_path)
    chess_data = load_results(chess_path)
    
    if not baseline_data:
        print(f"Warning: Baseline results not found at {baseline_path}")
        print("  You may need to run baseline evaluation first:")
        print("  python clip_as_encoder/evaluate.py --evaluate_baseline_only ...")
        return
    
    if not chess_data:
        print(f"Warning: Chess-CLIP results not found at {chess_path}")
        return
    
    # Extract summaries
    baseline_summary = baseline_data.get("baseline", {}).get("summary", {})
    chess_summary = chess_data.get("chess_clip", {}).get("summary", {})
    
    if not baseline_summary or not chess_summary:
        print("Error: Could not find summary in results files")
        return
    
    # Calculate improvements
    score_improvement = chess_summary["average_score"] - baseline_summary["average_score"]
    accuracy_improvement = chess_summary["accuracy"] - baseline_summary["accuracy"]
    score_improvement_percent = (score_improvement / max(baseline_summary['average_score'], 0.01)) * 100
    
    # Create comparison
    comparison = {
        "baseline_summary": baseline_summary,
        "chess_clip_summary": chess_summary,
        "improvements": {
            "score_improvement": float(score_improvement),
            "score_improvement_percent": float(score_improvement_percent),
            "accuracy_improvement": float(accuracy_improvement)
        },
        "per_question_type": {}
    }
    
    # Per-question comparison
    baseline_per_q = baseline_summary.get("per_question_type", {})
    chess_per_q = chess_summary.get("per_question_type", {})
    
    for qtype in baseline_per_q.keys():
        if qtype in chess_per_q:
            baseline_q = baseline_per_q[qtype]
            chess_q = chess_per_q[qtype]
            
            score_diff = chess_q["average_score"] - baseline_q["average_score"]
            acc_diff = chess_q["accuracy"] - baseline_q["accuracy"]
            
            comparison["per_question_type"][qtype] = {
                "baseline": baseline_q,
                "chess_clip": chess_q,
                "score_improvement": float(score_diff),
                "accuracy_improvement": float(acc_diff)
            }
    
    # Save comparison
    comparison_path = os.path.join(output_dir, "comparison_summary.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\nBaseline (Generic CLIP):")
    print(f"  Average Score: {baseline_summary['average_score']:.3f}")
    print(f"  Accuracy: {baseline_summary['accuracy']:.2f}%")
    
    print(f"\nChessCLIP (Chess-Finetuned CLIP):")
    print(f"  Average Score: {chess_summary['average_score']:.3f}")
    print(f"  Accuracy: {chess_summary['accuracy']:.2f}%")
    
    print(f"\nImprovements:")
    print(f"  Score Improvement: {score_improvement:+.3f} ({score_improvement_percent:+.1f}%)")
    print(f"  Accuracy Improvement: {accuracy_improvement:+.2f}%")
    
    print(f"\nPer-Question Type Comparison:")
    for qtype, qcomp in comparison["per_question_type"].items():
        print(f"\n  {qtype}:")
        print(f"    Baseline: Score={qcomp['baseline']['average_score']:.3f}, Acc={qcomp['baseline']['accuracy']:.2f}%")
        print(f"    ChessCLIP: Score={qcomp['chess_clip']['average_score']:.3f}, Acc={qcomp['chess_clip']['accuracy']:.2f}%")
        print(f"    Improvement: Score {qcomp['score_improvement']:+.3f}, Acc {qcomp['accuracy_improvement']:+.2f}%")
    
    print(f"\n[OK] Comparison saved to {comparison_path}")
    
    # Also combine full results
    combined_results = {
        "baseline": baseline_data.get("baseline", {}),
        "chess_clip": chess_data.get("chess_clip", {})
    }
    
    combined_path = os.path.join(output_dir, "combined_results.json")
    with open(combined_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"[OK] Combined results saved to {combined_path}")


def main():
    parser = argparse.ArgumentParser(description="Combine baseline and chess-CLIP evaluation results")
    parser.add_argument(
        "--baseline_results",
        type=str,
        default="clip_as_encoder/evaluation_results/baseline_results.json",
        help="Path to baseline results JSON"
    )
    parser.add_argument(
        "--chess_results",
        type=str,
        default="clip_as_encoder/evaluation_results/chess_clip_results.json",
        help="Path to chess-CLIP results JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="clip_as_encoder/evaluation_results",
        help="Output directory for combined results"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    combine_and_compare(args.baseline_results, args.chess_results, args.output_dir)


if __name__ == "__main__":
    main()

