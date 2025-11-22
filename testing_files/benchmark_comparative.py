# benchmark_comparative.py
import argparse
import pandas as pd
from benchmark_individual import evaluate_model # Imports the function from the other script

def run_comparison(model_a_path: str, model_b_path: str, test_data_a_dir: str, test_data_b_dir: str, 
                   model_a_name: str, model_b_name: str):
    """Runs evaluation on two models and prints a comparative summary."""
    print("="*60)
    print(f"Benchmarking Model A: '{model_a_name}'")
    print("="*60)
    metrics_a = evaluate_model(model_a_path, test_data_a_dir)
    
    if metrics_a is None: 
        print("❌ Failed to evaluate Model A")
        return

    print("\n" + "="*60)
    print(f"Benchmarking Model B: '{model_b_name}'")
    print("="*60)
    metrics_b = evaluate_model(model_b_path, test_data_b_dir)

    if metrics_b is None: 
        print("❌ Failed to evaluate Model B")
        return

    print("\n" + "="*70)
    print("                 COMPARATIVE BENCHMARK SUMMARY")
    print("="*70)
    
    # Remove Total Samples from comparison display
    display_metrics_a = {k: v for k, v in metrics_a.items() if "Samples" not in k}
    display_metrics_b = {k: v for k, v in metrics_b.items() if "Samples" not in k}
    
    df = pd.DataFrame([display_metrics_a, display_metrics_b], index=[model_a_name, model_b_name])
    print(df.to_string(float_format="%.2f"))
    
    print(f"\nDataset sizes:")
    print(f"  {model_a_name}: {metrics_a['Total Samples']} samples")
    print(f"  {model_b_name}: {metrics_b['Total Samples']} samples")
    
    # Show which model performed better
    print(f"\n🏆 Performance Summary:")
    for metric in display_metrics_a.keys():
        a_val = metrics_a[metric]
        b_val = metrics_b[metric]
        if a_val > b_val:
            winner = model_a_name
            diff = a_val - b_val
        else:
            winner = model_b_name
            diff = b_val - a_val
        print(f"  {metric}: {winner} wins by {diff:.2f}%")
    
    print("="*70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare two fine-tuned CLIP models for chess.")
    parser.add_argument("model_a_path", help="Path to the first model checkpoint (e.g., FEN only).")
    parser.add_argument("test_data_a_dir", help="Path to the test data for the first model.")
    
    parser.add_argument("model_b_path", help="Path to the second model checkpoint (e.g., FEN + Move).")
    parser.add_argument("test_data_b_dir", help="Path to the test data for the second model.")

    parser.add_argument("--model_a_name", default="FEN Only", help="Name for the first model.")
    parser.add_argument("--model_b_name", default="FEN + Next Move", help="Name for the second model.")

    args = parser.parse_args()

    run_comparison(args.model_a_path, args.model_b_path, 
                   args.test_data_a_dir, args.test_data_b_dir,
                   args.model_a_name, args.model_b_name)
