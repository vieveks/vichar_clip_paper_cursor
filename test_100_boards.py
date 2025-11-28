"""
Test the trained CLIP model on 100 boards from test.csv and plot accuracy graphs.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import open_clip
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from tqdm import tqdm
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))
from hf_chess_dataset_loader import load_hf_dataset_splits
from train_clip_hf_dataset_fast import create_transforms

# Setup
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Paths
MODEL_PATH = Path("runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt")
DATA_DIR = "data/hf_chess_puzzles"
OUTPUT_DIR = Path("test_100_boards_results")
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_TEST_BOARDS = 100


def load_trained_model(device):
    """Load the trained CLIP model from checkpoint."""
    print(f"Loading trained model from {MODEL_PATH}...")
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")
    
    # Create model with same architecture as training
    model, _, _ = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="laion2B-s34B-b79K",
        device=device
    )
    
    # Load trained weights
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    
    print(f"✅ Model loaded successfully (epoch {checkpoint.get('epoch', 'N/A')})")
    return model, tokenizer


def evaluate_on_100_boards(model, test_loader, device, tokenizer, num_boards=100):
    """Evaluate model on 100 boards from test set."""
    print(f"\nEvaluating on {num_boards} boards...")
    model.eval()
    
    # Track per-sample results for detailed analysis
    per_sample_results = []
    
    # Accuracy metrics
    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0
    total_samples = 0
    
    # For batch-wise accuracy tracking
    batch_accuracies = []
    
    with torch.no_grad():
        for batch_idx, (images, texts) in enumerate(tqdm(test_loader, desc="Evaluating")):
            if total_samples >= num_boards:
                break
                
            images = images.to(device)
            texts = list(texts)  # Convert to list for tokenization
            text_tokens = tokenizer(texts).to(device)
            
            # Get features
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                image_features, text_features, logit_scale = model(images, text_tokens)
            
            # Normalize features
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Compute similarity matrix
            logits_per_image = logit_scale * image_features @ text_features.t()
            
            # Get batch size (may be smaller for last batch)
            batch_size = logits_per_image.shape[0]
            actual_batch_size = min(batch_size, num_boards - total_samples)
            
            # Calculate accuracy for this batch
            for i in range(actual_batch_size):
                # Get top-k predictions for this sample
                top1_pred = logits_per_image[i].argmax().item()
                top5_preds = logits_per_image[i].topk(min(5, batch_size))[1].cpu().numpy()
                top10_preds = logits_per_image[i].topk(min(10, batch_size))[1].cpu().numpy()
                
                # Check if correct (diagonal should be highest)
                is_top1_correct = (top1_pred == i)
                is_top5_correct = (i in top5_preds)
                is_top10_correct = (i in top10_preds)
                
                # Store per-sample result
                per_sample_results.append({
                    'sample_idx': total_samples,
                    'top1_correct': is_top1_correct,
                    'top5_correct': is_top5_correct,
                    'top10_correct': is_top10_correct,
                    'top1_pred': top1_pred,
                    'true_idx': i,
                    'confidence': logits_per_image[i, i].item()
                })
                
                if is_top1_correct:
                    correct_top1 += 1
                if is_top5_correct:
                    correct_top5 += 1
                if is_top10_correct:
                    correct_top10 += 1
                
                total_samples += 1
            
            # Calculate batch accuracy
            batch_top1 = (logits_per_image[:actual_batch_size].argmax(dim=1) == 
                         torch.arange(actual_batch_size, device=device)).float().mean().item()
            batch_accuracies.append(batch_top1)
    
    # Calculate final accuracies
    top1_acc = (correct_top1 / total_samples) * 100
    top5_acc = (correct_top5 / total_samples) * 100
    top10_acc = (correct_top10 / total_samples) * 100
    
    results = {
        'total_samples': total_samples,
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'top10_accuracy': top10_acc,
        'correct_top1': correct_top1,
        'correct_top5': correct_top5,
        'correct_top10': correct_top10,
        'per_sample_results': per_sample_results,
        'batch_accuracies': batch_accuracies
    }
    
    return results


def plot_accuracy_graphs(results, output_dir):
    """Create comprehensive accuracy graphs."""
    print("\nGenerating accuracy graphs...")
    
    # 1. Main accuracy bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Top-1', 'Top-5', 'Top-10']
    accuracies = [
        results['top1_accuracy'],
        results['top5_accuracy'],
        results['top10_accuracy']
    ]
    
    bars = ax.bar(metrics, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.2f}%',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'CLIP Model Accuracy on {results["total_samples"]} Test Boards', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_bar_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Batch-wise accuracy progression
    if len(results['batch_accuracies']) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        batch_indices = np.arange(1, len(results['batch_accuracies']) + 1)
        ax.plot(batch_indices, [acc * 100 for acc in results['batch_accuracies']], 
               marker='o', linewidth=2, markersize=6, color='#3498db')
        ax.axhline(y=results['top1_accuracy'], color='r', linestyle='--', 
                  label=f'Overall: {results["top1_accuracy"]:.2f}%', linewidth=2)
        ax.set_xlabel('Batch Number', fontsize=12)
        ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        ax.set_title('Batch-wise Accuracy Progression', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "batch_accuracy_progression.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Confidence distribution for correct vs incorrect predictions
    per_sample = results['per_sample_results']
    correct_confidences = [r['confidence'] for r in per_sample if r['top1_correct']]
    incorrect_confidences = [r['confidence'] for r in per_sample if not r['top1_correct']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    if correct_confidences:
        ax.hist(correct_confidences, bins=20, alpha=0.7, label='Correct Predictions', 
               color='#2ecc71', edgecolor='black')
    if incorrect_confidences:
        ax.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect Predictions', 
               color='#e74c3c', edgecolor='black')
    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Confidence Distribution: Correct vs Incorrect Predictions', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Cumulative accuracy plot
    per_sample = results['per_sample_results']
    cumulative_correct = np.cumsum([1 if r['top1_correct'] else 0 for r in per_sample])
    cumulative_accuracy = (cumulative_correct / np.arange(1, len(per_sample) + 1)) * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.arange(1, len(per_sample) + 1), cumulative_accuracy, 
           linewidth=2, color='#3498db')
    ax.axhline(y=results['top1_accuracy'], color='r', linestyle='--', 
              label=f'Final: {results["top1_accuracy"]:.2f}%', linewidth=2)
    ax.set_xlabel('Number of Samples Evaluated', fontsize=12)
    ax.set_ylabel('Cumulative Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Cumulative Accuracy Over Test Samples', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Accuracy comparison (all metrics together)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.6
    bars = ax.bar(x, accuracies, width, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.2f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Model Performance on {results["total_samples"]} Test Boards', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved all graphs to {output_dir}")


def main():
    """Main evaluation pipeline."""
    print("="*60)
    print("Testing CLIP Model on 100 Boards from Test Set")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test dataset
    print(f"\nLoading test dataset from {DATA_DIR}...")
    transform = create_transforms()
    _, _, test_dataset = load_hf_dataset_splits(DATA_DIR, transform=transform)
    print(f"  Total test samples: {len(test_dataset):,}")
    
    # Create subset of 100 boards
    if len(test_dataset) < NUM_TEST_BOARDS:
        print(f"⚠️  Warning: Test set has only {len(test_dataset)} samples, using all of them")
        num_boards = len(test_dataset)
    else:
        num_boards = NUM_TEST_BOARDS
    
    test_subset = Subset(test_dataset, range(num_boards))
    test_loader = DataLoader(
        test_subset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"  Testing on {num_boards} boards")
    
    # Load trained model
    model, tokenizer = load_trained_model(device)
    
    # Evaluate
    results = evaluate_on_100_boards(model, test_loader, device, tokenizer, num_boards=num_boards)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total Samples Tested: {results['total_samples']}")
    print(f"\nAccuracy Metrics:")
    print(f"  Top-1 Accuracy:  {results['top1_accuracy']:.2f}% ({results['correct_top1']}/{results['total_samples']})")
    print(f"  Top-5 Accuracy:  {results['top5_accuracy']:.2f}% ({results['correct_top5']}/{results['total_samples']})")
    print(f"  Top-10 Accuracy: {results['top10_accuracy']:.2f}% ({results['correct_top10']}/{results['total_samples']})")
    
    # Generate plots
    plot_accuracy_graphs(results, OUTPUT_DIR)
    
    # Save results to JSON
    # Remove per_sample_results from JSON (too large)
    json_results = {k: v for k, v in results.items() if k != 'per_sample_results'}
    json_results['per_sample_count'] = len(results['per_sample_results'])
    
    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - accuracy_bar_chart.png")
    print("  - accuracy_comparison.png")
    if len(results['batch_accuracies']) > 1:
        print("  - batch_accuracy_progression.png")
    print("  - confidence_distribution.png")
    print("  - cumulative_accuracy.png")
    print("  - results.json")


if __name__ == "__main__":
    main()

