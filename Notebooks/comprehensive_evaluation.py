#!/usr/bin/env python3
"""
Comprehensive evaluation script to detect overfitting and test on independent datasets.
This script:
1. Properly splits data into train/val/test sets
2. Tests on completely independent external datasets
3. Analyzes train vs val vs test performance to detect overfitting
4. Tests on larger datasets from different time periods
"""

import torch
import clip
from torch.utils.data import DataLoader, random_split, Subset
from PIL import Image
import chess
import chess.pgn
import random
from pathlib import Path
import logging
from dataset_loader import ChessDataset
from dataset_prep_simple import create_chess_board_image
import pandas as pd
from tqdm import tqdm
import argparse
import numpy as np
from collections import defaultdict
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_retrieval_metrics(image_features, text_features, top_k_values=[1, 5, 10], batch_size=1000, device='cuda'):
    """
    Calculate retrieval accuracy metrics in batches to avoid memory issues.
    
    Args:
        image_features: Normalized image features [N, D] (on CPU)
        text_features: Normalized text features [N, D] (on CPU)
        top_k_values: List of k values for top-k accuracy
        batch_size: Batch size for computing similarity matrices
        device: Device to use for computation
    
    Returns:
        Dictionary with metrics
    """
    N = len(image_features)
    ground_truth = torch.arange(N, device=device)
    
    # Move text features to device once (they're used for all batches)
    text_features_gpu = text_features.to(device)
    
    metrics = {}
    
    # Initialize counters
    i2t_top_k_correct = {k: 0 for k in top_k_values}
    t2i_top_k_correct = {k: 0 for k in top_k_values}
    i2t_ranks = []
    t2i_ranks = []
    
    # Process in batches to avoid OOM
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        batch_image_features = image_features[i:end_i].to(device)
        
        # Image-to-Text: compute similarity with all texts
        logits_per_image = batch_image_features @ text_features_gpu.T  # [batch_size, N]
        i2t_sorted = torch.argsort(logits_per_image, dim=1, descending=True)
        
        # Calculate top-k accuracy for this batch
        batch_ground_truth = ground_truth[i:end_i]
        for k in top_k_values:
            top_k_preds = i2t_sorted[:, :k]
            correct = (top_k_preds == batch_ground_truth.unsqueeze(1)).any(dim=1).sum().item()
            i2t_top_k_correct[k] += correct
        
        # Calculate ranks for this batch
        for j, gt_idx in enumerate(batch_ground_truth):
            rank = (i2t_sorted[j] == gt_idx).nonzero().item() + 1
            i2t_ranks.append(rank)
        
        # Text-to-Image: compute similarity with all images (in chunks)
        batch_text_features = text_features[i:end_i].to(device)
        
        # Compute similarity in chunks to avoid loading all images at once
        all_similarities = []
        chunk_size = min(2000, N)  # Process images in chunks
        for img_start in range(0, N, chunk_size):
            img_end = min(img_start + chunk_size, N)
            img_chunk = image_features[img_start:img_end].to(device)
            similarities = batch_text_features @ img_chunk.T  # [batch_size, chunk_size]
            all_similarities.append(similarities.cpu())
            del img_chunk
            torch.cuda.empty_cache()
        
        # Concatenate all similarities
        logits_per_text = torch.cat(all_similarities, dim=1).to(device)  # [batch_size, N]
        t2i_sorted = torch.argsort(logits_per_text, dim=1, descending=True)
        
        # Calculate top-k accuracy for this batch
        for k in top_k_values:
            top_k_preds = t2i_sorted[:, :k]
            correct = (top_k_preds == batch_ground_truth.unsqueeze(1)).any(dim=1).sum().item()
            t2i_top_k_correct[k] += correct
        
        # Calculate ranks for this batch
        for j, gt_idx in enumerate(batch_ground_truth):
            rank = (t2i_sorted[j] == gt_idx).nonzero().item() + 1
            t2i_ranks.append(rank)
        
        # Clear GPU memory
        del batch_image_features, batch_text_features, logits_per_text, all_similarities
        torch.cuda.empty_cache()
    
    # Finalize metrics
    for k in top_k_values:
        metrics[f'Image-to-Text Top-{k}'] = i2t_top_k_correct[k] / N * 100
        metrics[f'Text-to-Image Top-{k}'] = t2i_top_k_correct[k] / N * 100
    
    metrics['Image-to-Text Avg Rank'] = np.mean(i2t_ranks)
    metrics['Text-to-Image Avg Rank'] = np.mean(t2i_ranks)
    metrics['Image-to-Text Median Rank'] = np.median(i2t_ranks)
    metrics['Text-to-Image Median Rank'] = np.median(t2i_ranks)
    
    return metrics

def evaluate_on_split(model, dataset, split_name, batch_size=64, device='cuda'):
    """Evaluate model on a dataset split."""
    model.eval()
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    all_image_features = []
    all_text_features = []
    
    logging.info(f"Encoding {split_name} set ({len(dataset)} samples)...")
    with torch.no_grad():
        for images, texts in tqdm(loader, desc=f"Encoding {split_name}"):
            images = images.to(device)
            texts = texts.to(device)
            
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            
            # Normalize before moving to CPU to save memory
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
    
    # Concatenate all features (keep on CPU, move to GPU in batches)
    image_features = torch.cat(all_image_features)
    text_features = torch.cat(all_text_features)
    
    # Calculate metrics (using batch processing to avoid OOM)
    metrics = calculate_retrieval_metrics(image_features, text_features, batch_size=500, device=device)
    metrics['Dataset Size'] = len(dataset)
    
    return metrics

def create_independent_test_set(pgn_path: str, output_dir: str, num_positions: int = 1000):
    """
    Create an independent test set from a PGN file that wasn't used in training.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_path / "images"
    texts_dir = output_path / "texts"
    images_dir.mkdir(exist_ok=True)
    texts_dir.mkdir(exist_ok=True)
    
    positions = []
    
    logging.info(f"Creating independent test set from {pgn_path}...")
    
    try:
        with open(pgn_path, "r", encoding="utf-8") as f:
            game_count = 0
            example_count = 0
            
            while example_count < num_positions:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                board = game.board()
                for move in game.mainline_moves():
                    if example_count >= num_positions:
                        break
                    
                    fen = board.fen()
                    
                    # Create image
                    try:
                        png_bytes = create_chess_board_image(fen, size=350)
                        
                        # Save image
                        img_path = images_dir / f"{example_count}.png"
                        with open(img_path, "wb") as img_file:
                            img_file.write(png_bytes)
                        
                        # Save FEN text
                        text_path = texts_dir / f"{example_count}.txt"
                        with open(text_path, "w", encoding="utf-8") as txt_file:
                            txt_file.write(fen)
                        
                        positions.append(fen)
                        example_count += 1
                    except Exception as e:
                        logging.warning(f"Failed to create image for position {example_count}: {e}")
                    
                    board.push(move)
                
                game_count += 1
        
        logging.info(f"Created {len(positions)} independent test positions from {game_count} games")
        return output_path
    
    except FileNotFoundError:
        logging.warning(f"PGN file not found: {pgn_path}. Skipping independent test set creation.")
        return None
    except Exception as e:
        logging.error(f"Error creating independent test set: {e}")
        return None

def analyze_overfitting(train_metrics, val_metrics, test_metrics):
    """Analyze overfitting by comparing train/val/test metrics."""
    
    print("\n" + "="*80)
    print("OVERFITTING ANALYSIS")
    print("="*80)
    
    # Create comparison DataFrame
    comparison_data = {
        'Train': train_metrics,
        'Validation': val_metrics,
        'Test': test_metrics
    }
    
    df = pd.DataFrame(comparison_data).T
    
    # Display metrics
    print("\nPerformance Comparison:")
    print(df.to_string(float_format="%.2f"))
    
    # Calculate overfitting indicators
    print("\nOverfitting Indicators:")
    
    key_metrics = ['Image-to-Text Top-1', 'Image-to-Text Top-5', 'Text-to-Image Top-1', 'Text-to-Image Top-5']
    
    for metric in key_metrics:
        if metric in train_metrics and metric in val_metrics and metric in test_metrics:
            train_val_gap = train_metrics[metric] - val_metrics[metric]
            val_test_gap = val_metrics[metric] - test_metrics[metric]
            
            print(f"\n{metric}:")
            print(f"  Train-Val Gap: {train_val_gap:.2f}%")
            print(f"  Val-Test Gap: {val_test_gap:.2f}%")
            
            if train_val_gap > 10:
                print(f"  WARNING: Large train-val gap suggests overfitting!")
            if val_test_gap > 5:
                print(f"  WARNING: Large val-test gap suggests poor generalization!")
    
    # Overall assessment
    avg_train_val_gap = np.mean([train_metrics.get(m, 0) - val_metrics.get(m, 0) 
                                  for m in key_metrics if m in train_metrics])
    avg_val_test_gap = np.mean([val_metrics.get(m, 0) - test_metrics.get(m, 0) 
                                for m in key_metrics if m in val_metrics])
    
    print(f"\nOverall Assessment:")
    print(f"  Average Train-Val Gap: {avg_train_val_gap:.2f}%")
    print(f"  Average Val-Test Gap: {avg_val_test_gap:.2f}%")
    
    if avg_train_val_gap > 10 or avg_val_test_gap > 5:
        print(f"  WARNING: Model shows signs of overfitting!")
    else:
        print(f"  Model shows good generalization!")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation to detect overfitting")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to training dataset directory")
    parser.add_argument("--independent_pgn", type=str, default=None,
                       help="Path to PGN file for independent test set (different time period/source)")
    parser.add_argument("--independent_test_dir", type=str, default="independent_test_set",
                       help="Directory to save independent test set")
    parser.add_argument("--num_independent", type=int, default=1000,
                       help="Number of positions in independent test set")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for evaluation")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Ratio of training data")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                       help="Ratio of validation data")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                       help="Ratio of test data (from training dataset)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Load model
    logging.info(f"Loading model from {args.model_path}...")
    try:
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return
    
    # Load dataset
    logging.info(f"Loading dataset from {args.data_dir}...")
    dataset = ChessDataset(args.data_dir, preprocess)
    logging.info(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        logging.error("Dataset is empty!")
        return
    
    # Split dataset into train/val/test
    total_size = len(dataset)
    train_size = int(args.train_ratio * total_size)
    val_size = int(args.val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    logging.info(f"Splitting dataset: Train={train_size}, Val={val_size}, Test={test_size}")
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Evaluate on train/val/test splits
    print("\n" + "="*80)
    print("EVALUATION ON TRAINING DATASET SPLITS")
    print("="*80)
    
    train_metrics = evaluate_on_split(model, train_dataset, "Train", args.batch_size, device)
    val_metrics = evaluate_on_split(model, val_dataset, "Validation", args.batch_size, device)
    test_metrics = evaluate_on_split(model, test_dataset, "Test", args.batch_size, device)
    
    # Analyze overfitting
    comparison_df = analyze_overfitting(train_metrics, val_metrics, test_metrics)
    
    # Create independent test set if PGN provided
    independent_test_path = None
    if args.independent_pgn:
        # Update path if using new data structure
        if not Path(args.independent_pgn).exists() and Path("../data/pgn_files/anand_pgns/Anand.pgn").exists():
            args.independent_pgn = "../data/pgn_files/anand_pgns/Anand.pgn"
        independent_test_path = create_independent_test_set(
            args.independent_pgn, 
            args.independent_test_dir,
            args.num_independent
        )
    
    # Evaluate on independent test set if available
    independent_metrics = None
    if independent_test_path and (independent_test_path / "images").exists():
        print("\n" + "="*80)
        print("EVALUATION ON INDEPENDENT TEST SET")
        print("="*80)
        
        independent_dataset = ChessDataset(str(independent_test_path), preprocess)
        if len(independent_dataset) > 0:
            independent_metrics = evaluate_on_split(
                model, independent_dataset, "Independent", args.batch_size, device
            )
            
            print("\nIndependent Test Set Results:")
            for key, value in independent_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
            
            # Compare with test set from training data
            print("\nComparison: Test Set vs Independent Test Set")
            comparison_independent = pd.DataFrame({
                'Test (from training data)': test_metrics,
                'Independent Test': independent_metrics
            })
            print(comparison_independent.to_string(float_format="%.2f"))
            
            # Check for significant drop
            key_metric = 'Image-to-Text Top-1'
            if key_metric in test_metrics and key_metric in independent_metrics:
                drop = test_metrics[key_metric] - independent_metrics[key_metric]
                if drop > 10:
                    print(f"\nWARNING: Significant performance drop ({drop:.2f}%) on independent test set!")
                    print("   This suggests the model may have overfitted to the training distribution.")
                else:
                    print(f"\nGood generalization: Only {drop:.2f}% drop on independent test set.")
    
    # Save results
    results = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'independent_metrics': independent_metrics if independent_test_path else None
    }
    
    results_file = "comprehensive_evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    comparison_df.to_csv("train_val_test_comparison.csv")
    print(f"\nResults saved to {results_file} and train_val_test_comparison.csv")
    
    print("\nComprehensive evaluation complete!")

if __name__ == "__main__":
    main()

