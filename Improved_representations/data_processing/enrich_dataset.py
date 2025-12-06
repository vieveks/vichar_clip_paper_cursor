"""
Script to enrich chess dataset with multiple representations.

This script reads the original dataset (CSV files with FEN strings) and creates
enriched versions with all representations: grid, JSON, graph, natural language, and tactics.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional

from .representations import (
    fen_to_grid,
    board_to_json,
    board_to_graph,
    board_to_natural_language,
    analyze_tactics
)


def enrich_single_sample(row: pd.Series, image_base_dir: Path) -> Dict:
    """
    Enrich a single dataset sample with all representations.
    
    Args:
        row: Pandas Series with dataset row (image_path, fen, etc.)
        image_base_dir: Base directory for image paths
    
    Returns:
        Dictionary with all representations
    """
    fen = row['fen']
    image_path = row['image_path']
    
    # Handle relative paths
    if not Path(image_path).is_absolute():
        image_path = str(image_base_dir / image_path)
    
    # Convert to all representations
    try:
        grid = fen_to_grid(fen)
        json_repr = board_to_json(fen)
        graph_repr = board_to_graph(fen)
        nl_repr = board_to_natural_language(fen)
        tactics = analyze_tactics(fen)
        
        return {
            'image_path': image_path,
            'fen': fen,
            'grid': grid,
            'json': json_repr,
            'graph': graph_repr,
            'natural_language': nl_repr,
            'tactics': tactics,
            # Preserve original fields if they exist
            'active_color': row.get('active_color', ''),
            'castling_rights': row.get('castling_rights', ''),
            'en_passant_target_square': row.get('en_passant_target_square', ''),
            'best_continuation': row.get('best_continuation', '')
        }
    except Exception as e:
        print(f"Error processing sample {row.name}: {e}")
        return None


def enrich_dataset_split(
    csv_path: str,
    output_path: str,
    image_base_dir: Optional[str] = None,
    max_samples: Optional[int] = None
):
    """
    Enrich a dataset split (train/val/test).
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to output JSON file
        image_base_dir: Base directory for image paths (if None, uses CSV directory)
        max_samples: Maximum number of samples to process (for testing)
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Determine image base directory
    if image_base_dir is None:
        csv_path_obj = Path(csv_path)
        # Try to find images directory relative to CSV
        if 'train' in csv_path:
            image_base_dir = csv_path_obj.parent.parent / 'train' / 'images'
        elif 'validation' in csv_path or 'val' in csv_path:
            image_base_dir = csv_path_obj.parent.parent / 'validation' / 'images'
        elif 'test' in csv_path:
            image_base_dir = csv_path_obj.parent.parent / 'test' / 'images'
        else:
            image_base_dir = csv_path_obj.parent
    else:
        image_base_dir = Path(image_base_dir)
    
    # Limit samples if specified
    if max_samples:
        df = df.head(max_samples)
        print(f"Processing first {max_samples} samples...")
    
    print(f"Enriching {len(df)} samples...")
    
    enriched_samples = []
    failed_samples = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        enriched = enrich_single_sample(row, image_base_dir)
        if enriched:
            enriched_samples.append(enriched)
        else:
            failed_samples += 1
    
    print(f"Successfully enriched {len(enriched_samples)} samples")
    if failed_samples > 0:
        print(f"Failed to process {failed_samples} samples")
    
    # Save enriched dataset
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving enriched dataset to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(enriched_samples, f, indent=2)
    
    print(f"Done! Saved {len(enriched_samples)} enriched samples.")


def main():
    parser = argparse.ArgumentParser(description='Enrich chess dataset with multiple representations')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to output JSON file')
    parser.add_argument('--image_base_dir', type=str, default=None,
                       help='Base directory for image paths (auto-detected if not specified)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    
    args = parser.parse_args()
    
    enrich_dataset_split(
        csv_path=args.csv_path,
        output_path=args.output_path,
        image_base_dir=args.image_base_dir,
        max_samples=args.max_samples
    )


if __name__ == '__main__':
    main()

