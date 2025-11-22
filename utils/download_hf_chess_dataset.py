"""
Script to download and prepare the Hugging Face chess-puzzles-images-mini dataset.

Dataset: https://huggingface.co/datasets/bingbangboom/chess-puzzles-images-mini
- Total: 125k samples
- Pre-split: train (100k), validation (12.5k), test (12.5k)
- Fields: image, board_state (FEN), active_color, castling_rights, en_passant_target_square, best_continuation
"""

import argparse
import os
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import json
from PIL import Image
import io

def download_and_prepare_dataset(output_dir: str, save_format: str = "csv"):
    """
    Download the Hugging Face chess dataset and prepare it for training.
    
    Args:
        output_dir: Directory to save the prepared dataset
        save_format: Format to save ('csv' or 'local_images')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("bingbangboom/chess-puzzles-images-mini")
    
    print(f"Dataset loaded. Splits: {list(dataset.keys())}")
    
    # Process each split
    splits_info = {}
    
    for split_name in ['train', 'validation', 'test']:
        if split_name not in dataset:
            print(f"Warning: Split '{split_name}' not found in dataset")
            continue
            
        split_data = dataset[split_name]
        print(f"\nProcessing {split_name} split ({len(split_data)} samples)...")
        
        records = []
        
        for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
            # Extract image and FEN
            image = sample['image']
            board_state = sample['board_state']  # This is the FEN string
            
            # Additional metadata
            active_color = sample.get('active_color', '')
            castling_rights = sample.get('castling_rights', '')
            en_passant = sample.get('en_passant_target_square', '')
            best_continuation = sample.get('best_continuation', '')
            
            if save_format == "local_images":
                # Save images locally
                image_dir = output_path / split_name / "images"
                text_dir = output_path / split_name / "texts"
                image_dir.mkdir(parents=True, exist_ok=True)
                text_dir.mkdir(parents=True, exist_ok=True)
                
                # Save image
                image_path = image_dir / f"{split_name}_{idx:06d}.png"
                image.save(image_path)
                
                # Save FEN text
                text_path = text_dir / f"{split_name}_{idx:06d}.txt"
                with open(text_path, 'w') as f:
                    f.write(board_state)
                
                records.append({
                    'image_path': str(image_path),
                    'fen': board_state,
                    'active_color': active_color,
                    'castling_rights': castling_rights,
                    'en_passant_target_square': en_passant,
                    'best_continuation': best_continuation
                })
            else:
                # CSV format - store image as bytes or path reference
                # For CSV, we'll save images separately and reference them
                image_dir = output_path / split_name / "images"
                image_dir.mkdir(parents=True, exist_ok=True)
                
                image_path = image_dir / f"{split_name}_{idx:06d}.png"
                image.save(image_path)
                
                records.append({
                    'image_path': str(image_path.relative_to(output_path)),
                    'fen': board_state,
                    'active_color': active_color,
                    'castling_rights': castling_rights,
                    'en_passant_target_square': en_passant,
                    'best_continuation': best_continuation
                })
        
        # Save CSV
        df = pd.DataFrame(records)
        csv_path = output_path / f"{split_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {split_name} CSV: {csv_path} ({len(df)} samples)")
        
        splits_info[split_name] = {
            'size': len(df),
            'csv_path': str(csv_path),
            'image_dir': str(output_path / split_name / "images")
        }
    
    # Save dataset metadata
    metadata = {
        'dataset_name': 'bingbangboom/chess-puzzles-images-mini',
        'source': 'https://huggingface.co/datasets/bingbangboom/chess-puzzles-images-mini',
        'total_samples': sum(splits_info[s]['size'] for s in splits_info),
        'splits': splits_info,
        'description': 'Chess puzzles dataset with 125k positions from Lichess puzzles',
        'fields': {
            'image': 'Chess board image (512x512)',
            'board_state': 'FEN string (shortened)',
            'active_color': 'w or b (whose turn)',
            'castling_rights': 'Castling availability',
            'en_passant_target_square': 'En passant target or -',
            'best_continuation': 'Best move sequence'
        }
    }
    
    metadata_path = output_path / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[SUCCESS] Dataset preparation complete!")
    print(f"Metadata saved to: {metadata_path}")
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {metadata['total_samples']}")
    for split_name, info in splits_info.items():
        print(f"  {split_name}: {info['size']} samples")
    
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare Hugging Face chess dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/hf_chess_puzzles",
        help="Output directory for prepared dataset"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "local_images"],
        default="csv",
        help="Save format: 'csv' (with image paths) or 'local_images' (full local structure)"
    )
    
    args = parser.parse_args()
    
    download_and_prepare_dataset(args.output_dir, args.format)

