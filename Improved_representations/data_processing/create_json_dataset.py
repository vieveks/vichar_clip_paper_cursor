"""
Script to create JSON dataset from original CSV files.

This script:
1. Reads the original CSV files (image_path, fen, etc.)
2. Converts each FEN to JSON representation
3. Verifies round-trip consistency
4. Saves as structured JSON files
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional

from .representations import board_to_json
from .converters import round_trip_test, validate_json_position


def process_single_row(row: pd.Series, image_base_dir: Path) -> Optional[Dict]:
    """
    Process a single CSV row into JSON dataset format.
    
    Args:
        row: Pandas Series with CSV row data
        image_base_dir: Base directory for resolving image paths
    
    Returns:
        Dictionary with image_path, fen, json_repr, or None if failed
    """
    try:
        fen = row['fen']
        image_path = row['image_path']
        
        # Resolve image path
        if not Path(image_path).is_absolute():
            # Make path relative to project root
            full_image_path = str(image_base_dir / image_path)
        else:
            full_image_path = image_path
        
        # Convert to JSON representation
        json_repr = board_to_json(fen)
        
        # Verify round-trip
        passed, original, reconstructed = round_trip_test(fen)
        if not passed:
            print(f"Warning: Round-trip failed for FEN: {fen[:30]}...")
            return None
        
        # Validate JSON
        is_valid, errors = validate_json_position(json_repr)
        if not is_valid:
            print(f"Warning: Invalid position: {errors}")
            return None
        
        return {
            'image_path': full_image_path,
            'fen': fen,
            'json_repr': json_repr,
            # Preserve original metadata
            'active_color': row.get('active_color', ''),
            'castling_rights': row.get('castling_rights', ''),
            'en_passant_target_square': row.get('en_passant_target_square', ''),
            'best_continuation': row.get('best_continuation', '')
        }
    
    except Exception as e:
        print(f"Error processing row: {e}")
        return None


def create_json_dataset(
    input_csv: str,
    output_path: str,
    image_base_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    verify_images: bool = False,
    use_jsonl: bool = True
):
    """
    Create JSON dataset from CSV file.
    
    Args:
        input_csv: Path to input CSV file
        output_path: Path to output JSON file
        image_base_dir: Base directory for image paths
        max_samples: Maximum samples to process (for testing)
        verify_images: Check if image files exist
        use_jsonl: Use JSONL format (one JSON per line) for large files
    """
    print(f"Loading CSV from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Determine image base directory
    if image_base_dir is None:
        # Assume images are relative to data/hf_chess_puzzles/
        csv_path = Path(input_csv)
        image_base_dir = csv_path.parent
    else:
        image_base_dir = Path(image_base_dir)
    
    # Limit samples if specified
    if max_samples:
        df = df.head(max_samples)
        print(f"Processing first {max_samples} samples...")
    
    print(f"Processing {len(df)} samples...")
    
    # Convert to Path and check format
    output_path = Path(output_path)
    output_path_str = str(output_path)
    
    # Auto-detect: use JSONL for large datasets (only if not already .jsonl)
    is_jsonl = output_path_str.endswith('.jsonl')
    if len(df) > 10000 and use_jsonl and not is_jsonl:
        output_path_str = output_path_str.replace('.json', '.jsonl')
        output_path = Path(output_path_str)
        is_jsonl = True
        print(f"Using JSONL format for large dataset: {output_path_str}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    
    # Write incrementally for large files
    with open(output_path, 'w', encoding='utf-8') as f:
        if not is_jsonl:
            f.write('[\n')
        
        first = True
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
            result = process_single_row(row, image_base_dir)
            
            if result:
                # Optionally verify image exists
                if verify_images:
                    img_path = Path(result['image_path'])
                    if not img_path.exists():
                        failed += 1
                        continue
                
                if is_jsonl:
                    # JSONL: one JSON object per line
                    f.write(json.dumps(result) + '\n')
                else:
                    # Regular JSON array
                    if not first:
                        f.write(',\n')
                    f.write(json.dumps(result))
                    first = False
                
                successful += 1
            else:
                failed += 1
        
        if not is_jsonl:
            f.write('\n]')
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Dataset creation complete!")
    print(f"  Total processed: {len(df)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {successful/len(df)*100:.2f}%")
    print(f"  Output: {output_path}")
    print(f"{'='*50}")
    
    return {
        'total': len(df),
        'successful': successful,
        'failed': failed,
        'output_path': str(output_path)
    }


def create_all_splits(
    data_dir: str,
    output_dir: str,
    max_samples: Optional[int] = None
):
    """
    Create JSON datasets for all splits (train, val, test).
    
    Args:
        data_dir: Directory containing train.csv, validation.csv, test.csv
        output_dir: Directory to save JSON files
        max_samples: Maximum samples per split (for testing)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = [
        ('test.csv', 'test.json'),
        ('validation.csv', 'val.json'),
        ('train.csv', 'train.json')
    ]
    
    results = {}
    
    for csv_name, json_name in splits:
        csv_path = data_dir / csv_name
        if not csv_path.exists():
            print(f"Skipping {csv_name} (not found)")
            continue
        
        output_path = output_dir / json_name
        print(f"\n{'='*50}")
        print(f"Processing {csv_name}...")
        print(f"{'='*50}")
        
        result = create_json_dataset(
            str(csv_path),
            str(output_path),
            image_base_dir=str(data_dir),
            max_samples=max_samples
        )
        results[csv_name] = result
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Create JSON dataset from CSV')
    parser.add_argument('--input_csv', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to output JSON file')
    parser.add_argument('--image_base_dir', type=str, default=None,
                       help='Base directory for image paths')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to process (for testing)')
    parser.add_argument('--verify_images', action='store_true',
                       help='Verify image files exist')
    
    args = parser.parse_args()
    
    create_json_dataset(
        args.input_csv,
        args.output_path,
        args.image_base_dir,
        args.max_samples,
        args.verify_images
    )


if __name__ == '__main__':
    main()

