"""
Create VLM fine-tuning dataset from JSON dataset.

Converts JSON dataset to instruction-following format for Qwen2-VL-2B fine-tuning.
Format: Image + instruction -> JSON response
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List


def create_instruction_prompt() -> str:
    """Create the instruction prompt for JSON prediction."""
    return """Analyze this chess board image and describe the position in JSON format. 

The JSON should include:
1. A list of all pieces with their squares, colors, types, and values
2. Metadata including side to move, castling rights, and material balance

Output only valid JSON, no additional text."""


def format_json_response(json_repr: Dict) -> str:
    """Format JSON representation as a string response (simplified for VLM training)."""
    # Simplify JSON - only include pieces and essential metadata
    # Remove complex relationships to make training easier
    simplified = {
        'pieces': json_repr.get('pieces', []),
        'metadata': {
            'to_move': json_repr.get('metadata', {}).get('to_move', 'white'),
            'castling_rights': json_repr.get('metadata', {}).get('castling_rights', {'white': [], 'black': []}),
            'en_passant': json_repr.get('metadata', {}).get('en_passant', None),
            'material': json_repr.get('metadata', {}).get('material', {'white': 0, 'black': 0}),
            'material_balance': json_repr.get('metadata', {}).get('material_balance', 0)
        }
    }
    return json.dumps(simplified, indent=2)


def create_vlm_dataset_entry(
    image_path: str,
    json_repr: Dict,
    base_dir: Path
) -> Dict:
    """
    Create a single VLM dataset entry.
    
    Format for Qwen2-VL:
    {
        "id": "unique_id",
        "image": "path/to/image",
        "conversations": [
            {
                "from": "user",
                "value": "<image>\n{instruction}"
            },
            {
                "from": "assistant", 
                "value": "{json_response}"
            }
        ]
    }
    """
    # Make image path relative to base_dir if needed
    if not Path(image_path).is_absolute():
        rel_image_path = image_path
    else:
        try:
            rel_image_path = str(Path(image_path).relative_to(base_dir))
        except ValueError:
            rel_image_path = image_path
    
    instruction = create_instruction_prompt()
    json_response = format_json_response(json_repr)
    
    return {
        "id": f"chess_{Path(image_path).stem}",
        "image": rel_image_path,
        "conversations": [
            {
                "from": "user",
                "value": f"<image>\n{instruction}"
            },
            {
                "from": "assistant",
                "value": json_response
            }
        ]
    }


def process_jsonl_file(
    input_path: Path,
    output_path: Path,
    base_dir: Path,
    max_samples: int = None
) -> int:
    """
    Process a JSONL file and create VLM dataset.
    
    Returns:
        Number of samples processed
    """
    vlm_data = []
    processed = 0
    
    print(f"Processing {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Converting to VLM format"):
            if max_samples and processed >= max_samples:
                break
                
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                image_path = data.get('image_path')
                json_repr = data.get('json_repr')
                
                if not image_path or not json_repr:
                    continue
                
                entry = create_vlm_dataset_entry(
                    image_path,
                    json_repr,
                    base_dir
                )
                vlm_data.append(entry)
                processed += 1
                
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
    
    # Save VLM dataset
    print(f"Saving {len(vlm_data)} samples to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vlm_data, f, indent=2, ensure_ascii=False)
    
    return len(vlm_data)


def main():
    parser = argparse.ArgumentParser(
        description='Create VLM fine-tuning dataset from JSON dataset'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing train.jsonl, val.jsonl, test.jsonl'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='Improved_representations/data/vlm_dataset',
        help='Output directory for VLM dataset files'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default='.',
        help='Base directory for resolving image paths'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum samples per split (for testing)'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    base_dir = Path(args.base_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    splits = ['train', 'val', 'test']
    total_samples = 0
    
    for split in splits:
        input_path = data_dir / f'{split}.jsonl'
        if not input_path.exists():
            print(f"Warning: {input_path} not found, skipping...")
            continue
        
        output_path = output_dir / f'{split}.json'
        count = process_jsonl_file(
            input_path,
            output_path,
            base_dir,
            max_samples=args.max_samples
        )
        total_samples += count
        print(f"{split}: {count} samples\n")
    
    print(f"Total samples processed: {total_samples}")
    print(f"VLM dataset saved to: {output_dir}")


if __name__ == '__main__':
    main()

