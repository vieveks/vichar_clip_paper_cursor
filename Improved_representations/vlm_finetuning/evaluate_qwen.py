"""
Evaluation script for fine-tuned Qwen2-VL-2B model.

Evaluates JSON prediction accuracy and compares with baseline models.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Improved_representations.vlm_finetuning.dataset import QwenVLDataset
from Improved_representations.data_processing.converters import json_to_fen, validate_json_position


def load_model(model_path: str, use_lora: bool = True):
    """Load fine-tuned model."""
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    processor = Qwen2VLProcessor.from_pretrained(
        model_path,
        token=hf_token
    )
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        token=hf_token
    )
    
    if use_lora:
        from peft import PeftModel
        # If model was saved with LoRA, load base model and LoRA weights
        try:
            model = PeftModel.from_pretrained(model, model_path)
        except:
            pass  # Model might be merged already
    
    model.eval()
    return model, processor


def parse_json_response(response: str) -> Dict:
    """Parse JSON from model response."""
    # Try to extract JSON from response
    response = response.strip()
    
    # Remove markdown code blocks if present
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON object in response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            try:
                return json.loads(response[start_idx:end_idx])
            except:
                pass
        return None


def evaluate_model(
    model,
    processor,
    test_data: List[Dict],
    image_base_dir: Path,
    device: str = "cuda"
):
    """Evaluate model on test dataset."""
    model.eval()
    
    results = {
        'total': 0,
        'valid_json': 0,
        'valid_position': 0,
        'exact_json_match': 0,
        'fen_accuracy': 0,
        'per_square_accuracy': 0.0,
        'exact_board_match': 0,
        'errors': []
    }
    
    print("Evaluating model...")
    with torch.no_grad():
        for item in tqdm(test_data, desc="Evaluating"):
            try:
                # Load image
                image_path = item['image']
                if not Path(image_path).is_absolute():
                    full_image_path = image_base_dir / image_path
                else:
                    full_image_path = Path(image_path)
                
                if not full_image_path.exists():
                    # Try alternative paths
                    alt_paths = [
                        image_base_dir.parent / 'data' / 'hf_chess_puzzles' / image_path,
                        Path(image_path),
                    ]
                    for alt_path in alt_paths:
                        if alt_path.exists():
                            full_image_path = alt_path
                            break
                    else:
                        results['errors'].append(f"Image not found: {image_path}")
                        continue
                
                image = Image.open(full_image_path).convert('RGB')
                
                # Get ground truth from conversations
                conversations = item['conversations']
                assistant_message = conversations[1]['value']  # JSON response
                gt_json = parse_json_response(assistant_message)
                if not gt_json:
                    results['errors'].append("Could not parse ground truth JSON")
                    continue
                
                # Prepare input messages (same format as training)
                instruction = "Analyze this chess board image and describe the position in JSON format."
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": instruction}
                        ]
                    }
                ]
                
                # Process messages (same as dataset.py)
                try:
                    inputs = processor(
                        messages,
                        padding=True,
                        truncation=True,
                        max_length=2048,
                        return_tensors="pt"
                    )
                except Exception as e:
                    # Fallback: process text and images separately
                    text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    inputs = processor(
                        text=[text],
                        images=[image],
                        padding=True,
                        truncation=True,
                        max_length=2048,
                        return_tensors="pt"
                    )
                
                # Move to device
                inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                
                # Ensure image_grid_thw is present
                if 'image_grid_thw' not in inputs or inputs.get('image_grid_thw') is None:
                    if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                        _, _, h, w = inputs['pixel_values'].shape
                        patch_size = 14  # Qwen2-VL uses 14x14 patches
                        grid_h = (h + patch_size - 1) // patch_size
                        grid_w = (w + patch_size - 1) // patch_size
                        inputs['image_grid_thw'] = torch.tensor([[1, grid_h, grid_w]], device=device)
                    else:
                        inputs['image_grid_thw'] = torch.tensor([[1, 32, 32]], device=device)
                
                # Generate
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False
                )
                
                # Extract only the generated tokens (remove input tokens)
                input_length = inputs['input_ids'].shape[1]
                generated_ids_trimmed = generated_ids[:, input_length:]
                
                response = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                # Parse response
                pred_json = parse_json_response(response)
                
                results['total'] += 1
                
                # Check valid JSON
                if pred_json:
                    results['valid_json'] += 1
                    
                    # Calculate per-square accuracy (for all valid JSON)
                    if 'pieces' in pred_json and 'pieces' in gt_json:
                            try:
                                pred_pieces = {p.get('square', ''): p for p in pred_json.get('pieces', []) if 'square' in p}
                                gt_pieces = {p.get('square', ''): p for p in gt_json.get('pieces', []) if 'square' in p}
                                
                                # Consider all 64 squares
                                all_squares = set([f"{file}{rank}" for file in "abcdefgh" for rank in "12345678"])
                                correct = 0
                                total_squares = 64
                                
                                for square in all_squares:
                                    pred_piece = pred_pieces.get(square, {}).get('piece', 'empty')
                                    gt_piece = gt_pieces.get(square, {}).get('piece', 'empty')
                                    if pred_piece == gt_piece:
                                        correct += 1
                                
                                if total_squares > 0:
                                    square_acc = correct / total_squares
                                    results['per_square_accuracy'] += square_acc
                                    
                                    # Exact board match
                                    if correct == total_squares:
                                        results['exact_board_match'] += 1
                            except Exception as e:
                                results['errors'].append(f"Per-square calc error: {str(e)}")
                    
                    # Check valid position
                    if validate_json_position(pred_json):
                        results['valid_position'] += 1
                        
                        # Check exact JSON match
                        if pred_json == gt_json:
                            results['exact_json_match'] += 1
                        
                        # Check FEN accuracy
                        try:
                            pred_fen = json_to_fen(pred_json)
                            gt_fen = json_to_fen(gt_json)
                            if pred_fen == gt_fen:
                                results['fen_accuracy'] += 1
                        except:
                            pass
                
            except Exception as e:
                results['errors'].append(str(e))
                continue
    
    # Calculate averages
    if results['total'] > 0:
        results['valid_json_rate'] = results['valid_json'] / results['total']
        results['valid_position_rate'] = results['valid_position'] / results['total']
        results['exact_json_match_rate'] = results['exact_json_match'] / results['total']
        results['fen_accuracy_rate'] = results['fen_accuracy'] / results['total']
        # Per-square accuracy: average over samples with valid JSON and pieces
        if results['valid_json'] > 0:
            results['avg_per_square_accuracy'] = results['per_square_accuracy'] / results['valid_json']
        else:
            results['avg_per_square_accuracy'] = 0.0
        results['exact_board_match_rate'] = results['exact_board_match'] / results['total']
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate fine-tuned Qwen2-VL-2B model'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to fine-tuned model'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        default='Improved_representations/data/vlm_dataset/test.json',
        help='Path to test dataset'
    )
    parser.add_argument(
        '--image_base_dir',
        type=str,
        default='data/hf_chess_puzzles',
        help='Base directory for images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='Improved_representations/results/qwen2vl_eval.json',
        help='Output path for results'
    )
    parser.add_argument(
        '--use_lora',
        action='store_true',
        default=True,
        help='Model uses LoRA'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum samples to evaluate'
    )
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, processor = load_model(args.model_path, use_lora=args.use_lora)
    
    # Load test data directly
    print(f"Loading test data from {args.test_data}...")
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    
    print(f"Evaluating on {len(test_data)} samples")
    
    image_base_dir = Path(args.image_base_dir)
    
    # Evaluate
    results = evaluate_model(model, processor, test_data, image_base_dir, device)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total samples: {results['total']}")
    print(f"Valid JSON rate: {results.get('valid_json_rate', 0):.4f} ({results['valid_json']}/{results['total']})")
    print(f"Valid position rate: {results.get('valid_position_rate', 0):.4f} ({results['valid_position']}/{results['total']})")
    print(f"Exact JSON match rate: {results.get('exact_json_match_rate', 0):.4f} ({results['exact_json_match']}/{results['total']})")
    print(f"FEN accuracy rate: {results.get('fen_accuracy_rate', 0):.4f} ({results['fen_accuracy']}/{results['total']})")
    print(f"Average per-square accuracy: {results.get('avg_per_square_accuracy', 0):.4f}")
    print(f"Exact board match rate: {results.get('exact_board_match_rate', 0):.4f} ({results['exact_board_match']}/{results['total']})")
    print("="*60)
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

