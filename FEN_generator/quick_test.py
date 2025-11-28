"""
Quick test script to evaluate on just a few examples.
Usage: python FEN_generator/quick_test.py --checkpoint_path <path> --num_samples 10
"""
import argparse
import torch
import logging
import sys
import os
from tqdm import tqdm
import Levenshtein

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import ChessFENGenerator
from tokenizer import FENTokenizer
from dataset import create_fen_dataloaders, collapse_fen
from train_clip_hf_dataset import create_transforms

def calculate_cer(reference, hypothesis):
    """Calculate Character Error Rate (CER)."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    dist = Levenshtein.distance(reference, hypothesis)
    return dist / len(reference)

def main():
    parser = argparse.ArgumentParser(description="Quick Test FEN Generator")
    parser.add_argument("--data_dir", type=str, default="data/hf_chess_puzzles", help="Path to dataset directory")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    # Tokenizer
    tokenizer = FENTokenizer()

    # Data
    transform = create_transforms()
    _, _, test_loader = create_fen_dataloaders(
        args.data_dir, tokenizer, transform, batch_size=args.batch_size
    )

    # Model
    logging.info("Initializing model...")
    model = ChessFENGenerator(vocab_size=len(tokenizer)).to(device)

    # Load Checkpoint
    logging.info(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    state_dict = checkpoint
    msg = model.load_state_dict(state_dict, strict=True)
    logging.info(f"Loaded weights: {msg}")

    model.eval()

    total_samples = 0
    exact_matches = 0
    total_cer = 0.0
    total_length = 0
    total_gt_length = 0
    
    results = []

    logging.info(f"Testing on {args.num_samples} samples...")
    with torch.no_grad():
        for images, tgt_tokens in test_loader:
            if total_samples >= args.num_samples:
                break
                
            images = images.to(device)
            
            # Generate FENs
            generated_tokens = model.generate(images, tokenizer, device=device, min_length=70)
            
            # Decode
            for i in range(images.size(0)):
                if total_samples >= args.num_samples:
                    break
                    
                # Get ground truth
                gt_ids = tgt_tokens[i].tolist()
                gt_expanded_fen = tokenizer.decode(gt_ids)
                
                # Get generated
                gen_ids = generated_tokens[i].tolist()
                gen_expanded_fen = tokenizer.decode(gen_ids)
                
                # Collapse both to standard FEN
                gt_fen = collapse_fen(gt_expanded_fen)
                gen_fen = collapse_fen(gen_expanded_fen)
                
                # Calculate metrics
                gen_length = len(gen_expanded_fen.replace('<SOS>', '').replace('<EOS>', '').replace('<PAD>', ''))
                gt_length = len(gt_expanded_fen.replace('<SOS>', '').replace('<EOS>', '').replace('<PAD>', ''))
                
                match = gen_fen == gt_fen
                if match:
                    exact_matches += 1
                
                cer = calculate_cer(gt_fen, gen_fen)
                total_cer += cer
                total_length += gen_length
                total_gt_length += gt_length
                total_samples += 1
                
                # Save example
                results.append({
                    'gt_expanded': gt_expanded_fen,
                    'gt_standard': gt_fen,
                    'gen_expanded': gen_expanded_fen,
                    'gen_standard': gen_fen,
                    'match': match,
                    'cer': cer,
                    'gen_length': gen_length,
                    'gt_length': gt_length
                })
                
                # Print immediately
                print(f"\n=== Sample {total_samples} ===")
                print(f"GT (standard):  {gt_fen}")
                print(f"GEN (standard): {gen_fen}")
                print(f"Match: {match}, CER: {cer:.4f}")
                print(f"Length: GT={gt_length}, GEN={gen_length}")

    accuracy = exact_matches / total_samples if total_samples > 0 else 0
    avg_cer = total_cer / total_samples if total_samples > 0 else 0
    avg_gen_length = total_length / total_samples if total_samples > 0 else 0
    avg_gt_length = total_gt_length / total_samples if total_samples > 0 else 0

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Samples: {total_samples}")
    print(f"Exact Match Accuracy: {accuracy:.4f} ({exact_matches}/{total_samples})")
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average Generated Length: {avg_gen_length:.1f} tokens")
    print(f"Average GT Length: {avg_gt_length:.1f} tokens")
    print(f"Length Ratio: {avg_gen_length/avg_gt_length:.2f}" if avg_gt_length > 0 else "N/A")
    print("="*60)

if __name__ == "__main__":
    main()

