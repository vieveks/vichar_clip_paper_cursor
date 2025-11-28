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
    """
    Calculate Character Error Rate (CER).
    CER = (Substitutions + Deletions + Insertions) / Length of Reference
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    dist = Levenshtein.distance(reference, hypothesis)
    return dist / len(reference)

def main():
    parser = argparse.ArgumentParser(description="Evaluate FEN Generator")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_file", type=str, default="evaluation_results.txt", help="Output file for results")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    # Tokenizer
    tokenizer = FENTokenizer()

    # Data
    transform = create_transforms()
    # We only need the test loader, but the function returns all three
    _, _, test_loader = create_fen_dataloaders(
        args.data_dir, tokenizer, transform, batch_size=args.batch_size
    )

    # Model
    logging.info("Initializing model...")
    model = ChessFENGenerator(vocab_size=len(tokenizer)).to(device)

    # Load Checkpoint
    logging.info(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Handle state dict keys (remove 'module.' prefix if present, handle 'visual.' prefix)
    state_dict = checkpoint
    # If it's a full checkpoint dict (e.g. from train.py which saves model.state_dict())
    # It should be directly loadable.
    # Note: train.py saves `model.state_dict()`.
    
    msg = model.load_state_dict(state_dict, strict=True)
    logging.info(f"Loaded weights: {msg}")

    model.eval()

    total_samples = 0
    exact_matches = 0
    total_cer = 0.0
    
    results = []

    logging.info("Starting evaluation...")
    with torch.no_grad():
        for images, tgt_tokens in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            
            # Generate FENs
            # generate returns [batch_size, seq_len]
            generated_tokens = model.generate(images, tokenizer, device=device)
            
            # Decode
            for i in range(images.size(0)):
                # Get ground truth string (in expanded FEN format)
                # tgt_tokens includes SOS and EOS. We need to strip them for comparison or let decode handle it.
                # tokenizer.decode handles SOS/EOS skipping/breaking.
                gt_ids = tgt_tokens[i].tolist()
                gt_expanded_fen = tokenizer.decode(gt_ids)
                
                # Get generated string (also in expanded FEN format)
                gen_ids = generated_tokens[i].tolist()
                gen_expanded_fen = tokenizer.decode(gen_ids)
                
                # Collapse both to standard FEN format for comparison
                # This allows us to compare with standard FEN notation
                gt_fen = collapse_fen(gt_expanded_fen)
                gen_fen = collapse_fen(gen_expanded_fen)
                
                # Metrics (comparing standard FEN format)
                if gen_fen == gt_fen:
                    exact_matches += 1
                
                cer = calculate_cer(gt_fen, gen_fen)
                total_cer += cer
                
                total_samples += 1
                
                # Save some examples (show both expanded and collapsed for debugging)
                if total_samples <= 20:
                    results.append(
                        f"GT (expanded):  {gt_expanded_fen}\n"
                        f"GT (standard): {gt_fen}\n"
                        f"GEN (expanded): {gen_expanded_fen}\n"
                        f"GEN (standard): {gen_fen}\n"
                        f"Match: {gen_fen == gt_fen}\n"
                        f"CER: {cer:.4f}\n---"
                    )

    accuracy = exact_matches / total_samples
    avg_cer = total_cer / total_samples

    logging.info(f"Evaluation Complete.")
    logging.info(f"Total Samples: {total_samples}")
    logging.info(f"Exact Match Accuracy: {accuracy:.4f} ({exact_matches}/{total_samples})")
    logging.info(f"Average CER: {avg_cer:.4f}")

    # Write results to file
    with open(args.out_file, "w") as f:
        f.write(f"Evaluation Results\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Exact Match Accuracy: {accuracy:.4f}\n")
        f.write(f"Average CER: {avg_cer:.4f}\n")
        f.write("\n--- Examples ---\n")
        for res in results:
            f.write(res + "\n")
            
    logging.info(f"Results saved to {args.out_file}")

if __name__ == "__main__":
    main()
