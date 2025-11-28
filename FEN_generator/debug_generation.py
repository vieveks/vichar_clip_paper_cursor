"""
Debug script to see what tokens are actually being generated.
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import ChessFENGenerator
from tokenizer import FENTokenizer
from dataset import create_fen_dataloaders, collapse_fen
from train_clip_hf_dataset import create_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = FENTokenizer()
transform = create_transforms()
_, _, test_loader = create_fen_dataloaders("data/hf_chess_puzzles", tokenizer, transform, batch_size=1)

# Load model
model = ChessFENGenerator(vocab_size=len(tokenizer)).to(device)
checkpoint = torch.load("runs/fen_generator_v4_spatial_fix/best_model_stage1.pt", map_location=device)
model.load_state_dict(checkpoint, strict=True)
model.eval()

# Test on one sample
images, tgt_tokens = next(iter(test_loader))
images = images.to(device)

print("="*60)
print("DEBUGGING GENERATION")
print("="*60)

# Get ground truth
gt_ids = tgt_tokens[0].tolist()
gt_expanded = tokenizer.decode(gt_ids)
gt_standard = collapse_fen(gt_expanded)

print(f"\nGround Truth (expanded): {gt_expanded[:100]}...")
print(f"Ground Truth (standard): {gt_standard}")
print(f"GT token IDs (first 20): {gt_ids[:20]}")
print(f"GT token IDs (last 20): {gt_ids[-20:]}")

# Generate
print("\nGenerating...")
with torch.no_grad():
    generated_tokens = model.generate(images, tokenizer, device=device, max_len=80, min_length=70)

gen_ids = generated_tokens[0].tolist()
gen_expanded = tokenizer.decode(gen_ids)
gen_standard = collapse_fen(gen_expanded)

print(f"\nGenerated token IDs (first 30): {gen_ids[:30]}")
print(f"Generated token IDs (last 30): {gen_ids[-30:]}")
print(f"\nGenerated (expanded): {gen_expanded[:100]}...")
print(f"Generated (standard): {gen_standard}")

# Check what tokens are being generated
print(f"\nToken frequency:")
token_counts = {}
for tid in gen_ids:
    if tid == tokenizer.pad_token_id:
        continue
    token = tokenizer.id_to_token.get(tid, f"UNK_{tid}")
    token_counts[token] = token_counts.get(token, 0) + 1

for token, count in sorted(token_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  {token}: {count}")

# Check if it's hitting max_len
print(f"\nSequence length: {len([t for t in gen_ids if t != tokenizer.pad_token_id])}")
print(f"Max length: 80")
print(f"Contains EOS: {tokenizer.eos_token_id in gen_ids}")

