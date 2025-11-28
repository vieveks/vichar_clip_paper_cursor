import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from FEN_generator.model import ChessFENGenerator
from FEN_generator.tokenizer import FENTokenizer

def test_model():
    print("Creating model...")
    tokenizer = FENTokenizer()
    model = ChessFENGenerator(vocab_size=len(tokenizer))
    model.eval()
    
    # Test forward pass
    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_tgt = torch.randint(0, len(tokenizer), (2, 10))
    
    print("Testing forward...")
    try:
        logits = model(dummy_img, dummy_tgt)
        print(f"Logits shape: {logits.shape}")  # Should be [2, 10, vocab_size]
        print("Forward pass SUCCESS!")
    except Exception as e:
        print(f"Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test generation
    print("\nTesting generation...")
    try:
        generated = model.generate(dummy_img, tokenizer, max_len=20, device='cpu')
        print(f"Generated shape: {generated.shape}")  # Should be [2, <=20]
        print("Generation SUCCESS!")
    except Exception as e:
        print(f"Generation FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
