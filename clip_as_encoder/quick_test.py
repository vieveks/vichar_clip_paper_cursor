"""
Quick test script to verify the setup works.
Tests loading both baseline and chess-CLIP models.
"""

import argparse
import torch
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from clip_as_encoder.model import ChessLLaVA
from PIL import Image
from torchvision import transforms


def test_model_loading(vision_type, chess_checkpoint=None, device="cuda"):
    """Test loading a model."""
    print(f"\n{'='*60}")
    print(f"Testing {vision_type} model loading...")
    print(f"{'='*60}")
    
    try:
        model = ChessLLaVA(
            vision_encoder_type=vision_type,
            chess_clip_checkpoint=chess_checkpoint,
            language_model_name="llava-hf/llava-v1.6-mistral-7b-hf",
            device=device
        )
        print(f"✅ Successfully loaded {vision_type} model")
        
        # Test image encoding
        print("\nTesting image encoding...")
        dummy_image = torch.randn(1, 3, 224, 224).to(device)
        features = model.encode_image(dummy_image)
        print(f"✅ Image encoding works: output shape {features.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading {vision_type} model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Quick test of ChessLLaVA models")
    parser.add_argument(
        "--chess_clip_checkpoint",
        type=str,
        default="runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt",
        help="Path to chess-finetuned CLIP checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test baseline
    baseline_ok = test_model_loading("generic", device=device)
    
    # Test chess-CLIP
    chess_ok = test_model_loading(
        "chess_finetuned",
        chess_checkpoint=args.chess_clip_checkpoint,
        device=device
    )
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline model: {'✅ OK' if baseline_ok else '❌ FAILED'}")
    print(f"Chess-CLIP model: {'✅ OK' if chess_ok else '❌ FAILED'}")
    
    if baseline_ok and chess_ok:
        print("\n✅ All tests passed! Ready to run experiments.")
    else:
        print("\n❌ Some tests failed. Check errors above.")


if __name__ == "__main__":
    main()

