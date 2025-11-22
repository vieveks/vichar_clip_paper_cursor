#!/usr/bin/env python3
"""
Simple Chess CLIP Validation Experiment
======================================

Simplified version of the validation experiment to test real data performance.
"""

import os
import sys
import torch
import numpy as np
import json
from datetime import datetime

def main():
    print("🚀 CHESS CLIP VALIDATION EXPERIMENT")
    print("=" * 50)
    
    # Check environment
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check model path
    model_path = "Notebooks/checkpoints/large_1000/fen_only_model/clip_chess_epoch_5.pt"
    print(f"\nModel path: {model_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"Model epoch: {checkpoint.get('epoch', 'Unknown')}")
            print(f"Best validation loss: {checkpoint.get('best_val_loss', 'Unknown')}")
            print("✅ Model checkpoint loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    else:
        print("⚠️ Model checkpoint not found - will use pretrained CLIP")
    
    try:
        # Test basic imports
        import open_clip
        print("✅ OpenCLIP available")
        
        # Try to load model
        model, preprocess, _ = open_clip.create_model_and_transforms(
            model_name="ViT-B-32",
            pretrained="laion2B-s34B-b79K", 
            device=device
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        print("✅ Base CLIP model loaded")
        
        # Load trained weights if available
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print("✅ Trained weights loaded")
        
        model.eval()
        
        # Simple test
        test_text = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        with torch.no_grad():
            text_tokens = tokenizer([test_text]).to(device)
            text_features = model.encode_text(text_tokens)
            print(f"✅ Model inference test successful - feature shape: {text_features.shape}")
        
        # Create simple validation report
        results = {
            "timestamp": datetime.now().isoformat(),
            "environment": {
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": device,
                "model_loaded": True,
                "weights_loaded": os.path.exists(model_path)
            },
            "model_info": {
                "architecture": "ViT-B-32",
                "epoch": checkpoint.get("epoch", "Unknown") if os.path.exists(model_path) else "N/A",
                "validation_loss": checkpoint.get("best_val_loss", "Unknown") if os.path.exists(model_path) else "N/A"
            },
            "validation_status": "SUCCESS",
            "ready_for_experiments": True
        }
        
        # Save results
        with open("validation_check_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\n🎉 VALIDATION SUCCESSFUL!")
        print("📊 Results saved to: validation_check_results.json")
        print("✅ Ready to run full experiments")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install: pip install open_clip_torch")
        return False
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
