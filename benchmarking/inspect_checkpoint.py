import torch
import sys

path = "../Improved_representations/checkpoints/exp1a_base_frozen/best_model.pt"
try:
    checkpoint = torch.load(path, map_location="cpu")
    print("Args:", checkpoint.get('args', {}))
    if 'model_state_dict' in checkpoint:
        keys = list(checkpoint['model_state_dict'].keys())
        print("First few keys:", keys[:5])
        # Check for visual.proj or similar to guess dim
        for k in keys:
            if 'visual.proj' in k or 'visual_encoder.proj' in k:
                print(f"{k} shape: {checkpoint['model_state_dict'][k].shape}")
except Exception as e:
    print(e)
