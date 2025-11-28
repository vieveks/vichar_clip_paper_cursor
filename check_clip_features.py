import open_clip

def check_model():
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2B-s34B-b79K')
    visual = model.visual
    print(f"visual.output_dim: {visual.output_dim}") # 512
    if hasattr(visual, 'width'):
        print(f"visual.width: {visual.width}") # Should be 768
    
    # Check if we can get it from config
    
if __name__ == "__main__":
    check_model()
