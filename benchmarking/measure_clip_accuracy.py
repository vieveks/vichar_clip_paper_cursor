
import argparse
import pandas as pd
import os
import torch
from tqdm import tqdm
from clip_fen_extractor import CLIPFENExtractor
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Measure CLIP FEN Retrieval Accuracy")
    parser.add_argument("--clip_checkpoint", type=str, required=True, help="Path to CLIP checkpoint")
    parser.add_argument("--dataset_csv", type=str, required=True, help="Path to dataset CSV (test.csv)")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to test")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Initialize Extractor
    print(f"Loading CLIP model from {args.clip_checkpoint}...")
    extractor = CLIPFENExtractor(args.clip_checkpoint, device=args.device)
    
    # Load Dataset
    print(f"Loading dataset from {args.dataset_csv}...")
    df = pd.read_csv(args.dataset_csv)
    
    # Filter for existing images
    valid_samples = []
    print("Validating image paths...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Handle different path formats
        img_name = os.path.basename(row['image_path'])
        full_path = os.path.join(args.images_dir, img_name)
        
        if os.path.exists(full_path):
            valid_samples.append({
                'image_path': full_path,
                'fen': row['fen']
            })
            
        if len(valid_samples) >= args.num_samples:
            break
            
    print(f"Testing on {len(valid_samples)} samples...")
    
    # Pre-compute candidate embeddings
    print("Pre-computing candidate embeddings...")
    candidates_df = pd.read_csv(args.dataset_csv)
    candidates = candidates_df['fen'].dropna().unique().tolist()
    
    # Batch encode candidates to avoid OOM
    batch_size = 100
    candidate_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(candidates), batch_size)):
            batch = candidates[i:i+batch_size]
            text_tokens = extractor.tokenizer(batch).to(extractor.device)
            text_features = extractor.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            candidate_embeddings.append(text_features)
            
    candidate_embeddings = torch.cat(candidate_embeddings)
    print(f"Computed embeddings for {len(candidates)} candidates.")
    
    # Run Evaluation
    correct = 0
    total = 0
    
    print(f"Testing on {len(valid_samples)} samples...")
    
    for sample in tqdm(valid_samples):
        image_path = sample['image_path']
        gt_fen = sample['fen']
        
        try:
            # Load image
            # Note: CLIPFENExtractor doesn't expose _load_image publicly, so we use PIL directly
            image = Image.open(image_path).convert("RGB")
            image_tensor = extractor.transform(image).unsqueeze(0).to(extractor.device)
            
            with torch.no_grad():
                image_features = extractor.model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity
                similarity = (100.0 * image_features @ candidate_embeddings.T).softmax(dim=-1)
                
                # Get top-1
                values, indices = torch.topk(similarity.squeeze(0), k=1)
                predicted_fen = candidates[indices[0].item()]
            
            if predicted_fen == gt_fen:
                correct += 1
            total += 1
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nResults:")
    print(f"Total Samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
