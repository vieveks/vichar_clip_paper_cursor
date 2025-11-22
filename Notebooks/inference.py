# inference.py
import torch
import clip
from PIL import Image
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def perform_inference(model_path: str, image_path: str, text_candidates_path: str, top_k: int):
    """Loads a model and ranks text candidates for a given image."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logging.info(f"Successfully loaded model from: {model_path}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    if not Path(image_path).exists():
        logging.error(f"Image file not found: {image_path}")
        return
    
    try:
        with open(text_candidates_path, 'r', encoding='utf-8') as f:
            candidates = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(candidates)} text candidates")
    except FileNotFoundError:
        logging.error(f"Text candidates file not found: {text_candidates_path}")
        return

    if len(candidates) == 0:
        logging.error("No valid text candidates found in file")
        return

    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = clip.tokenize(candidates, truncate=True).to(device)
    except Exception as e:
        logging.error(f"Error processing inputs: {e}")
        return

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity scores
        scores = (image_features @ text_features.T)[0]
        top_scores, top_indices = torch.topk(scores, k=min(top_k, len(candidates)))

    print("\n" + "="*80)
    print(f"🔍 Inference Results for Image: {Path(image_path).name}")
    print(f"📁 Model: {Path(model_path).name}")
    print(f"📊 Showing top {len(top_scores)} out of {len(candidates)} candidates")
    print("="*80)
    for i, (score, idx) in enumerate(zip(top_scores, top_indices)):
        confidence = f"{score.item():.4f}"
        print(f"  🥇 Rank {i+1:2d} | Confidence: {confidence} | Text: \"{candidates[idx]}\"")
    print("="*80)

def create_sample_candidates_file(output_path: str):
    """Creates a sample text candidates file for testing."""
    sample_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # After e4
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # After e4 e5
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # After Nc6 Nf3
        "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # Italian Game
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for fen in sample_fens:
            f.write(fen + '\n')
    
    print(f"✅ Created sample candidates file: {output_path}")
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform inference with a fine-tuned chess CLIP model.")
    parser.add_argument("model_path", help="Path to the trained model checkpoint (.pt file).")
    parser.add_argument("image_path", help="Path to the input chessboard image.")
    parser.add_argument("text_candidates_path", nargs='?', help="Path to a .txt file with text candidates (one per line). If not provided, will create a sample file.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to show.")
    parser.add_argument("--create_sample", action='store_true', help="Create a sample candidates file for testing.")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_candidates_file("sample_candidates.txt")
        print("Use this file as text_candidates_path for testing!")
    elif args.text_candidates_path is None:
        print("Creating sample candidates file since none was provided...")
        candidates_file = create_sample_candidates_file("sample_candidates.txt")
        perform_inference(args.model_path, args.image_path, candidates_file, args.top_k)
    else:
        perform_inference(args.model_path, args.image_path, args.text_candidates_path, args.top_k)
