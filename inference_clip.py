import argparse
import os
import torch
import pandas as pd
import open_clip
from PIL import Image
from torchvision import transforms

# --- New dependencies for generating sample data ---
try:
    import chess
    import chess.svg
    import cairosvg
except ImportError:
    print("Dependencies for generating sample data not found.")
    print("Please run: pip install 'chess<2.0' cairosvg Pillow")
    exit()

# -----------------
# Helper Functions for Creating Sample Data
# -----------------

def create_candidates_csv(fen_list, filename="candidates.csv"):
    """Creates a CSV file with a list of FEN candidates."""
    df = pd.DataFrame(fen_list, columns=['fen'])
    df.to_csv(filename, index=False)
    print(f"✅ Generated sample candidates file: {filename}")
    return filename

def create_chessboard_image(fen_string, filename="sample_chess_board.png"):
    """Creates a PNG image of a chessboard from a FEN string."""
    board = chess.Board(fen_string)
    # Generate SVG of the board
    svg_data = chess.svg.board(board=board, size=350)
    # Convert SVG to PNG
    cairosvg.svg2png(bytestring=svg_data, write_to=filename)
    print(f"✅ Generated sample chessboard image: {filename}")
    return filename

# -----------------
# Inference Functions
# -----------------

def get_image_transform():
    """Returns the same image transformations used during training."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])

def load_model_from_checkpoint(model_name, checkpoint_path, device):
    """Loads the OpenCLIP model and tokenizer, then applies the saved checkpoint weights."""
    model, _, _ = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained="laion2B-s34B-b79K",
        device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)
    
    print(f"✅ Model loaded and weights restored from epoch {checkpoint.get('epoch', 'N/A')} in {checkpoint_path}")
    return model, tokenizer

def find_best_fen_match(model, tokenizer, image_path, fen_candidates, device, top_k=5):
    """Encodes an image and a list of FEN candidates, then computes similarity."""
    transform = get_image_transform()
    
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return

    if not fen_candidates:
        print("Error: The list of FEN candidates is empty.")
        return
        
    text_tokens = tokenizer(fen_candidates).to(device)
    
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast() if "cuda" in device else torch.autocast("cpu"):
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
    k = min(top_k, len(fen_candidates))
    values, indices = torch.topk(similarity.squeeze(0), k=k)
    
    print(f"\n--- Top {k} Predictions ---")
    for i, (value, index) in enumerate(zip(values, indices)):
        print(f"{i+1}. FEN: '{fen_candidates[index]}'")
        print(f"   Confidence: {value.item():.2%}\n")

# -----------------
# Main Inference Code
# -----------------
def main():
    parser = argparse.ArgumentParser(description="Inference script for Chess CLIP model. Can generate its own sample data.")
    # This argument is now the only one that's truly required
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint (e.g., 'best_model.pt').")
    # These are now optional; the script will generate them if not provided
    parser.add_argument("--image_path", type=str, default=None, help="Path to an input chessboard image. If not provided, a sample will be generated.")
    parser.add_argument("--candidates_csv", type=str, default=None, help="Path to a CSV of FEN candidates. If not provided, a sample will be generated.")
    
    parser.add_argument("--model_name", type=str, default="ViT-B-32", help="Name of the CLIP model architecture used during training.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions to display.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    image_path = args.image_path
    candidates_csv = args.candidates_csv
    
    # --- Generate sample data if paths are not provided ---
    if image_path is None or candidates_csv is None:
        print("\n--- Generating sample data for demonstration ---")
        # A small list of FENs for our sample CSV
        sample_fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", # Start
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", # e4
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", # e4 c5 (Sicilian)
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"  # e4 e5 (Correct one)
        ]
        # The FEN we will use to create the image
        correct_fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        
        # Override paths with our generated files
        candidates_csv = create_candidates_csv(sample_fens, "candidates.csv")
        image_path = create_chessboard_image(correct_fen, "sample_chess_board.png")
        print(f"The correct FEN for the generated image is: '{correct_fen}'")

    # -----------------
    # Load Model
    # -----------------
    try:
        model, tokenizer = load_model_from_checkpoint(args.model_name, args.checkpoint_path, device)
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading model: {e}")
        return
    
    # -----------------
    # Load FEN Candidates
    # -----------------
    try:
        df = pd.read_csv(candidates_csv)
        fen_candidates = df['fen'].dropna().tolist()
        print(f"\nLoaded {len(fen_candidates)} FEN candidates from {candidates_csv}")
    except Exception as e:
        print(f"Error reading candidates CSV: {e}")
        return

    # -----------------
    # Perform Inference
    # -----------------
    find_best_fen_match(model, tokenizer, image_path, fen_candidates, device, args.top_k)


if __name__ == "__main__":
    main()