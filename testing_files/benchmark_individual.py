# benchmark_individual.py
import torch
import clip
from torch.utils.data import DataLoader
import logging
from dataset_loader import ChessDataset # Assumes dataset_loader.py is present
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_accuracy(logits):
    """Calculates top-1 and top-5 accuracy."""
    preds = torch.argmax(logits, dim=1)
    top5_preds = torch.topk(logits, 5, dim=1).indices
    
    ground_truth = torch.arange(len(logits), dtype=torch.long, device=logits.device)
    
    top1_correct = (preds == ground_truth).sum().item()
    top5_correct = (top5_preds == ground_truth.view(-1, 1)).sum().item()
    
    return top1_correct, top5_correct

def evaluate_model(model_path: str, data_dir: str, batch_size: int = 64):
    """Evaluates a fine-tuned CLIP model and returns accuracy metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    try:
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"Successfully loaded model weights from: {model_path}")
    except Exception as e:
        logging.error(f"FATAL: Could not load model from '{model_path}'. Error: {e}")
        return None

    dataset = ChessDataset(data_dir, preprocess)
    if len(dataset) == 0:
        logging.error(f"No data found in test directory: {data_dir}.")
        return None
    
    logging.info(f"Dataset loaded: {len(dataset)} examples")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model.eval()
    all_image_features, all_text_features = [], []

    with torch.no_grad():
        for images, texts in tqdm(loader, desc="Encoding test data"):
            image_features = model.encode_image(images.to(device))
            text_features = model.encode_text(texts.to(device))
            all_image_features.append(image_features)
            all_text_features.append(text_features)

    image_features = torch.cat(all_image_features)
    text_features = torch.cat(all_text_features)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Image-to-Text retrieval accuracy
    logits_per_image = image_features @ text_features.T
    i2t_top1, i2t_top5 = calculate_accuracy(logits_per_image)
    
    # Text-to-Image retrieval accuracy
    logits_per_text = text_features @ image_features.T
    t2i_top1, t2i_top5 = calculate_accuracy(logits_per_text)
    
    total_samples = len(dataset)
    metrics = {
        "Image-to-Text Top-1 Acc (%)": (i2t_top1 / total_samples) * 100,
        "Image-to-Text Top-5 Acc (%)": (i2t_top5 / total_samples) * 100,
        "Text-to-Image Top-1 Acc (%)": (t2i_top1 / total_samples) * 100,
        "Text-to-Image Top-5 Acc (%)": (t2i_top5 / total_samples) * 100,
        "Total Samples": total_samples
    }
    
    # Print individual results
    print(f"\n📊 Evaluation Results for {model_path}")
    print("="*60)
    for metric, value in metrics.items():
        if "Samples" in metric:
            print(f"{metric}: {value}")
        else:
            print(f"{metric}: {value:.2f}%")
    print("="*60)
    
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned CLIP model for chess.")
    parser.add_argument("model_path", help="Path to the trained model checkpoint.")
    parser.add_argument("data_dir", help="Path to the test dataset directory.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation.")
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.data_dir, args.batch_size)
