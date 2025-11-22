import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import clip
from dataset_loader import ChessDataset
import argparse
from pathlib import Path
import logging
from tqdm import tqdm

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_clip_model(data_dir: str, save_path: str, epochs: int, batch_size: int, lr: float, split_ratio: float):
    """
    Trains a CLIP model on the chess dataset.

    Args:
        data_dir (str): Directory of the dataset.
        save_path (str): Directory to save model checkpoints.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        split_ratio (float): Ratio of training data to total data.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Ensure save directory exists
    Path(save_path).mkdir(parents=True, exist_ok=True)

    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    dataset = ChessDataset(data_dir, preprocess)

    # Splitting the dataset
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.98), eps=1e-6, weight_decay=0.2)
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, texts in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            texts = texts.to(device)

            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

            loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        logging.info(f"Epoch [{epoch+1}/{epochs}] Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, texts in tqdm(val_loader, desc="Validating"):
                images = images.to(device)
                texts = texts.to(device)
                logits_per_image, logits_per_text = model(images, texts)
                ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
                loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        logging.info(f"Epoch [{epoch+1}/{epochs}] Validation Loss: {avg_val_loss:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), f"{save_path}/clip_chess_epoch_{epoch+1}.pt")

    logging.info("✅ Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CLIP model on chess data.")
    parser.add_argument("data_dir", type=str, help="Directory of the dataset.")
    parser.add_argument("save_path", type=str, help="Directory to save model checkpoints.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Training/validation split ratio.")
    args = parser.parse_args()

    train_clip_model(args.data_dir, args.save_path, args.epochs, args.batch_size, args.lr, args.split_ratio)
