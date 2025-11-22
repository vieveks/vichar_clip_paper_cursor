import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
from torch.amp import autocast, GradScaler
from torchvision import transforms
from datasets import load_dataset
import pandas as pd
import open_clip

# -----------------
# Dataset Class
# -----------------
class FenImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        df = pd.read_csv(csv_file)
        self.image_paths = df['image_path'].tolist()
        self.texts = df['fen'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.texts[idx]

# -----------------
# Main Training Code
# -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to FEN–image CSV")
    parser.add_argument("--out_dir", type=str, default="runs/clip_fen")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # -----------------
    # Data Loading
    # -----------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])

    dataset = FenImageDataset(args.csv, transform=transform)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Dataset size: {len(dataset)} (train: {len(train_dataset)}, val: {len(val_dataset)})")

    # -----------------
    # Model & Loss
    # -----------------
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model,
        pretrained="laion2B-s34B-b79K",
        device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    from open_clip.loss import ClipLoss
    loss_fn = ClipLoss()


    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(device if device == "cuda" else "cpu", enabled=args.fp16 and device == "cuda")

    start_epoch = 0
    best_val_loss = float("inf")

    # -----------------
    # Resume from checkpoint
    # -----------------
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        print(f"Resumed from checkpoint at epoch {start_epoch}")

    # -----------------
    # Training Loop
    # -----------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0

        for images, texts in train_loader:
            images = images.to(device)
            texts = tokenizer(texts).to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=args.fp16 and device == "cuda"):
                image_features, text_features, logit_scale = model(images, texts)
                loss = loss_fn(image_features, text_features, logit_scale)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # -----------------
        # Validation Loop
        # -----------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, texts in val_loader:
                images = images.to(device)
                texts = tokenizer(texts).to(device)

                with autocast(device_type="cuda", enabled=args.fp16 and device == "cuda"):
                    image_features, text_features, logit_scale = model(images, texts)
                    loss = loss_fn(image_features, text_features, logit_scale)
                    val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # -----------------
        # Save Best Model
        # -----------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_loss": best_val_loss
            }, os.path.join(args.out_dir, "best_model.pt"))
            print(f"✅ Saved best model at epoch {epoch+1}")

        # Save checkpoint every epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss
        }, os.path.join(args.out_dir, "last_checkpoint.pt"))

if __name__ == "__main__":
    main()
