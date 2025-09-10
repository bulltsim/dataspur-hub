import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

# Import the clip dataset loader
from dataset_frames import ClipDataset

# Import C3D model from the c3d-pytorch submodule
try:
    from c3d_model import C3D
except ImportError:
    import sys
    # Adjust the import path when running from the repository root
    sys.path.append(str(Path(__file__).resolve().parents[1] / "action-recognition" / "c3d-pytorch"))
    from c3d_model import C3D

# Number of behavior classes for rodeo action recognition
NUM_CLASSES = 3

def load_pretrained_model() -> nn.Module:
    """Load the pretrained C3D model and adapt the classifier for our tasks."""
    # Initialize model with original number of classes (Sports-1M has 487 classes)
    model = C3D(num_classes=487)
    # Load pretrained weights from the submodule if available
    weights_path = Path("action-recognition/c3d-pytorch") / "c3d.pickle"
    if weights_path.is_file():
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    # Replace the final fully connected layer to match our NUM_CLASSES
    model.fc8 = nn.Linear(4096, NUM_CLASSES)
    return model

def train(model: nn.Module, dataloader: DataLoader, epochs: int = 10, lr: float = 1e-4, device: str | None = None) -> nn.Module:
    """Simple training loop for C3D action recognition."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for clips, labels in dataloader:
            # clips has shape (batch, clip_len, C, H, W)
            clips = clips.to(device)
            labels = labels.to(device)
            # Permute to (batch, C, clip_len, H, W) as expected by C3D
            clips = clips.permute(0, 2, 1, 3, 4)
            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    return model

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a C3D model on frame clips for rodeo behavior classification")
    parser.add_argument("--data_root", type=str, required=True, help="Path to directory with frame clips organised by class and video")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--clip_len", type=int, default=16, help="Number of frames per clip")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    args = parser.parse_args()

    dataset = ClipDataset(args.data_root, clip_len=args.clip_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    model = load_pretrained_model()
    train(model, dataloader, epochs=args.epochs, lr=args.lr)

if __name__ == "__main__":
    main()
