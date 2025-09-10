import torch
import torch.nn as nn
from pathlib import Path

# Import C3D model from the c3d-pytorch submodule
# Adjust the import path according to your package structure
try:
    from c3d_model import C3D
except ImportError:
    # Fallback: add the path to sys.path if running locally
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1] / "action-recognition" / "c3d-pytorch"))
    from c3d_model import C3D

# Number of target classes for rodeo behavior classification
NUM_CLASSES = 3


def load_pretrained_model():
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


def main():
    model = load_pretrained_model()
    # TODO: implement data loading, training loop, and evaluation for action recognition
    # This is a skeleton script. Extend it according to your project requirements.
    print("Loaded C3D model:", model)


if __name__ == "__main__":
    main()
