"""
Contains utility functions like saving a PyTorch model.
"""

import torch
from pathlib import Path

def save_model(model: torch.nn.Module, dir_path: str) -> None:
    """
    Saves a PyTorch model's state_dict to a target directory.

    Args:
        model (torch.nn.Module): A target PyTorch model to save.
        dir_path (str): A directory for saving the model to.
    """
    target_dir_path = Path(dir_path)

    # Create directory if it doesn't exist
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_save_path = target_dir_path / "model.pth"
    torch.save(obj=model.state_dict(), f=model_save_path)

    print(f"[INFO] Model saved to: {model_save_path}")
