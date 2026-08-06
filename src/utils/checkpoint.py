from pathlib import Path
from typing import Dict, Any, Union
import torch

from src.utils.logger import get_logger
from src.utils.exceptions import ModelLoadError

logger = get_logger("Checkpoint")

def save_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_acc: float,
    model_name: str,
    class_to_idx: Dict[str, int]
) -> None:
    """Save model checkpoint with metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "model_name": model_name,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }
    
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved successfully to {path} (Epoch {epoch}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f})")

def load_checkpoint(path: Union[str, Path], device: torch.device = torch.device("cpu")) -> Dict[str, Any]:
    """Load model checkpoint from path."""
    path = Path(path)
    if not path.exists():
        raise ModelLoadError(f"Checkpoint file not found at: {path}")
        
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        logger.info(f"Loaded checkpoint from {path} (Model: {checkpoint.get('model_name')}, Epoch: {checkpoint.get('epoch')})")
        return checkpoint
    except Exception as e:
        raise ModelLoadError(f"Failed to load checkpoint from {path}: {str(e)}") from e
