from pathlib import Path
from typing import Dict, Any, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.model_factory import create_model
from src.utils.checkpoint import load_checkpoint
from src.evaluation.metrics import calculate_metrics, save_evaluation_outputs
from src.utils.logger import get_logger

logger = get_logger("Evaluator")

class Evaluator:
    """
    Evaluator class for running model evaluation on held-out test set.
    """
    def __init__(self, checkpoint_path: Union[str, Path], device: torch.device):
        self.device = device
        self.checkpoint = load_checkpoint(checkpoint_path, device=device)
        
        self.model_name = self.checkpoint["model_name"]
        self.class_to_idx = self.checkpoint["class_to_idx"]
        self.idx_to_class = self.checkpoint["idx_to_class"]
        self.target_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
        
        # Instantiate model architecture & restore weights
        self.model = create_model(
            model_name=self.model_name,
            num_classes=len(self.target_names),
            pretrained=False,
            freeze_backbone=False
        )
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader, output_dir: Union[str, Path]) -> Dict[str, Any]:
        logger.info(f"Evaluating model '{self.model_name}' on test set ({len(test_loader.dataset)} samples)...")
        y_true = []
        y_pred = []
        
        for images, labels in test_loader:
            images = images.to(self.device)
            outputs = self.model(images)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            
        metrics = calculate_metrics(y_true, y_pred, self.target_names)
        save_evaluation_outputs(metrics, self.target_names, output_dir)
        return metrics
