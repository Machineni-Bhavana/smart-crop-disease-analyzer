from typing import Dict, List, Any
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.training.early_stopping import EarlyStopping
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import get_logger

logger = get_logger("Trainer")

class Trainer:
    """
    Modular training orchestrator for PyTorch plant disease classification.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        model_name: str,
        class_to_idx: Dict[str, int],
        checkpoint_path: Path,
        scheduler: Any = None,
        early_stopping_patience: int = 5
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_name = model_name
        self.class_to_idx = class_to_idx
        self.checkpoint_path = checkpoint_path
        self.scheduler = scheduler
        self.early_stopper = EarlyStopping(patience=early_stopping_patience)
        
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

    def train_epoch(self) -> tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate_epoch(self) -> tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def fit(self, num_epochs: int) -> Dict[str, List[float]]:
        logger.info(f"Starting training run for {num_epochs} epochs on device '{self.device}'...")
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()
            epoch_duration = time.time() - epoch_start
            
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            
            # Step Learning Rate Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
            lr_current = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] ({epoch_duration:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | LR: {lr_current:.6f}"
            )
            
            # Early stopping & model checkpointing check
            is_best = self.early_stopper(val_loss)
            if is_best:
                save_checkpoint(
                    path=self.checkpoint_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    model_name=self.model_name,
                    class_to_idx=self.class_to_idx
                )
                
            if self.early_stopper.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best validation loss: {self.early_stopper.best_loss:.4f}")
                break
                
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.2f} minutes.")
        return self.history
