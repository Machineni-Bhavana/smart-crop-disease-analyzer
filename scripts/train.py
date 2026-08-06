import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import Config
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.data.dataset import create_dataloaders
from src.models.model_factory import create_model
from src.training.trainer import Trainer
from src.evaluation.metrics import plot_training_curves
from src.utils.logger import get_logger

logger = get_logger("TrainScript")

def main():
    parser = argparse.ArgumentParser(description="Train PyTorch Plant Disease Classifier.")
    parser.add_argument("--model_name", type=str, default=Config.MODEL_NAME, choices=["resnet18", "mobilenet_v2", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=Config.WEIGHT_DECAY)
    parser.add_argument("--data_dir", type=str, default=str(Config.DATA_DIR))
    parser.add_argument("--checkpoint_path", type=str, default=str(Config.BEST_MODEL_PATH))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    # Set random seeds
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    device = Config.get_device()
    logger.info(f"Target execution device: {device}")

    # Build transforms & DataLoaders
    train_transform = get_train_transforms(image_size=Config.IMAGE_SIZE)
    eval_transform = get_eval_transforms(image_size=Config.IMAGE_SIZE)

    train_loader, val_loader, _, class_to_idx = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=args.batch_size,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    num_classes = len(class_to_idx)
    logger.info(f"Classes ({num_classes}): {list(class_to_idx.keys())}")

    # Create PyTorch CNN Model
    model = create_model(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=True
    )

    # Loss function & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # Instantiate Trainer & run fit
    checkpoint_path = Path(args.checkpoint_path)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        model_name=args.model_name,
        class_to_idx=class_to_idx,
        checkpoint_path=checkpoint_path,
        scheduler=scheduler,
        early_stopping_patience=Config.EARLY_STOPPING_PATIENCE
    )

    history = trainer.fit(num_epochs=args.epochs)

    # Plot training curves
    curves_path = Config.OUTPUT_DIR / "training_curves.png"
    plot_training_curves(history, curves_path)
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
