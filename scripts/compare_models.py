import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.data.dataset import create_dataloaders
from src.models.model_factory import create_model, count_parameters
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.utils.logger import get_logger

logger = get_logger("CompareModels")

def benchmark_latency(model: nn.Module, device: torch.device, input_size: int = 224, num_runs: int = 50) -> float:
    """Measure average single-image inference latency in milliseconds."""
    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end = time.perf_counter()
    
    avg_latency_ms = ((end - start) / num_runs) * 1000.0
    return round(avg_latency_ms, 2)

def main():
    parser = argparse.ArgumentParser(description="Train and compare multiple CNN architectures.")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per architecture for comparison")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--data_dir", type=str, default=str(Config.DATA_DIR))
    parser.add_argument("--output_dir", type=str, default=str(Config.OUTPUT_DIR))
    args = parser.parse_args()

    architectures = ["resnet18", "mobilenet_v2", "efficientnet_b0"]
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = Config.get_device()
    logger.info(f"Comparing architectures {architectures} on device '{device}'...")

    train_transform = get_train_transforms(image_size=Config.IMAGE_SIZE)
    eval_transform = get_eval_transforms(image_size=Config.IMAGE_SIZE)

    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        train_dir=data_dir / "train",
        val_dir=data_dir / "val",
        test_dir=data_dir / "test",
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=args.batch_size,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    num_classes = len(class_to_idx)
    results = []

    for arch in architectures:
        logger.info(f"\n========================================================")
        logger.info(f"  Training and Evaluating Architecture: {arch}")
        logger.info(f"========================================================")
        
        torch.manual_seed(Config.SEED)
        
        model = create_model(
            model_name=arch,
            num_classes=num_classes,
            pretrained=True,
            freeze_backbone=True
        )
        
        total_params, trainable_params = count_parameters(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LEARNING_RATE)
        
        checkpoint_path = output_dir / f"checkpoint_{arch}.pth"
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            model_name=arch,
            class_to_idx=class_to_idx,
            checkpoint_path=checkpoint_path,
            early_stopping_patience=Config.EARLY_STOPPING_PATIENCE
        )
        
        trainer.fit(num_epochs=args.epochs)
        
        # Evaluate on Test Set
        evaluator = Evaluator(checkpoint_path=checkpoint_path, device=device)
        arch_output_dir = output_dir / f"eval_{arch}"
        metrics = evaluator.evaluate(test_loader=test_loader, output_dir=arch_output_dir)
        
        # Measure Latency
        latency_ms = benchmark_latency(evaluator.model, device=device)
        
        results.append({
            "Model": arch,
            "Accuracy": f"{metrics['accuracy']*100:.2f}%",
            "Precision": round(metrics['precision_weighted'], 4),
            "Recall": round(metrics['recall_weighted'], 4),
            "F1-Score": round(metrics['f1_weighted'], 4),
            "Total Params": f"{total_params:,}",
            "Trainable Params": f"{trainable_params:,}",
            "Avg Inference Time (ms)": latency_ms
        })

    # Generate Comparison Table CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    
    logger.info("\n=== MODEL ARCHITECTURE COMPARISON SUMMARY ===")
    logger.info("\n" + df.to_string(index=False))
    logger.info(f"Comparison report saved to '{csv_path}'")

if __name__ == "__main__":
    main()
