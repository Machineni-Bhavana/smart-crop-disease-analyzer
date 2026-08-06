import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from config import Config
from src.data.transforms import get_eval_transforms
from src.data.dataset import create_dataloaders
from src.evaluation.evaluator import Evaluator
from src.utils.logger import get_logger

logger = get_logger("EvaluateScript")

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PyTorch Crop Disease Classifier.")
    parser.add_argument("--checkpoint_path", type=str, default=str(Config.BEST_MODEL_PATH))
    parser.add_argument("--data_dir", type=str, default=str(Config.DATA_DIR))
    parser.add_argument("--output_dir", type=str, default=str(Config.OUTPUT_DIR))
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    data_dir = Path(args.data_dir)
    test_dir = data_dir / "test"
    output_dir = Path(args.output_dir)

    device = Config.get_device()
    logger.info(f"Target evaluation device: {device}")

    eval_transform = get_eval_transforms(image_size=Config.IMAGE_SIZE)

    _, _, test_loader, _ = create_dataloaders(
        train_dir=data_dir / "train",
        val_dir=data_dir / "val",
        test_dir=test_dir,
        eval_transform=eval_transform,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    if test_loader is None:
        logger.error(f"Test dataset directory '{test_dir}' does not exist or is empty.")
        return

    evaluator = Evaluator(checkpoint_path=checkpoint_path, device=device)
    metrics = evaluator.evaluate(test_loader=test_loader, output_dir=output_dir)

    logger.info(f"Test Set Evaluation Results for '{evaluator.model_name}':")
    logger.info(f"  - Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"  - Weighted Precision: {metrics['precision_weighted']:.4f}")
    logger.info(f"  - Weighted Recall: {metrics['recall_weighted']:.4f}")
    logger.info(f"  - Weighted F1-Score: {metrics['f1_weighted']:.4f}")

if __name__ == "__main__":
    main()
