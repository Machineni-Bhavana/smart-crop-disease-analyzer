import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import shutil
from typing import List, Tuple
from sklearn.model_selection import train_test_split

from config import Config
from src.utils.logger import get_logger

logger = get_logger("PrepareDataset")

def prepare_splits(
    data_dir: Path = Config.DATA_DIR,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = Config.SEED
) -> None:
    """
    Split raw class subfolders in data_dir into train, val, and test splits.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Splits must sum to 1.0"
    
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    
    # Check if splits already exist
    if train_dir.exists() and val_dir.exists() and test_dir.exists():
        logger.info(f"Split directories already exist at '{data_dir}'. Skipping dataset preparation.")
        return
        
    logger.info(f"Scanning for raw class directories in '{data_dir}'...")
    
    # Identify class subdirectories excluding train, val, test, and output dirs
    ignore_names = {"train", "val", "test", "models", "outputs", "venv", "__pycache__", ".git"}
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name not in ignore_names]
    
    if not class_dirs:
        logger.warning(f"No class folders found in '{data_dir}' to split.")
        return
        
    image_paths: List[Path] = []
    labels: List[str] = []
    
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    for c_dir in class_dirs:
        for f in c_dir.iterdir():
            if f.is_file() and f.suffix.lower() in supported_exts:
                image_paths.append(f)
                labels.append(c_dir.name)
                
    total_images = len(image_paths)
    logger.info(f"Found {total_images} total images across {len(class_dirs)} classes: {[c.name for c in class_dirs]}")
    
    if total_images == 0:
        logger.error(f"No supported image files found in '{data_dir}'.")
        return
        
    # First split into train and temp (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=val_test_ratio, random_state=seed, stratify=labels
    )
    
    # Second split temp into val and test
    relative_test_ratio = test_ratio / val_test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=relative_test_ratio, random_state=seed, stratify=temp_labels
    )
    
    logger.info(f"Stratified Split Counts -> Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Copy files into split directories
    def copy_files(paths: List[Path], target_base: Path):
        for path in paths:
            class_name = path.parent.name
            dest_dir = target_base / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest_dir / path.name)
            
    logger.info("Copying split files...")
    copy_files(train_paths, train_dir)
    copy_files(val_paths, val_dir)
    copy_files(test_paths, test_dir)
    
    logger.info("Dataset preparation successfully completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare stratified dataset train/val/test splits.")
    parser.add_argument("--data_dir", type=str, default=str(Config.DATA_DIR), help="Base dataset path")
    parser.add_argument("--seed", type=int, default=Config.SEED, help="Random seed")
    args = parser.parse_args()
    
    prepare_splits(data_dir=Path(args.data_dir), seed=args.seed)
