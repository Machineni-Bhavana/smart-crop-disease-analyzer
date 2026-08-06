import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable, Union
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.logger import get_logger
from src.utils.exceptions import InvalidImageError, ConfigurationError

logger = get_logger("Dataset")

class PlantDiseaseDataset(Dataset):
    """
    Custom PyTorch Dataset for Plant Disease image classification.
    Uses OpenCV for loading and color space transformation (BGR -> RGB).
    """
    
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            root_dir: Directory with class subfolders (e.g. data/train/ClassA/img1.jpg)
            transform: Transformation callable to apply on RGB numpy array
            class_to_idx: Optional class mapping dictionary for dataset consistency
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        if not self.root_dir.exists():
            raise ConfigurationError(f"Dataset root directory does not exist: {self.root_dir}")
            
        self.classes, self.class_to_idx = self._find_classes(class_to_idx)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.samples: List[Tuple[Path, int]] = self._make_dataset()
        
        logger.info(f"Initialized PlantDiseaseDataset from '{self.root_dir}' with {len(self.samples)} samples across {len(self.classes)} classes.")

    def _find_classes(self, class_to_idx: Optional[Dict[str, int]]) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        if not classes:
            raise ConfigurationError(f"No class subdirectories found in '{self.root_dir}'. Expected structure: root_dir/class_name/image.jpg")
            
        if class_to_idx is None:
            c_to_i = {cls_name: i for i, cls_name in enumerate(classes)}
        else:
            c_to_i = class_to_idx
            classes = sorted(list(c_to_i.keys()))
            
        return classes, c_to_i

    def _make_dataset(self) -> List[Tuple[Path, int]]:
        samples = []
        for target_class in self.classes:
            class_dir = self.root_dir / target_class
            if not class_dir.exists() or not class_dir.is_dir():
                continue
                
            class_idx = self.class_to_idx[target_class]
            for file_path in sorted(class_dir.iterdir()):
                if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    samples.append((file_path, class_idx))
                    
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]
        
        # Load image via OpenCV
        bgr_img = cv2.imread(str(image_path))
        if bgr_img is None:
            raise InvalidImageError(f"Failed to load image with OpenCV from: {image_path}")
            
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        
        # Apply torchvision / PIL transforms
        if self.transform is not None:
            image_tensor = self.transform(rgb_img)
        else:
            # Default fallback tensor conversion if no transform supplied
            image_tensor = torch.from_numpy(rgb_img.transpose((2, 0, 1))).float() / 255.0
            
        return image_tensor, label


def create_dataloaders(
    train_dir: Union[str, Path],
    val_dir: Union[str, Path],
    test_dir: Optional[Union[str, Path]] = None,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict[str, int]]:
    """
    Factory function to create PyTorch DataLoaders for Training, Validation, and Test datasets.
    """
    train_dataset = PlantDiseaseDataset(train_dir, transform=train_transform)
    class_to_idx = train_dataset.class_to_idx
    
    val_dataset = PlantDiseaseDataset(val_dir, transform=eval_transform, class_to_idx=class_to_idx)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training dataset to ensure i.i.d batches & prevent ordering bias
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep deterministic ordering for deterministic loss/metric tracking
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = None
    if test_dir and Path(test_dir).exists():
        test_dataset = PlantDiseaseDataset(test_dir, transform=eval_transform, class_to_idx=class_to_idx)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,  # Deterministic test set evaluation
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
    return train_loader, val_loader, test_loader, class_to_idx
