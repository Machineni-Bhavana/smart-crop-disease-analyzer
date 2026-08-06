import tempfile
from pathlib import Path
import numpy as np
import cv2
import pytest
import torch

from src.data.dataset import PlantDiseaseDataset
from src.data.transforms import get_eval_transforms
from src.utils.exceptions import InvalidImageError, ConfigurationError

@pytest.fixture
def temp_dataset_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_path = Path(tmp_dir)
        class_a = base_path / "ClassA"
        class_b = base_path / "ClassB"
        class_a.mkdir()
        class_b.mkdir()
        
        # Create dummy valid RGB images
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img1[:, :] = (255, 0, 0)
        cv2.imwrite(str(class_a / "img1.jpg"), img1)
        
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2[:, :] = (0, 255, 0)
        cv2.imwrite(str(class_b / "img2.jpg"), img2)
        
        yield base_path

def test_dataset_init_and_len(temp_dataset_dir):
    transform = get_eval_transforms(image_size=224)
    dataset = PlantDiseaseDataset(root_dir=temp_dataset_dir, transform=transform)
    assert len(dataset) == 2
    assert len(dataset.classes) == 2
    assert "ClassA" in dataset.class_to_idx
    assert "ClassB" in dataset.class_to_idx

def test_dataset_getitem(temp_dataset_dir):
    transform = get_eval_transforms(image_size=224)
    dataset = PlantDiseaseDataset(root_dir=temp_dataset_dir, transform=transform)
    
    img_tensor, label = dataset[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 224, 224)
    assert isinstance(label, int)

def test_dataset_invalid_dir():
    with pytest.raises(ConfigurationError):
        _ = PlantDiseaseDataset(root_dir="/non_existent_directory_xyz")

def test_dataset_corrupted_image(temp_dataset_dir):
    corrupt_file = temp_dataset_dir / "ClassA" / "corrupt.jpg"
    with open(corrupt_file, "w") as f:
        f.write("Not an image data")
        
    dataset = PlantDiseaseDataset(root_dir=temp_dataset_dir)
    with pytest.raises(InvalidImageError):
        # The last sample is the corrupt file
        idx = [i for i, (p, _) in enumerate(dataset.samples) if p == corrupt_file][0]
        _ = dataset[idx]
