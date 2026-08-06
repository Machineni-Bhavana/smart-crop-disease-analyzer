import tempfile
from pathlib import Path
import numpy as np
import cv2
import pytest
import torch

from src.models.model_factory import create_model
from src.utils.checkpoint import save_checkpoint
from src.inference import DiseasePredictor

@pytest.fixture
def temp_checkpoint():
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "test_model.pth"
        
        class_to_idx = {"Apple Black Rot": 0, "Healthy": 1}
        model = create_model("mobilenet_v2", num_classes=2, pretrained=False)
        
        save_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=None,
            epoch=1,
            val_loss=0.1,
            val_acc=0.95,
            model_name="mobilenet_v2",
            class_to_idx=class_to_idx
        )
        yield checkpoint_path

def test_disease_predictor_inference(temp_checkpoint):
    predictor = DiseasePredictor(checkpoint_path=temp_checkpoint, device=torch.device("cpu"))
    
    # Create dummy RGB numpy image
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_img[:, :] = (100, 150, 200)
    
    result = predictor.predict(dummy_img, top_k=2)
    
    assert "predicted_class" in result
    assert "plant" in result
    assert "disease" in result
    assert "confidence" in result
    assert "top_predictions" in result
    assert "inference_time_ms" in result
    
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_predictions"]) == 2
    assert result["inference_time_ms"] > 0.0

def test_parse_class_name():
    plant, disease = DiseasePredictor.parse_class_name("Tomato___Early_blight")
    assert plant == "Tomato"
    assert disease == "Early Blight"
    
    plant, disease = DiseasePredictor.parse_class_name("Apple Black Rot")
    assert plant == "Apple"
    assert disease == "Black Rot"
