import time
from pathlib import Path
from typing import Dict, Any, Union, List
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from config import Config
from src.models.model_factory import create_model
from src.data.transforms import get_eval_transforms
from src.utils.checkpoint import load_checkpoint
from src.utils.logger import get_logger
from src.utils.exceptions import InvalidImageError, InferenceError, ModelLoadError

logger = get_logger("Inference")

class DiseasePredictor:
    """
    Reusable inference engine for plant disease classification.
    """
    def __init__(self, checkpoint_path: Union[str, Path] = Config.BEST_MODEL_PATH, device: torch.device = None):
        self.device = device if device is not None else Config.get_device()
        self.checkpoint_path = Path(checkpoint_path)
        
        if not self.checkpoint_path.exists():
            raise ModelLoadError(f"Checkpoint not found at '{self.checkpoint_path}'. Train a model first.")
            
        self.checkpoint = load_checkpoint(self.checkpoint_path, device=self.device)
        self.model_name = self.checkpoint["model_name"]
        self.class_to_idx = self.checkpoint["class_to_idx"]
        self.idx_to_class = {int(k): v for k, v in self.checkpoint["idx_to_class"].items()} if isinstance(list(self.checkpoint["idx_to_class"].keys())[0], (int, str)) else self.checkpoint["idx_to_class"]
        # Ensure integer keys
        self.idx_to_class = {int(k): str(v) for k, v in self.idx_to_class.items()}
        
        self.num_classes = len(self.idx_to_class)
        
        # Instantiate & restore model
        self.model = create_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            pretrained=False,
            freeze_backbone=False
        )
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = get_eval_transforms(image_size=Config.IMAGE_SIZE)
        logger.info(f"DiseasePredictor initialized on device '{self.device}' with model '{self.model_name}'.")

    @staticmethod
    def parse_class_name(raw_class: str) -> tuple[str, str]:
        """
        Parse raw class string like 'Tomato___Early_blight' or 'Apple Black Rot' into (Plant, Disease).
        """
        raw_class = raw_class.strip()
        if "___" in raw_class:
            parts = raw_class.split("___")
            plant = parts[0].replace("_", " ").title()
            disease = parts[1].replace("_", " ").title()
        elif " " in raw_class:
            parts = raw_class.split(" ", 1)
            plant = parts[0].title()
            disease = parts[1].title()
        else:
            plant = "Crop"
            disease = raw_class.replace("_", " ").title()
            
        return plant, disease

    def _load_image_rgb(self, image_input: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Read and convert image input to OpenCV RGB numpy array.
        """
        if isinstance(image_input, (str, Path)):
            path = str(image_input)
            bgr_img = cv2.imread(path)
            if bgr_img is None:
                raise InvalidImageError(f"OpenCV failed to read image path: {path}")
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr_img is None:
                raise InvalidImageError("OpenCV failed to decode image bytes.")
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                # Assume standard RGB if already numpy array
                rgb_img = image_input
            else:
                raise InvalidImageError("Input numpy array must have shape (H, W, 3).")
                
        elif isinstance(image_input, Image.Image):
            rgb_img = np.array(image_input.convert("RGB"))
            
        else:
            raise InvalidImageError(f"Unsupported image input type: {type(image_input)}")
            
        return rgb_img

    def predict(self, image_input: Union[str, Path, bytes, np.ndarray, Image.Image], top_k: int = 3) -> Dict[str, Any]:
        """
        Run inference on an input image and return predictions with confidence and latency.
        """
        try:
            start_time = time.perf_counter()
            
            # Load and preprocess image
            rgb_img = self._load_image_rgb(image_input)
            tensor_img = self.transform(rgb_img).unsqueeze(0).to(self.device)
            
            # Model forward pass inside torch.no_grad()
            with torch.no_grad():
                outputs = self.model(tensor_img)
                probabilities = torch.softmax(outputs, dim=1).squeeze(0)
                
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            
            # Get Top-K predictions
            top_probs, top_indices = torch.topk(probabilities, k=min(top_k, self.num_classes))
            
            top_predictions = []
            for prob, idx in zip(top_probs, top_indices):
                class_idx = idx.item()
                raw_class = self.idx_to_class[class_idx]
                plant, disease = self.parse_class_name(raw_class)
                conf = float(prob.item())
                
                top_predictions.append({
                    "class_name": raw_class,
                    "plant": plant,
                    "disease": disease,
                    "confidence": conf,
                    "confidence_percent": f"{conf * 100:.1f}%"
                })
                
            top_pred = top_predictions[0]
            
            result = {
                "predicted_class": top_pred["class_name"],
                "plant": top_pred["plant"],
                "disease": top_pred["disease"],
                "confidence": top_pred["confidence"],
                "confidence_percent": top_pred["confidence_percent"],
                "top_predictions": top_predictions,
                "inference_time_ms": round(latency_ms, 2)
            }
            
            logger.info(f"Inference completed in {latency_ms:.2f}ms -> {top_pred['plant']} ({top_pred['disease']}): {top_pred['confidence_percent']}")
            return result
            
        except CropAnalyzerException:
            raise
        except Exception as e:
            raise InferenceError(f"Unexpected error during inference: {str(e)}") from e
