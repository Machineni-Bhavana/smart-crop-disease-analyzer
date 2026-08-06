import os
from pathlib import Path
from dotenv import load_dotenv
import torch

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

class Config:
    # Directory paths
    DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "val"
    TEST_DIR = DATA_DIR / "test"
    
    MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "outputs"))
    BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
    
    # Model & Hyperparameters
    MODEL_NAME = os.getenv("MODEL_NAME", "mobilenet_v2")
    IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 224))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 1e-4))
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 10))
    EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 5))
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", 2))
    SEED = int(os.getenv("SEED", 42))
    
    # Normalization constants (ImageNet defaults)
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def get_device(cls) -> torch.device:
        forced_device = os.getenv("FORCE_DEVICE", None)
        if forced_device:
            return torch.device(forced_device)
            
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @classmethod
    def ensure_directories(cls):
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)

Config.ensure_directories()
