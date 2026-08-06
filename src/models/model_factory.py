import ssl
from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as models

# Disable SSL verification for torchvision pretrained weight downloads on macOS if needed
ssl._create_default_https_context = ssl._create_unverified_context

from src.utils.logger import get_logger
from src.utils.exceptions import ConfigurationError

logger = get_logger("ModelFactory")

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params) of a PyTorch module."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True
) -> nn.Module:
    """
    Factory function to instantiate CNN architectures for transfer learning.
    
    Supported models:
    - resnet18
    - mobilenet_v2
    - efficientnet_b0
    
    Args:
        model_name: Architecture identifier string
        num_classes: Target number of output disease categories
        pretrained: Use ImageNet pretrained weights if True
        freeze_backbone: Freeze feature extractor layers if True
    """
    name = model_name.lower().strip()
    
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        # Replace final classification head (fc)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
        
    elif name in ["mobilenet_v2", "mobilenetv2"]:
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
                
        # Replace classifier head
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    elif name in ["efficientnet_b0", "efficientnetb0"]:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
                
        # Replace classifier head
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
        
    else:
        raise ConfigurationError(
            f"Unsupported architecture '{model_name}'. Choose from: ['resnet18', 'mobilenet_v2', 'efficientnet_b0']"
        )
        
    total_params, trainable_params = count_parameters(model)
    logger.info(
        f"Created model '{name}' (num_classes={num_classes}, pretrained={pretrained}, freeze_backbone={freeze_backbone}). "
        f"Params: Total={total_params:,}, Trainable={trainable_params:,}"
    )
    
    return model
