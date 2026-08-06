import pytest
import torch

from src.models.model_factory import create_model, count_parameters
from src.utils.exceptions import ConfigurationError

@pytest.mark.parametrize("model_name", ["resnet18", "mobilenet_v2", "efficientnet_b0"])
def test_create_model_architectures(model_name):
    num_classes = 5
    model = create_model(model_name=model_name, num_classes=num_classes, pretrained=False, freeze_backbone=True)
    
    dummy_input = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == (2, num_classes)

def test_count_parameters():
    model = create_model("mobilenet_v2", num_classes=3, pretrained=False, freeze_backbone=True)
    total_params, trainable_params = count_parameters(model)
    
    assert total_params > 0
    assert trainable_params > 0
    assert trainable_params < total_params  # Backbone is frozen

def test_invalid_model_name():
    with pytest.raises(ConfigurationError):
        _ = create_model("unsupported_cnn_architecture", num_classes=5)
