class CropAnalyzerException(Exception):
    """Base exception class for Smart Crop Disease Analyzer."""
    pass

class InvalidImageError(CropAnalyzerException):
    """Raised when an image cannot be read, decoded, or processed by OpenCV/PIL."""
    pass

class ModelLoadError(CropAnalyzerException):
    """Raised when loading model weights or checkpoint fails."""
    pass

class InferenceError(CropAnalyzerException):
    """Raised when inference on an input fails."""
    pass

class ConfigurationError(CropAnalyzerException):
    """Raised when invalid options or missing paths are provided in configuration."""
    pass
