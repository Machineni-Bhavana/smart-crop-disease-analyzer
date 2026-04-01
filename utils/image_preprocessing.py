import cv2  # type: ignore

# pylint: disable=no-member
import numpy as np  # type: ignore
import time


def preprocess_image(image_input, target_size=(224, 224)):
    """
    Robust OpenCV preprocessing pipeline for crop disease detection.
    Includes resizing, normalization, and handling lighting/contrast variations.
    """
    start_time = time.time()

    # 1. Image Loading and validation
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Image not found or format not supported: {image_input}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Assume it's a numpy array (e.g., from PIL)
        img = np.array(image_input)
        if len(img.shape) == 3 and img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # 2. Handle blurry images using a simple Laplacian variance check (optional logging)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = blur_val < 100.0

    # 3. Skip CLAHE/Contrast enhancement 
    # (Applying contrast algorithms during inference but not during training causes severe Feature Distribution Shift)
    # 4. Resize natively
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # 5. Normalization is handled INTRINSICALLY by the TensorFlow Model's Rescaling Layer
    # The neural network expects values from [0, 255] so the built-in Rescaling(1./127.5) handles it perfectly.
    img_normalized = img_resized.astype("float32")

    # 6. Convert to tensor format (Batch dimension)
    img_tensor = np.expand_dims(img_normalized, axis=0)

    end_time = time.time()
    latency = end_time - start_time

    return img_tensor, is_blurry, latency
