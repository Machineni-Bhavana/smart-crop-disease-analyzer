import os
import numpy as np
import tensorflow as tf  # type: ignore

# pylint: disable=no-member
from utils.image_preprocessing import preprocess_image  # type: ignore

# Match the class names with train_model.py
CLASS_NAMES = [
    "Apple Black Rot",
    "Apple Cedar Rust",
    "Apple Scab",
    "Corn Common Rust",
    "Healthy",
    "Tomato Blight",
]


def load_trained_model(model_path="models/crop_disease_model.keras"):
    """
    Loads the trained CNN model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file {model_path} not found. Please run train_model.py first."
        )

    model = tf.keras.models.load_model(model_path)
    return model


def predict_disease(image_input):
    """
    Predicts the disease of a given image.
    :param image_input: File path or numpy array / PIL Image.
    :return: dict with disease_name, probability, is_blurry, confidence_threshold_met, latency
    """
    # 1. Preprocess the image using our custom utility
    img_tensor, is_blurry, prep_latency = preprocess_image(image_input)

    # 2. Load the trained CNN model
    import time

    try:
        model = load_trained_model("models/crop_disease_model.keras")
    except FileNotFoundError as e:
        # Standard fallback for demonstration or fail gracefully
        print(e)
        return {
            "disease_name": "Unknown Disease",
            "probability": 0.0,
            "is_blurry": is_blurry,
            "confidence_threshold_met": False,
            "latency": prep_latency,
        }

    # 3. Make the prediction and measure pure inference time
    start_infer = time.time()
    predictions = model.predict(img_tensor)
    end_infer = time.time()

    infer_latency = end_infer - start_infer
    total_latency = prep_latency + infer_latency

    # 4. Get the class with highest probability
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    prediction_probability = float(np.max(predictions))

    disease_name = CLASS_NAMES[predicted_class_idx]

    # 5. Confidence Threshold (70%)
    confidence_threshold_met = prediction_probability >= 0.70

    return {
        "disease_name": disease_name,
        "probability": prediction_probability,
        "is_blurry": is_blurry,
        "confidence_threshold_met": confidence_threshold_met,
        "latency": total_latency,
    }
