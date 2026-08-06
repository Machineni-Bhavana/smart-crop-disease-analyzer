import json
from pathlib import Path
from typing import Dict, List, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

from src.utils.logger import get_logger

logger = get_logger("Metrics")

def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    target_names: List[str]
) -> Dict[str, Any]:
    """
    Calculate classification metrics: Accuracy, Precision, Recall, F1-score (weighted and macro).
    """
    acc = float(accuracy_score(y_true, y_pred))
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        "accuracy": acc,
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
    }
    
    logger.info(f"Evaluation Metrics - Accuracy: {acc*100:.2f}%, Weighted F1: {f1_weighted:.4f}, Macro F1: {f1_macro:.4f}")
    return metrics

def save_evaluation_outputs(
    metrics: Dict[str, Any],
    target_names: List[str],
    output_dir: Union[str, Path]
) -> None:
    """
    Save evaluation output files: metrics.json, classification_report.csv, confusion_matrix.png.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save metrics JSON
    json_path = output_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump({
            "accuracy": metrics["accuracy"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
            "f1_weighted": metrics["f1_weighted"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"]
        }, f, indent=4)
    logger.info(f"Saved metrics JSON to {json_path}")
    
    # 2. Save Classification Report CSV
    report_df = pd.DataFrame(metrics["classification_report"]).transpose()
    csv_path = output_dir / "classification_report.csv"
    report_df.to_csv(csv_path)
    logger.info(f"Saved classification report CSV to {csv_path}")
    
    # 3. Plot & Save Confusion Matrix PNG
    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45, ha="right")
    plt.yticks(tick_marks, target_names)
    
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
                     
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {cm_path}")

def plot_training_curves(history: Dict[str, List[float]], output_path: Union[str, Path]) -> None:
    """
    Plot and save training/validation loss and accuracy curves.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history["train_loss"], "b-o", label="Training Loss")
    ax1.plot(epochs, history["val_loss"], "r-s", label="Validation Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Training Accuracy")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "r-s", label="Validation Accuracy")
    ax2.set_title("Training & Validation Accuracy (%)")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved training curves plot to {output_path}")
