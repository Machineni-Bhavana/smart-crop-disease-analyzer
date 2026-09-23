# 🌿 Smart Crop Disease Analyzer

An end-to-end, production-grade Computer Vision and PyTorch application for automated plant disease identification. Built using **PyTorch**, **torchvision**, **OpenCV**, **Streamlit**, and **Docker**, this system leverages transfer learning to classify crop leaf health with **98.54% measured accuracy** on 10,000+ leaf images and an average single-image inference latency of **11.12 ms** (< 0.02 seconds).

---

## 🌟 Key Highlights & Resume Summary

* **High Performance**: Achieved **98.54% test classification accuracy** (Weighted F1-score: **0.9850**) across 10,025 leaf images using PyTorch transfer learning with MobileNetV2.
* **Custom ML Pipeline**: Developed custom PyTorch `Dataset` and `DataLoader` classes with OpenCV BGR→RGB preprocessing, data augmentation, early stopping, model checkpointing, and `ReduceLROnPlateau` scheduling.
* **Interactive Streamlit Web Dashboard**: Created an interactive web app with image uploader, instant OpenCV preprocessing, Top-3 prediction softmax confidence bars, and inference latency tracking.
* **Sub-2-Second Latency**: Measured real single-image model inference latency averaging **11.12 ms** (95th percentile **11.76 ms**), comfortably meeting sub-2-second SLAs.
* **Empirical Architecture Comparison**: Benchmarked **ResNet18**, **MobileNetV2**, and **EfficientNet-B0**, selecting MobileNetV2 as the optimal balance of parameter efficiency (2.2M params) and accuracy.
* **Production MLOps**: Containerized with **Docker** & **Docker Compose**, featuring structured logging, custom exception handling, environment configuration, unit tests (`pytest`), and GitHub Actions CI.

---

## 🏗️ System Architecture & Data Flow

```
                      ┌─────────────────────────────────┐
                      │    Input Leaf Photograph        │
                      │       (JPG / JPEG / PNG)        │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │ OpenCV Image Processing Engine  │
                      │  • Read File / Bytes (cv2)      │
                      │  • Color Space BGR ➔ RGB        │
                      │  • Resize to (224 × 224)        │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │  Torchvision Tensor Transforms  │
                      │  • ImageNet Normalization       │
                      │    Mean: [0.485, 0.456, 0.406]  │
                      │    Std:  [0.229, 0.224, 0.225]  │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │   PyTorch MobileNetV2 CNN       │
                      │  • Pretrained Feature Backbone  │
                      │  • Custom Linear Head           │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │ Softmax Probability Calculation │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │       Streamlit Dashboard       │
                      │  • Crop Category & Disease      │
                      │  • Top-3 Softmax Probabilities  │
                      │  • Measured Latency (ms)        │
                      └─────────────────────────────────┘
```

---

## 📊 Measured Performance & Benchmark Results

### 1. Test Set Evaluation Metrics (MobileNetV2)
Evaluated on **1,504 held-out test images** (15% split):

| Metric | Measured Value |
|---|---|
| **Test Accuracy** | **98.54%** |
| **Weighted Precision** | **0.9855** |
| **Weighted Recall** | **0.9854** |
| **Weighted F1-Score** | **0.9850** |
| **Macro F1-Score** | **0.9746** |
| **Test Samples Evaluated** | 1,504 images |

### 2. Architecture Comparison Study

| Architecture | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Total Parameters | Trainable Parameters | Avg Latency (ms) |
|---|---|---|---|---|---|---|---|
| **ResNet18** | 97.41% | 0.9746 | 0.9741 | 0.9740 | 11,179,590 | 3,078 | 3.29 ms |
| **MobileNetV2** 🏆 | **98.47%** | **0.9848** | **0.9847** | **0.9843** | **2,231,558** | **7,686** | **8.59 ms** |
| **EfficientNet-B0** | 98.14% | 0.9815 | 0.9814 | 0.9814 | 4,015,234 | 7,686 | 10.86 ms |

*Selection Rationale*: MobileNetV2 achieved the highest classification accuracy (**98.47%**) with only **2.2M total parameters** (5x lighter than ResNet18), making it the winner for edge and web deployment.

### 3. Inference Latency Benchmark (`outputs/inference_benchmark.json`)

| Benchmark Stat | Measured Latency |
|---|---|
| **Mean Latency** | **11.12 ms** (0.0111 s) |
| **Median Latency** | **10.39 ms** |
| **95th Percentile (p95)** | **11.76 ms** (0.0118 s) |
| **Min / Max Latency** | 9.96 ms / 27.28 ms |
| **SLA Compliance (<2.0s)** | **PASSED (YES)** |

---

## 🛠️ Project Structure

```
smart-crop-disease-analyzer/
├── app.py                      # Streamlit Interactive Web Application
├── config.py                   # Centralized Configuration & Settings
├── requirements.txt            # Python Dependencies
├── Dockerfile                  # Production Docker Container Specification
├── docker-compose.yml          # Container Orchestration Specification
├── .env.example                # Sample Environment Variables
├── src/                        # Core Application Source Code
│   ├── data/
│   │   ├── dataset.py          # Custom PyTorch Dataset (OpenCV + PIL)
│   │   └── transforms.py       # Data Augmentation & Normalization Transforms
│   ├── models/
│   │   └── model_factory.py    # CNN Factory (ResNet18, MobileNetV2, EfficientNet)
│   ├── training/
│   │   ├── trainer.py          # Training Loop & Metric Tracker
│   │   └── early_stopping.py   # Early Stopping Handler
│   ├── evaluation/
│   │   ├── evaluator.py        # Test Set Evaluator
│   │   └── metrics.py          # Metric Computations & Plotting
│   ├── utils/
│   │   ├── logger.py           # Structured Python Logging
│   │   ├── exceptions.py       # Custom Exceptions
│   │   └── checkpoint.py       # Model Checkpoint Save/Load Utilities
│   └── inference.py            # DiseasePredictor Inference Engine
├── scripts/                    # CLI Operational Scripts
│   ├── prepare_dataset.py      # Stratified Train/Val/Test Splitter
│   ├── train.py                # Model Training Script
│   ├── evaluate.py             # Model Evaluation Script
│   ├── compare_models.py       # Multi-CNN Architecture Comparison Script
│   └── benchmark.py            # Inference Latency Benchmark Script
├── tests/                      # Pytest Automated Test Suite
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_inference.py
├── models/                     # Model Checkpoints (best_model.pth)
├── data/                       # Dataset directory (train/ val/ test/)
└── outputs/                    # Plots, Reports, Confusion Matrices, CSVs
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
```bash
# Clone the repository
git clone https://github.com/Machineni-Bhavana/smart-crop-disease-analyzer.git
cd smart-crop-disease-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Dataset Preparation
Organize dataset into stratified 70% Train, 15% Validation, and 15% Test splits:
```bash
python scripts/prepare_dataset.py
```

### 3. Model Training
Train the default MobileNetV2 classifier with early stopping and LR scheduling:
```bash
python scripts/train.py --model_name mobilenet_v2 --epochs 10 --batch_size 32
```

### 4. Model Evaluation
Evaluate the saved checkpoint against the held-out test set:
```bash
python scripts/evaluate.py
```
*Outputs generated in `outputs/`: `metrics.json`, `classification_report.csv`, `confusion_matrix.png`, `training_curves.png`.*

### 5. Multi-Architecture Comparison
Train and compare ResNet18, MobileNetV2, and EfficientNet-B0:
```bash
python scripts/compare_models.py --epochs 5
```

### 6. Inference Latency Benchmarking
Measure single-image inference latency across test images:
```bash
python scripts/benchmark.py
```

### 7. Run Streamlit Application
Launch the web dashboard locally:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 8. Run Unit Test Suite
Execute unit tests using pytest:
```bash
python -m pytest -v
```

---

## 🐳 Docker Deployment

### Run with Docker CLI
```bash
# Build Docker image
docker build -t smart-crop-disease-analyzer .

# Run container on port 8501
docker run -p 8501:8501 smart-crop-disease-analyzer
```

### Run with Docker Compose
```bash
docker-compose up --build
```

---

## ⚠️ Real-World Limitations

1. **Domain Shift**: Lab-captured PlantVillage leaf images feature clean, isolated backgrounds. Real field images with complex soils, shadows, or multiple leaves may experience reduced accuracy.
2. **Lighting Sensitivity**: Extreme overexposure or darkness can alter color histogram signals; pre-filtering or CLAHE histogram equalization can further mitigate lighting variations.
3. **Medical & Diagnostic Disclaimer**: The system provides algorithmic confidence scores for decision support and should be paired with expert agricultural extension verification before applying chemical treatments.
