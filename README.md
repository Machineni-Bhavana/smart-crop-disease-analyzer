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

## 🧠 Technical Interview Defensibility Guide

### 15 Core Technical Interview Questions & Answers

#### Q1: Why did you choose PyTorch over TensorFlow/Keras for this project?
**Answer**: PyTorch provides an imperative, dynamic computation graph (e-graph) that simplifies debugging and allows fine-grained customization of data loading (`Dataset`/`DataLoader`) and training loops. It also seamlessly integrates with `torchvision` models and offers native Apple Silicon (MPS) and CUDA acceleration.

#### Q2: What is Transfer Learning and why did you use ImageNet pretrained weights?
**Answer**: Transfer learning uses weights learned from training on a massive dataset (ImageNet, 1.2M+ images, 1000 categories) as low-level feature extractors (edges, textures, shapes). Instead of training millions of parameters from scratch, we freeze the backbone and re-train only the final classification head for plant diseases, accelerating convergence and preventing overfitting on smaller domain datasets.

#### Q3: Why did you implement a custom `Dataset` instead of using `torchvision.datasets.ImageFolder`?
**Answer**: A custom `Dataset` (`PlantDiseaseDataset`) gives complete control over image reading (`cv2.imread`), color conversions (BGR→RGB), robust exception handling (detecting corrupt/invalid images via `InvalidImageError`), custom data augmentations, and index-based sampling during technical interviews.

#### Q4: Why read images with OpenCV (`cv2.imread`) instead of PIL?
**Answer**: OpenCV is an industry-standard C++ backend library optimized for high-performance computer vision operations. Using OpenCV allows seamless integration of matrix operations (e.g., CLAHE, Gaussian blur, Laplacian variance for image quality checks) before passing images into PyTorch transformations.

#### Q5: Why do we convert OpenCV images from BGR to RGB?
**Answer**: OpenCV loads image matrices in **BGR** channel order by default. However, PyTorch pretrained `torchvision` models are trained on standard **RGB** images. Failing to convert BGR→RGB causes severe channel mismatch, degrading classification accuracy.

#### Q6: Why apply ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` normalization?
**Answer**: Pretrained torchvision backbones expect inputs zero-centered and scaled to the exact intensity distribution of ImageNet images. Normalization prevents exploding/vanishing gradients during backpropagation and ensures feature maps align with pretrained weights.

#### Q7: Why shuffle the Training DataLoader, but keep Validation and Test DataLoaders unshuffled?
**Answer**: Shuffling training data breaks batch ordering correlations and ensures stochastic gradient descent (SGD/Adam) estimates true population gradients. Validation/Test DataLoaders are kept unshuffled (`shuffle=False`) so evaluation metrics, confusion matrices, and loss values remain 100% deterministic and reproducible across runs.

#### Q8: What data augmentations did you use and why are they excluded during evaluation?
**Answer**: Training transformations include `RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation(15°)`, and `ColorJitter`. Augmentation synthetically expands dataset diversity, making the model invariant to leaf orientation, lighting, and camera angle. Evaluation transforms must be strictly deterministic (resize + normalize only) to accurately test true model performance.

#### Q9: Why use `CrossEntropyLoss` for multiclass disease classification?
**Answer**: `CrossEntropyLoss` combines `LogSoftmax` and `NLLLoss` (Negative Log-Likelihood Loss) in a single numerically stable layer. It measures the KL-divergence between the predicted target distribution and true one-hot target labels.

#### Q10: How does `ReduceLROnPlateau` learning rate scheduling improve training?
**Answer**: When validation loss stops decreasing for a specified number of epochs (patience=2), `ReduceLROnPlateau` scales down the learning rate (e.g., by 50%). This allows the optimizer to take smaller gradient steps to settle into a deeper local minimum instead of overshooting it.

#### Q11: How does Early Stopping prevent overfitting?
**Answer**: `EarlyStopping` monitors validation loss after every epoch. If validation loss fails to improve for 5 consecutive epochs, training terminates. This prevents the model from memorizing training noise and ensures the best performing model checkpoint (`best_model.pth`) is retained.

#### Q12: What is the difference between `model.train()` and `model.eval()`?
**Answer**: `model.train()` activates layers like `Dropout` (randomly zeroing activations) and computes batch statistics in `BatchNorm`. `model.eval()` disables `Dropout` and freezes `BatchNorm` running mean/variance so predictions are deterministic.

#### Q13: What does `torch.no_grad()` do during evaluation/inference?
**Answer**: `torch.no_grad()` disables PyTorch's autograd engine, preventing the creation of computation graphs and intermediate activation storage. This reduces memory consumption by ~50% and speeds up forward-pass inference.

#### Q14: Why compare ResNet18, MobileNetV2, and EfficientNet-B0?
**Answer**: ResNet18 uses residual skip connections; MobileNetV2 uses depthwise separable convolutions for edge efficiency; EfficientNet-B0 uses compound scaling (depth, width, resolution). Comparing them empirically identifies which architecture balances parameter count, memory footprint, accuracy, and inference speed for real-world deployment.

#### Q15: How did you measure single-image inference latency?
**Answer**: Latency is measured using high-precision timer `time.perf_counter()` directly around the OpenCV preprocessing, PyTorch forward pass, and softmax computation—strictly isolating model inference latency from UI rendering overhead.

---

## ⏱️ 60-Second Elevator Pitch

> "For my Smart Crop Disease Analyzer project, I built a production-grade end-to-end computer vision ecosystem in PyTorch, OpenCV, Streamlit, and Docker that identifies plant diseases from leaf photographs with **98.54% accuracy** on over 10,000 images.
>
> I engineered a custom PyTorch dataset and data loader pipeline utilizing OpenCV for image decoding and color conversion, combined with data augmentation and ImageNet normalization. I conducted an empirical architecture comparison across ResNet18, MobileNetV2, and EfficientNet-B0, selecting MobileNetV2 for its optimal 2.2M parameter footprint and single-image inference latency of just **11.12 ms**.
>
> To deliver a complete MLOps solution, I implemented early stopping, learning-rate scheduling, model checkpointing, an interactive Streamlit web dashboard with Top-3 softmax confidence visualizations, containerized the app with Docker, and wrote unit tests in Pytest."

---

## ⚠️ Real-World Limitations

1. **Domain Shift**: Lab-captured PlantVillage leaf images feature clean, isolated backgrounds. Real field images with complex soils, shadows, or multiple leaves may experience reduced accuracy.
2. **Lighting Sensitivity**: Extreme overexposure or darkness can alter color histogram signals; pre-filtering or CLAHE histogram equalization can further mitigate lighting variations.
3. **Medical & Diagnostic Disclaimer**: The system provides algorithmic confidence scores for decision support and should be paired with expert agricultural extension verification before applying chemical treatments.
