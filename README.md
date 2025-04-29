# 🧠 Brain Tumor Segmentation Using Deep Learning

A deep learning-based system for automated brain tumor segmentation in MRI scans using 3D U-Net, developed as part of an academic research project. The solution is trained and validated on the BraTS 2020 dataset and demonstrates strong performance in segmenting complex tumor structures with high spatial accuracy.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Contributors](#contributors)


---

## 🚀 Overview
This project implements an end-to-end pipeline for:
- Preprocessing 3D MRI scans (normalization, skull stripping, resizing)
- Training a custom 3D U-Net architecture using TensorFlow/Keras
- Segmenting tumor regions into enhancing tumor, edema, and necrotic core
- Evaluating model performance using metrics like Dice Coefficient and IoU

---

## 📂 Dataset
We use the [BraTS 2020](https://drive.google.com/drive/folders/1O64tMYUBfJQNor4w2QgEjOS2Ph-vwLGy?usp=sharing) dataset, which includes multimodal MRI scans (T1, T1c, T2, FLAIR) and ground-truth annotations for:
- Enhancing Tumor
- Tumor Core
- Whole Tumor

---

## 🧠 Model Architecture
We use a modified 3D U-Net:
- Encoder-decoder structure with skip connections
- Dice + Weighted Cross-Entropy loss
- 5-fold cross-validation
- Data augmentation: rotation, flipping, noise, elastic deformation

---

## 🛠 Installation

Clone this repository:
```bash
git clone https://github.com/yourusername/brain-tumor-segmentation.git
cd brain-tumor-segmentation
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Make sure you have:
- Python 3.8+
- TensorFlow 2.x
- CUDA (for GPU support)

---

## ▶️ Usage

### 1. Preprocess the data
```bash
python preprocess.py --input /path/to/brats --output /path/to/processed
```

### 2. Train the model
```bash
python train.py --config config.yaml
```

### 3. Evaluate or run inference
```bash
python predict.py --model saved_model.h5 --input /path/to/test_data
```

---

## 📊 Results

| Metric              | Score       |
|---------------------|-------------|
| Dice Coefficient    | 0.85        |
| Intersection over Union (IoU) | 0.81 |

Visualizations are available in the `/results/` directory.

---

## 📁 Project Structure
```
.
├── data/
├── models/
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── results/
├── config.yaml
├── requirements.txt
└── README.md
```

---

## 🔭 Future Work
- Integrate attention mechanisms / transformer layers
- Add support for multi-modal image fusion
- Deploy lightweight version for real-time inference
- Add explainability using Grad-CAM or SHAP

---

## 👨‍💻 Contributors
- Siddhant Pawbake  
- Mohit Chaudhary  
- Nikhil Thakur  
- Lovish Wadhwa  
- Supervisor: Dr. Rosevir Singh

---
