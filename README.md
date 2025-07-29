# 🧠 Optimized_Brain_Tumor_Segmentation_in_MRI_Scans_using_U-Net_and_Hausdorff_Distance_Minimization

---

## 📑 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Evaluation Metrics](#evaluation-metrics)
- [Execution Environment](#execution-environment)
- [Results](#results)
- [Future Scope](#future-scope)

---

## 🧩 Overview

This project presents a deep learning-based solution for automated brain tumor segmentation in MRI scans. Leveraging the U-Net architecture, widely recognized for its efficiency in biomedical image segmentation, the model incorporates a Hausdorff Distance-driven loss function to improve boundary-level accuracy. The goal is to reduce human error and variability in tumor delineation by delivering precise and consistent segmentation maps.

---

## ❓ Problem Statement

Manual segmentation of brain tumors is a labor-intensive and error-prone task, often varying between experts. Traditional loss functions like Dice and IoU focus primarily on region overlap, ignoring the accuracy of boundary alignment. This project addresses the challenge of enhancing tumor boundary precision in MRI segmentation by integrating Hausdorff Distance into the learning process, allowing the model to focus on minimizing edge discrepancies.

---

## 🎯 Objectives

- Design and implement a U-Net model tailored for high-precision brain tumor segmentation.
- Incorporate a Hausdorff Distance-based loss function to improve contour alignment.
- Evaluate the model's performance against standard metrics and conventional loss functions.
- Build a reusable and modular framework that can support expansion across imaging modalities and tumor types.

---

## 📂 Dataset

Two publicly available datasets were used to train and evaluate the model:

1. **BraTS Dataset**  
   A comprehensive MRI dataset for brain tumor segmentation, featuring multimodal scans (T1, T2, FLAIR) with pixel-level annotations.

2. **Brain MRI Classification Dataset**  
   Includes categorized MRI images (glioma, meningioma, pituitary tumors, and healthy cases), used for robustness checks and additional validation.

All data underwent preprocessing including resizing, normalization, and augmentation to improve model generalization.

---

## ⚙️ Methodology

The pipeline follows a modular deep learning approach structured into five key stages:

- **Model Architecture**: U-Net with an encoder-decoder layout optimized for biomedical segmentation.
- **Loss Function**: Combination of Hausdorff Distance-based loss and Dice loss to balance region accuracy with boundary precision.
- **Preprocessing**: Standard resizing and intensity normalization of MRI slices.
- **Data Augmentation**: Random rotations, flips, zooms, and contrast shifts to improve variability tolerance.
- **Training Strategy**: Validation against standard benchmarks using k-fold cross-validation for robustness.

---

## 📏 Evaluation Metrics

Model performance is measured across both overlap-based and boundary-aware metrics:

- **Dice Coefficient** – Measures similarity between predicted and actual masks.
- **Intersection over Union (IoU)** – Evaluates the extent of agreement between predictions and ground truth.
- **Hausdorff Distance (HD)** – Quantifies the maximum deviation between prediction and ground truth boundaries, capturing worst-case edge errors.

These metrics together provide a comprehensive assessment of both segmentation accuracy and clinical reliability.

---

## 💻 Execution Environment

The project was developed and tested in a cloud-based environment using:

- **Platform**: Kaggle Notebooks
- **Language**: Python 3.x

### Key Libraries & Frameworks

- **PyTorch** – Model implementation and training
- **NumPy / OpenCV** – Image handling and preprocessing
- **Matplotlib / Seaborn** – Visualization and plotting
- **SciPy** – Scientific computing and Hausdorff Distance computation
- **Albumentations** – Advanced data augmentation pipeline

---

## 📊 Results

### 📈 Quantitative Results

| Metric                  | Training | Validation |
|-------------------------|----------|------------|
| Dice Coefficient        | 0.9054   | 0.8807     |
| Intersection over Union | 0.8275   | 0.7877     |
| Accuracy                | 99.51%   | 99.47%     |
| Hausdorff Distance      | ↓ Significant improvement compared to baseline loss models |

### 🧠 Qualitative Analysis

- The model demonstrated high overlap with ground truth masks across all tumor types.
- Boundary definitions were more accurate and stable due to Hausdorff optimization.
- Segmentation robustness was retained across data augmentation and unseen variations.

---

## 🚀 Future Scope

- **Dataset Expansion**: Integrate multi-center and real-world clinical datasets for greater generalization.
- **Tumor Subtype Segmentation**: Extend to segment multiple tumor components (e.g., edema, necrosis, core).
- **Multi-Modality Extension**: Adapt the model to work with CT, PET, and ultrasound imaging.
- **Post-Processing Techniques**: Explore morphological filters and graph-based refinement methods.
- **Web Application**: Build a lightweight front-end for clinicians to upload and segment MRI scans in real-time.
- **Hybrid Architectures**: Incorporate transformer-based modules (e.g., TransUNet) for capturing global features.

---
