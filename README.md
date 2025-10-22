# PR_Group_Assignment-NetMinds  
by **Eshin Menusha (220165F)**, **Dilanka Heshan (220338N)**, **Pulindu Ranaweera (220508L)**, **Anuja Kalhara (220414U)**

---

## Overview  
This repository contains a comparative study of three **deep-learning-based image classification** approaches, implemented as independent Jupyter Notebooks.  
Each experiment explores a modern architecture (CNNs and Transformers) across binary and multi-class tasks.

---

## Purpose & Scope  
- Investigate and compare different architectures and frameworks for image classification tasks.  
- Implement three independent approaches:  
  - **Binary classification (real vs fake videos)** — using *EfficientNetB0 (TensorFlow/Keras)*  
  - **Binary classification (AI-generated vs human images)** — using *EfficientNetV2-M (PyTorch)*  
  - **Multi-class classification (CIFAR-10 dataset)** — using *Vision Transformer ViT-B/16 (PyTorch)*  
- Provide insights into workflow design, model fine-tuning, and performance comparison.

---

## Repository Structure  

| Notebook | Task | Architecture | Framework | Dataset |
|-----------|------|--------------|------------|----------|
| `deepfake-detection-efficientnetb0.ipynb` | Binary (real vs fake videos) | EfficientNetB0 | TensorFlow/Keras | Video frames |
| `efficientnet-approach.ipynb` | Binary (AI vs human images) | EfficientNetV2-M | PyTorch | Image dataset + CSV metadata |
| `visiontransformer-based-classification.ipynb` | Multi-class (CIFAR-10) | ViT-B/16 | PyTorch | CIFAR-10 |

---

## Technology Stack  
- **TensorFlow 2.x + Keras** (for CNN-based deepfake detection)  
- **PyTorch + torchvision** (for EfficientNetV2-M and ViT experiments)  
- **Supporting libraries:** numpy, pandas, scikit-learn, matplotlib  

This setup demonstrates that transfer-learning principles (feature extraction → fine-tuning → evaluation) are *framework-agnostic*.

---

## Common Workflow  
Across all experiments, the following pattern is used:

1. Load a pre-trained model with ImageNet weights  
2. Freeze early layers (feature extractor)  
3. Add task-specific classification head  
4. Optionally unfreeze deeper layers for fine-tuning  
5. Apply data augmentation and regularization  
6. Train with early stopping and evaluate using accuracy, F1-score, and confusion matrix  

---

## Getting Started  

### Prerequisites  
- Python 3.x  
- Jupyter Notebook / JupyterLab  
- TensorFlow 2.x and/or PyTorch  
- GPU (optional but recommended)

### Setup  

```bash
# Clone the repository
git clone https://github.com/<your-username>/PR_Group_Assignment-NetMinds.git
cd PR_Group_Assignment-NetMinds

# Install dependencies
pip install -r requirements.txt
