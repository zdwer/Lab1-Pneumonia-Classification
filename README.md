# Lab1: Pneumonia Classification (ResNet18 vs ResNet50)

## Introduction
Binary classification of chest X-ray images (Normal / Pneumonia) using ResNet18 and ResNet50.

## Dataset
Chest X-ray Pneumonia Dataset (Kaggle)

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Models
- ResNet18 and ResNet50
- ImageNet pretrained weights
- Fine-tuning for binary classification

## Requirements
Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage
Train both models:

```bash
python train.py --dataset "./chest_xray" --epochs 10 --batch-size 128
```

The training script saves model weights, evaluation metrics, and comparison figures.

## Evaluation
Models are evaluated using Accuracy, F1-score, and Confusion Matrix.
