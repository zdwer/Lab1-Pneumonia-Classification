from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score



def classification_metrics(labels, predictions):
    """Positive class: PNEUMONIA=1, confusion matrix rows=true, columns=pred."""
    if len(labels) == 0:
        raise ValueError("Evaluation dataset is empty")
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(labels, predictions) * 100),
        "f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "confusion_matrix": cm.astype(int).tolist(),
        "num_samples": int(len(labels)),
    }


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_history_csv(path, history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        raise ValueError("Cannot save empty epoch history")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_accuracy(history, filepath, model_name):
    x = [int(h["epoch"]) for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, label in (("train_accuracy", "Train"), ("val_accuracy", "Validation"), ("test_accuracy", "Test (diagnostic only)")):
        y = [float(h[col]) if h[col] is not None else float("nan") for h in history]
        if not np.isnan(y).all():
            ax.plot(x, y, marker="o", markersize=3, label=label)
    ax.set(xlabel="Epoch", ylabel="Accuracy (%)", title=f"{model_name}: Accuracy by epoch")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath, dpi=170)
    plt.close(fig)


def plot_f1_score(history, filepath, model_name):
    x = [int(h["epoch"]) for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, label in (("train_f1", "Train"), ("val_f1", "Validation"), ("test_f1", "Test (diagnostic only)")):
        y = [float(h[col]) if h[col] is not None else float("nan") for h in history]
        if not np.isnan(y).all():
            ax.plot(x, y, marker="o", markersize=3, label=label)
    ax.set(xlabel="Epoch", ylabel="Pneumonia F1-score", title=f"{model_name}: F1-score by epoch")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath, dpi=170)
    plt.close(fig)


def plot_confusion_matrix(matrix, filepath, title):
    cm = np.asarray(matrix, dtype=np.int64)
    if cm.shape != (2, 2):
        raise ValueError("Confusion matrix must be 2x2")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Normal", "Pneumonia"])
    ax.set_yticks([0, 1], labels=["Normal", "Pneumonia"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    threshold = cm.max() / 2
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", color="white" if cm[row, col] > threshold else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(filepath, dpi=170)
    plt.close(fig)


def plot_model_comparison(all_histories, filepath, metric):
    if metric not in {"accuracy", "f1"}:
        raise ValueError("metric must be accuracy or f1")
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, hist in all_histories.items():
        x = [int(h["epoch"]) for h in hist]
        # Training and validation appear for both architectures; test is reported as optional diagnostic.
        for part, linestyle in (("train", "-"), ("val", "--"), ("test", ":")):
            key = f"{part}_{metric}"
            ys = [float(h[key]) if h[key] is not None else float("nan") for h in hist]
            if not np.isnan(ys).all():
                ax.plot(x, ys, linestyle=linestyle, label=f"{model} {part}")
    ax.set(xlabel="Epoch", ylabel="Accuracy (%)" if metric == "accuracy" else "Pneumonia F1-score", title=f"ResNet18 vs ResNet50: {metric.upper()} comparison")
    ax.set_ylim(0, 100 if metric == "accuracy" else 1)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(filepath, dpi=170)
    plt.close(fig)
