from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models
from tqdm.auto import tqdm

from dataset import CLASS_NAMES, build_dataloaders, inverse_frequency_class_weights
from utils import (
    classification_metrics,
    plot_accuracy,
    plot_confusion_matrix,
    plot_f1_score,
    plot_model_comparison,
    save_history_csv,
    save_json,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    """Use torchvision ResNet backbones and REINITIALIZE the 2-class FC layer."""
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported architecture: {name}")
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(mode=training)
    epoch_loss = 0.0
    true_labels, pred_labels = [], []
    use_amp = device.type == "cuda"

    for images, labels in tqdm(loader, desc="Training" if training else "Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        epoch_loss += float(loss.detach().item()) * images.shape[0]
        predictions = logits.detach().argmax(dim=1)
        true_labels.extend(labels.detach().cpu().tolist())
        pred_labels.extend(predictions.cpu().tolist())

    metrics = classification_metrics(true_labels, pred_labels)
    metrics["loss"] = epoch_loss / len(loader.dataset)
    return metrics


def train_one_model(name, loaders, class_counts, args, device):
    print(f"\n{'=' * 15} {name.upper()} {'=' * 15}", flush=True)
    set_seed(args.seed)
    model = build_model(name, pretrained=not args.no_pretrained).to(device)
    class_weights = inverse_frequency_class_weights(class_counts["train"], device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model_dir = Path(args.output) / name
    model_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_f1 = -1.0
    best_val_acc = -1.0
    best_epoch = None
    ckpt_path = model_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"[{name}] Epoch {epoch}/{args.epochs}", flush=True)
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler)
        val_metrics = run_epoch(model, loaders["val"], criterion, device)
        # The assignment requests per-epoch TEST accuracy and F1 curves. These
        # diagnostics must NEVER be used to choose a checkpoint or tune settings.
        test_metrics = None if args.no_epoch_test else run_epoch(model, loaders["test"], criterion, device)

        entry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "test_loss": test_metrics["loss"] if test_metrics else None,
            "test_accuracy": test_metrics["accuracy"] if test_metrics else None,
            "test_f1": test_metrics["f1"] if test_metrics else None,
        }
        history.append(entry)
        print(
            f" Train acc={train_metrics['accuracy']:.2f}%, F1={train_metrics['f1']:.4f}"
            f" | Val acc={val_metrics['accuracy']:.2f}%, F1={val_metrics['f1']:.4f}"
            + (f" | Test diagnostic acc={test_metrics['accuracy']:.2f}%, F1={test_metrics['f1']:.4f}" if test_metrics else ""),
            flush=True,
        )
        # Selection criterion decided BEFORE observing test metrics.
        if (val_metrics["f1"] > best_val_f1 + 1e-12 or
            (abs(val_metrics["f1"] - best_val_f1) <= 1e-12 and val_metrics["accuracy"] > best_val_acc)):
            best_val_f1 = val_metrics["f1"]
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save({
                "model_name": name,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "class_names": list(CLASS_NAMES),
                "image_size": args.image_size,
                "seed": args.seed,
                "val_f1": best_val_f1,
                "val_accuracy": best_val_acc,
            }, ckpt_path)

    # Reload best-VALIDATION checkpoint to obtain the final TEST numbers.
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    final_test = run_epoch(model, loaders["test"], criterion, device)
    final_val = run_epoch(model, loaders["val"], criterion, device)
    highest_test_acc = max((h["test_accuracy"] for h in history if h["test_accuracy"] is not None), default=None)
    highest_test_f1 = max((h["test_f1"] for h in history if h["test_f1"] is not None), default=None)
    result = {
        "architecture": name,
        "best_epoch_by_validation_f1": best_epoch,
        "best_validation_f1": best_val_f1,
        "best_validation_accuracy": best_val_acc,
        "selected_model_test": final_test,
        "selected_model_validation": final_val,
        "highest_observed_epoch_test_accuracy_diagnostic": highest_test_acc,
        "highest_observed_epoch_test_f1_diagnostic": highest_test_f1,
        "checkpoint": str(ckpt_path.as_posix()),
        "important_note": "Model selection uses validation F1 only; per-epoch test numbers are diagnostic for the assignment, NOT for selecting the model.",
    }
    save_history_csv(model_dir / "history.csv", history)
    save_json(model_dir / "metrics.json", result)
    plot_accuracy(history, model_dir / "accuracy.png", name)
    plot_f1_score(history, model_dir / "f1.png", name)
    plot_confusion_matrix(final_test["confusion_matrix"], model_dir / "confusion_matrix.png", f"{name} - Final test confusion matrix")
    print(f"[{name}] FINAL selected epoch={best_epoch}, test accuracy={final_test['accuracy']:.2f}%, test F1={final_test['f1']:.4f}")
    if highest_test_acc is not None:
        print(f"[{name}] Highest observed epoch test accuracy={highest_test_acc:.2f}% and F1={highest_test_f1:.4f} (diagnostic only)")
    return history, result


def build_parser():
    parser = argparse.ArgumentParser(description="Lab1 X-ray pneumonia classification using torchvision ResNet")
    parser.add_argument("--dataset", required=True, help="Path containing train/NORMAL, train/PNEUMONIA, test/NORMAL and test/PNEUMONIA")
    parser.add_argument("--output", default="results")
    parser.add_argument("--models", nargs="+", default=["resnet18", "resnet50"], choices=["resnet18", "resnet50"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, help="0 is Windows-safe; try 2 or 4 later")
    parser.add_argument("--no-pretrained", action="store_true", help="For smoke tests only; official experiment uses pretrained")
    parser.add_argument("--no-epoch-test", action="store_true", help="Strict held-out test protocol; skips assignment-style test curves")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser


def main():
    args = build_parser().parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0 or args.image_size < 64:
        raise ValueError("epochs, batch-size, lr must be positive and image-size >= 64")
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != "cuda" and not args.cpu:
        print("WARNING: CUDA unavailable. Training ResNet50 on CPU can take a long time.")
    if not args.no_pretrained:
        print("Using ImageNet pretrained weights; torchvision may download them on first run.")

    loaders, counts = build_dataloaders(
        dataset_root=args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        use_cuda=device.type == "cuda",
    )
    print("Class counts (NORMAL=0; PNEUMONIA=1):")
    print(json.dumps(counts, indent=2))
    Path(args.output).mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["device_used"] = str(device)
    config["class_counts"] = counts
    save_json(Path(args.output) / "experiment_config.json", config)

    histories, summaries = {}, {}
    for model_name in args.models:
        history, result = train_one_model(model_name, loaders, counts, args, device)
        histories[model_name] = history
        summaries[model_name] = result

    plot_model_comparison(histories, Path(args.output) / "comparison_accuracy.png", "accuracy")
    plot_model_comparison(histories, Path(args.output) / "comparison_f1.png", "f1")
    save_json(Path(args.output) / "summary.json", summaries)
    with (Path(args.output) / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "selected_epoch_val_f1", "final_test_accuracy_percent", "final_test_f1", "final_test_precision", "final_test_recall", "highest_observed_test_acc_diagnostic", "highest_observed_test_f1_diagnostic"])
        for name, stats in summaries.items():
            test = stats["selected_model_test"]
            writer.writerow([name, stats["best_epoch_by_validation_f1"], round(test["accuracy"], 4), round(test["f1"], 5), round(test["precision"], 5), round(test["recall"], 5), stats["highest_observed_epoch_test_accuracy_diagnostic"], stats["highest_observed_epoch_test_f1_diagnostic"]])
    print("\nDone. Results saved to:", Path(args.output).resolve())
    print("Run build_report.py to generate bilingual PDFs using the real results.")


if __name__ == "__main__":
    main()
