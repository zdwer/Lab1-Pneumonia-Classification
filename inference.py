from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader

from dataset import ChestXrayDataset, CLASS_NAMES, image_transform, scan_class_folders
from train import build_model
from utils import classification_metrics, plot_confusion_matrix, save_json


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_model(checkpoint["model_name"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def predict_image(image_path, model, transform, device):
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    probabilities = model(image_tensor).softmax(dim=1)[0].cpu().tolist()
    prediction = int(max(range(len(probabilities)), key=lambda index: probabilities[index]))
    return {"image": str(image_path), "class": CLASS_NAMES[prediction], "pneumonia_probability": probabilities[1], "normal_probability": probabilities[0]}


@torch.inference_mode()
def evaluate_folder(test_dir, model, transform, device, batch_size):
    samples = scan_class_folders(Path(test_dir))
    loader = DataLoader(ChestXrayDataset(samples, transform), batch_size=batch_size, shuffle=False)
    labels, predictions = [], []
    for images, batch_labels in loader:
        predicted = model(images.to(device)).argmax(dim=1).cpu().tolist()
        labels.extend(batch_labels.tolist())
        predictions.extend(predicted)
    return classification_metrics(labels, predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="e.g. results/resnet18/best.pt")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--image", help="Single .jpeg image to predict")
    target.add_argument("--dataset", help="Dataset root with test/NORMAL and test/PNEUMONIA")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="inference_results")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(args.checkpoint, device)
    transform = image_transform(meta["image_size"], augment=False)

    if args.image:
        result = predict_image(args.image, model, transform, device)
    else:
        result = evaluate_folder(Path(args.dataset) / "test", model, transform, device, args.batch_size)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(result["confusion_matrix"], output_dir / "confusion_matrix.png", "Held-out test confusion matrix")
        save_json(output_dir / "metrics.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
