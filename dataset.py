from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASS_NAMES = ("NORMAL", "PNEUMONIA")
EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ChestXrayDataset(Dataset):
    """User-defined Dataset for the Kaggle chest_xray/{train,test}/CLASS folders."""

    def __init__(self, samples: Sequence[Tuple[Path, int]], transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as source:
            # Apply EXIF orientation before ensuring 3 channels for ImageNet ResNet.
            image = ImageOps.exif_transpose(source).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def scan_class_folders(split_dir: Path) -> List[Tuple[Path, int]]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")
    samples: List[Tuple[Path, int]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        class_files = sorted(
            p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS
        )
        if not class_files:
            raise ValueError(f"No X-ray images found in: {class_dir}")
        samples.extend((path, class_id) for path in class_files)
    return samples


def stratified_train_val_split(
    samples: Sequence[Tuple[Path, int]], val_fraction: float, seed: int
):
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5.")
    by_class = {label: [] for label in range(len(CLASS_NAMES))}
    for sample in samples:
        by_class[sample[1]].append(sample)

    rng = random.Random(seed)
    train_samples, val_samples = [], []
    for class_id, class_samples in by_class.items():
        if len(class_samples) < 2:
            raise ValueError(f"At least two {CLASS_NAMES[class_id]} samples are needed.")
        rng.shuffle(class_samples)
        num_val = max(1, int(round(len(class_samples) * val_fraction)))
        val_samples.extend(class_samples[:num_val])
        train_samples.extend(class_samples[num_val:])
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def image_transform(image_size: int, augment: bool):
    base = [transforms.Resize((image_size, image_size))]
    if augment:
        # Mild geometric augmentation; do not rotate by 90 degrees or flip anatomy.
        base.append(transforms.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.97, 1.03)))
    return transforms.Compose(base + [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_dataloaders(
    dataset_root: str,
    batch_size: int = 32,
    image_size: int = 224,
    val_fraction: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    use_cuda: bool = False,
):
    root = Path(dataset_root).expanduser().resolve()
    train_all = scan_class_folders(root / "train")
    test_samples = scan_class_folders(root / "test")
    train_samples, val_samples = stratified_train_val_split(train_all, val_fraction, seed)
    train_set = ChestXrayDataset(train_samples, image_transform(image_size, augment=True))
    val_set = ChestXrayDataset(val_samples, image_transform(image_size, augment=False))
    test_set = ChestXrayDataset(test_samples, image_transform(image_size, augment=False))

    def loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=num_workers > 0,
        )

    loaders = {"train": loader(train_set, True), "val": loader(val_set, False), "test": loader(test_set, False)}
    class_counts = {}
    for name, split_samples in (("train", train_samples), ("val", val_samples), ("test", test_samples)):
        class_counts[name] = {cls: sum(label == idx for _, label in split_samples) for idx, cls in enumerate(CLASS_NAMES)}
    # Never use test data or the tiny original Kaggle val folder for training.
    return loaders, class_counts


def inverse_frequency_class_weights(counts: dict, device: torch.device):
    """Inverse-frequency loss weights, normalized to mean 1 (train partition only)."""
    frequencies = torch.tensor([counts[name] for name in CLASS_NAMES], dtype=torch.float32)
    if (frequencies <= 0).any():
        raise ValueError("Both training classes must have nonzero count.")
    weights = 1.0 / frequencies
    return (weights / weights.mean()).to(device)
