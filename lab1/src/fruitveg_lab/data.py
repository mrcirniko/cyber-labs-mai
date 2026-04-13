from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable
import warnings

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRUIT_VEGETABLE_KAGGLE_SLUG = "kritikseth/fruit-and-vegetable-image-recognition"
PIL_PALETTE_TRANSPARENCY_WARNING = (
    "Palette images with Transparency expressed in bytes should be converted to RGBA images"
)

warnings.filterwarnings(
    "ignore",
    message=PIL_PALETTE_TRANSPARENCY_WARNING,
    category=UserWarning,
    module="PIL.Image",
)


@dataclass(frozen=True)
class DatasetInfo:
    """Compact description of an ImageFolder-style dataset."""

    root: Path
    classes: list[str]
    split_counts: Dict[str, Dict[str, int]]


@dataclass(frozen=True)
class ImageLoaders:
    """Train, validation and test loaders with their source datasets."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_dataset: datasets.ImageFolder
    val_dataset: datasets.ImageFolder
    test_dataset: datasets.ImageFolder
    class_names: list[str]


def resolve_dataset_root(candidate: str | Path) -> Path:
    """Find the directory containing train, validation and test folders."""

    candidate = Path(candidate).expanduser().resolve()
    search_roots = list(
        dict.fromkeys(
            [
                candidate,
                candidate / "fruit-and-vegetable-image-recognition",
                candidate / "data",
                candidate / "data" / "fruit-and-vegetable-image-recognition",
            ]
        )
    )

    for root in search_roots:
        if _has_required_splits(root):
            return root

    for base in search_roots:
        if not base.is_dir():
            continue
        for root in _iter_nested_dirs(base, max_depth=3):
            if _has_required_splits(root):
                return root

    checked = "\n".join(str(path) for path in search_roots)
    raise FileNotFoundError(
        "Dataset root was not found. Checked:\n"
        f"{checked}\n\n"
        "Expected folders: train/, validation/ and test/."
    )


def ensure_kaggle_dataset(
    download_dir: str | Path,
    *,
    dataset_slug: str = FRUIT_VEGETABLE_KAGGLE_SLUG,
    force: bool = False,
) -> Path:
    """Download and unzip the Kaggle dataset if it is not already available."""

    download_dir = Path(download_dir).expanduser().resolve()
    if not force:
        try:
            return resolve_dataset_root(download_dir)
        except FileNotFoundError:
            pass

    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "The kaggle package is required for automatic dataset download. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except OSError as exc:
        raise RuntimeError(
            "Kaggle credentials were not found. Place kaggle.json into ~/.kaggle/kaggle.json "
            "or download the dataset manually into the data directory."
        ) from exc

    api.dataset_download_files(dataset_slug, path=str(download_dir), unzip=True, quiet=False)
    return resolve_dataset_root(download_dir)


def describe_dataset(dataset_root: str | Path) -> DatasetInfo:
    """Count images per split and class."""

    root = resolve_dataset_root(dataset_root)
    split_counts = {
        split: count_images_by_class(_split_dir(root, split))
        for split in ("train", "validation", "test")
    }
    classes = sorted(split_counts["train"])
    return DatasetInfo(root=root, classes=classes, split_counts=split_counts)


def count_images_by_class(split_dir: str | Path) -> Dict[str, int]:
    """Return image count for every class directory inside a split."""

    split_dir = Path(split_dir)
    counts: Dict[str, int] = {}
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        counts[class_dir.name] = sum(
            1
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return counts


def make_transforms(
    image_size: int,
    *,
    mode: str,
    normalize: bool = True,
) -> transforms.Compose:
    """Create preprocessing or augmentation transforms."""

    if mode == "plain":
        steps: list[object] = [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        ]
    elif mode == "augmented":
        steps = [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.72, 1.0),
                ratio=(0.85, 1.15),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12, interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.14, hue=0.03),
        ]
    else:
        raise ValueError(f"Unknown transform mode: {mode!r}")

    steps.append(transforms.ToTensor())
    if normalize:
        steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(steps)


def create_dataloaders(
    dataset_root: str | Path,
    *,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool | None = None,
) -> ImageLoaders:
    """Build ImageFolder datasets and DataLoader objects."""

    root = resolve_dataset_root(dataset_root)
    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory

    train_dataset = datasets.ImageFolder(
        _split_dir(root, "train"),
        transform=train_transform,
        loader=rgb_image_loader,
    )
    val_dataset = datasets.ImageFolder(
        _split_dir(root, "validation"),
        transform=eval_transform,
        loader=rgb_image_loader,
    )
    test_dataset = datasets.ImageFolder(
        _split_dir(root, "test"),
        transform=eval_transform,
        loader=rgb_image_loader,
    )

    _validate_class_alignment(train_dataset, val_dataset, test_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return ImageLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        class_names=list(train_dataset.classes),
    )


def rgb_image_loader(path: str | Path) -> Image.Image:
    """Load an image as RGB while hiding harmless PIL palette warnings."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=PIL_PALETTE_TRANSPARENCY_WARNING,
            category=UserWarning,
            module="PIL.Image",
        )
        with open(path, "rb") as file:
            image = Image.open(file)
            return image.convert("RGB")


def _has_required_splits(root: Path) -> bool:
    return all(_split_dir(root, split).is_dir() for split in ("train", "validation", "test"))


def _iter_nested_dirs(base: Path, *, max_depth: int) -> Iterable[Path]:
    base_depth = len(base.parts)
    for path in base.rglob("*"):
        if path.is_dir() and len(path.parts) - base_depth <= max_depth:
            yield path


def _split_dir(root: Path, split: str) -> Path:
    if split == "validation":
        validation = root / "validation"
        return validation if validation.exists() else root / "val"
    return root / split


def _validate_class_alignment(*image_folders: datasets.ImageFolder) -> None:
    reference = image_folders[0].classes
    for folder in image_folders[1:]:
        if folder.classes != reference:
            raise ValueError(
                "Class folders differ between dataset splits. "
                f"Expected {reference}, got {folder.classes}."
            )


def total_images(counts: Dict[str, int] | Iterable[int]) -> int:
    """Return total number of images for a split or a count iterable."""

    if isinstance(counts, dict):
        return sum(counts.values())
    return sum(counts)
