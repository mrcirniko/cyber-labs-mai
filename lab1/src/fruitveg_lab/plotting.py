from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from .data import IMAGENET_MEAN, IMAGENET_STD


def show_image_grid(
    loader: torch.utils.data.DataLoader,
    class_names: list[str],
    *,
    count: int = 12,
    normalized: bool = True,
) -> None:
    """Show a grid of images from a DataLoader batch."""

    images, labels = next(iter(loader))
    count = min(count, len(images))
    columns = min(4, count)
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for index in range(count):
        image = images[index].detach().cpu()
        if normalized:
            image = _unnormalize(image)
        image = image.permute(1, 2, 0).numpy().clip(0, 1)
        axes[index].imshow(image)
        axes[index].set_title(class_names[int(labels[index])])
        axes[index].axis("off")

    for axis in axes[count:]:
        axis.axis("off")
    plt.tight_layout()
    plt.show()


def plot_history(result: dict[str, Any]) -> None:
    """Plot train/validation loss and accuracy curves."""

    history = pd.DataFrame(result["history"])
    if history.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation")
    axes[0].set_title(f"{result['name']} loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_accuracy"], label="train accuracy")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="validation accuracy")
    axes[1].plot(history["epoch"], history["val_macro_f1"], label="validation macro F1")
    axes[1].set_title(f"{result['name']} quality")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(result: dict[str, Any], *, split: str = "test") -> None:
    """Plot a normalized confusion matrix."""

    matrix = np.asarray(result[split]["confusion_matrix"], dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        normalized,
        cmap="viridis",
        xticklabels=result["class_names"],
        yticklabels=result["class_names"],
        square=True,
        cbar_kws={"label": "share"},
    )
    plt.title(f"{result['name']} normalized confusion matrix ({split})")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.show()


def _unnormalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return image * std + mean

