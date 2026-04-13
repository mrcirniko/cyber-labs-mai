from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass(frozen=True)
class RunConfig:
    """Training settings shared by all experiments."""

    epochs: int = 6
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "none"
    patience: int = 0
    fast_dev_run: bool = False
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    seed: int = 42


def seed_everything(seed: int) -> None:
    """Make the experiment as reproducible as PyTorch allows."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_classifier(
    model: nn.Module,
    *,
    name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: list[str],
    config: RunConfig,
    device: torch.device | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train a classifier, keep the best validation checkpoint and evaluate on test."""

    seed_everything(config.seed)
    device = get_device() if device is None else device
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("The model has no trainable parameters.")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = _make_scheduler(optimizer, config)

    effective_epochs = 1 if config.fast_dev_run else config.epochs
    max_train_batches = config.max_train_batches
    max_eval_batches = config.max_eval_batches
    if config.fast_dev_run:
        max_train_batches = 2 if max_train_batches is None else max_train_batches
        max_eval_batches = 2 if max_eval_batches is None else max_eval_batches

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_gain = 0

    print(f"\n{name}: training on {device}")
    for epoch in range(1, effective_epochs + 1):
        started = time.time()
        train_loss, train_acc = _train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=max_train_batches,
        )
        val_metrics = evaluate_classifier(
            model,
            val_loader,
            criterion,
            class_names,
            device,
            max_batches=max_eval_batches,
        )
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{effective_epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            )
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1

        if config.patience > 0 and epochs_without_gain >= config.patience:
            print(f"early stop: validation macro F1 did not improve for {config.patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate_classifier(
        model,
        val_loader,
        criterion,
        class_names,
        device,
        max_batches=max_eval_batches,
    )
    test_metrics = evaluate_classifier(
        model,
        test_loader,
        criterion,
        class_names,
        device,
        max_batches=max_eval_batches,
    )

    result = {
        "name": name,
        "best_epoch": best_epoch,
        "history": history,
        "val": val_metrics,
        "test": test_metrics,
        "class_names": class_names,
        "config": config,
    }
    if output_dir is not None:
        _save_outputs(result, model, Path(output_dir), name)
    return result


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    class_names: list[str],
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate a classifier and return aggregate metrics plus raw labels."""

    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    top3_correct = 0
    total = 0

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            losses.append(loss.item())

            predictions = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
            total += labels.numel()

            k = min(3, logits.shape[1])
            topk = logits.topk(k=k, dim=1).indices
            top3_correct += topk.eq(labels.view(-1, 1)).any(dim=1).sum().item()

    labels = list(range(len(class_names)))
    accuracy = accuracy_score(y_true, y_pred)
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    micro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="micro", zero_division=0
    )

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(accuracy),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "micro_precision": float(micro[0]),
        "micro_recall": float(micro[1]),
        "micro_f1": float(micro[2]),
        "top3_accuracy": float(top3_correct / total) if total else float("nan"),
        "y_true": y_true,
        "y_pred": y_pred,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def summarize_results(results: list[dict[str, Any]], *, split: str = "test") -> pd.DataFrame:
    """Build a comparison table for experiment results."""

    rows = []
    for result in results:
        metrics = result[split]
        rows.append(
            {
                "model": result["name"],
                "best_epoch": result.get("best_epoch", None),
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "top3_accuracy": metrics["top3_accuracy"],
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def print_report(result: dict[str, Any], *, split: str = "test") -> None:
    """Print sklearn classification report for a trained experiment."""

    print(f"{result['name']} | {split} classification report")
    print(result[split]["classification_report"])


def make_conclusion(results: list[dict[str, Any]], *, split: str = "test") -> str:
    """Generate a short metric-based conclusion after experiments are run."""

    table = summarize_results(results, split=split)
    best = table.iloc[0]
    worst = table.iloc[-1]
    return (
        f"Лучший результат по {split} macro F1 показала модель {best['model']} "
        f"с macro F1={best['macro_f1']:.4f} и accuracy={best['accuracy']:.4f}. "
        f"Разница с наименее успешной конфигурацией в этой таблице составляет "
        f"{best['macro_f1'] - worst['macro_f1']:.4f} по macro F1. "
        "Для итогового сравнения важнее macro F1, потому что он одинаково учитывает все 36 классов."
    )


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_batches: int | None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    batches = 0

    for batch_index, (images, labels) in enumerate(tqdm(loader, leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
        batches += 1

    return total_loss / max(batches, 1), correct / max(total, 1)


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: RunConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.scheduler == "none":
        return None
    if config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(config.epochs // 2, 1), gamma=0.35)
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    raise ValueError(f"Unknown scheduler: {config.scheduler}")


def _save_outputs(
    result: dict[str, Any],
    model: nn.Module,
    output_dir: Path,
    name: str,
) -> None:
    safe_name = name.lower().replace(" ", "_").replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / f"{safe_name}.pt")
    summarize_results([result]).to_csv(output_dir / f"{safe_name}_metrics.csv", index=False)
