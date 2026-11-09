"""Confusion-matrix based metrics (no torchmetrics dependency)."""
from __future__ import annotations

import numpy as np
import torch


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Rows = true class, columns = predicted class."""
    idx = targets.reshape(-1).long() * num_classes + preds.reshape(-1).long()
    return torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes).cpu()


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def classification_metrics(cm: torch.Tensor, class_names: list[str] | tuple[str, ...]) -> dict:
    m = cm.numpy().astype(np.float64)
    tp = np.diag(m)
    support = m.sum(1)
    predicted = m.sum(0)
    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, support)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    f1 = np.where(np.isnan(f1) & (support > 0), 0.0, f1)
    present = support > 0
    return {
        "n": int(m.sum()),
        "accuracy": float(tp.sum() / m.sum()) if m.sum() else float("nan"),
        "balanced_accuracy": float(np.nanmean(recall[present])) if present.any() else float("nan"),
        "macro_precision": float(np.nanmean(np.nan_to_num(precision[present]))) if present.any() else float("nan"),
        "macro_recall": float(np.nanmean(recall[present])) if present.any() else float("nan"),
        "macro_f1": float(np.mean(f1[present])) if present.any() else float("nan"),
        "per_class": {
            name: {
                "precision": None if np.isnan(precision[i]) else float(precision[i]),
                "recall": None if np.isnan(recall[i]) else float(recall[i]),
                "f1": None if np.isnan(f1[i]) else float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
    }


def segmentation_metrics(cm: torch.Tensor) -> dict:
    """IoU / Dice per class; ``fg_*`` is class 1 (the pseudo-mask foreground)."""
    m = cm.numpy().astype(np.float64)
    tp = np.diag(m)
    fp = m.sum(0) - tp
    fn = m.sum(1) - tp
    iou = _safe_div(tp, tp + fp + fn)
    dice = _safe_div(2 * tp, 2 * tp + fp + fn)
    return {
        "mean_iou": float(np.nanmean(iou)) if not np.all(np.isnan(iou)) else float("nan"),
        "fg_iou": float(iou[1]) if len(iou) > 1 and not np.isnan(iou[1]) else float("nan"),
        "fg_dice": float(dice[1]) if len(dice) > 1 and not np.isnan(dice[1]) else float("nan"),
    }


def aggregate_by_group(probs: np.ndarray, labels: np.ndarray, groups: list[str]):
    """Patient-level prediction: average slice probabilities within each group.
    Returns (group_probs, group_labels, group_names)."""
    names = sorted(set(groups))
    index = {g: i for i, g in enumerate(names)}
    gi = np.array([index[g] for g in groups])
    gp = np.zeros((len(names), probs.shape[1]))
    np.add.at(gp, gi, probs)
    gp /= np.bincount(gi, minlength=len(names))[:, None]
    gl = np.zeros(len(names), dtype=int)
    gl[gi] = labels  # groups are single-class by construction
    return gp, gl, names


def group_bootstrap_ci(
    labels: np.ndarray, preds: np.ndarray, groups: list[str], num_classes: int,
    n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> dict:
    """95% CIs for accuracy / balanced accuracy / macro-F1, resampling whole groups
    (patients), because slices of one patient are not independent samples."""
    rng = np.random.default_rng(seed)
    names = sorted(set(groups))
    members = {g: [] for g in names}
    for i, g in enumerate(groups):
        members[g].append(i)
    members_arr = [np.array(members[g]) for g in names]
    keys = ("accuracy", "balanced_accuracy", "macro_f1")
    stats = {k: [] for k in keys}
    dummy_names = [str(c) for c in range(num_classes)]
    for _ in range(n_boot):
        pick = rng.integers(0, len(names), len(names))
        idx = np.concatenate([members_arr[j] for j in pick])
        cm = confusion_matrix(torch.from_numpy(preds[idx]), torch.from_numpy(labels[idx]), num_classes)
        m = classification_metrics(cm, dummy_names)
        for k in keys:
            stats[k].append(m[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))] for k, v in stats.items()}
