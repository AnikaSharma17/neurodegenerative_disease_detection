"""Headless plotting helpers (never call plt.show())."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def plot_confusion_matrix(cm: torch.Tensor, class_names, path: str | Path, title: str = "") -> None:
    m = cm.numpy()
    fig, ax = plt.subplots(figsize=(1.6 * len(class_names) + 3, 1.4 * len(class_names) + 2))
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    thresh = m.max() / 2 if m.size else 0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, int(m[i, j]), ha="center", va="center",
                    color="white" if m[i, j] > thresh else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_history(history: list[dict], path: str | Path) -> None:
    ep = [h["epoch"] for h in history]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
    a.plot(ep, [h["train_loss"] for h in history], label="train")
    a.plot(ep, [h["val_loss"] for h in history], label="val")
    a.set_title("Loss")
    a.set_xlabel("epoch")
    a.legend()
    for key in ("train_macro_f1", "val_macro_f1", "val_bal_acc", "val_fg_dice"):
        if key in history[0]:
            b.plot(ep, [h[key] for h in history], label=key)
    b.set_title("Metrics")
    b.set_xlabel("epoch")
    b.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _reliability(probs, labels, n_bins=10):
    conf, correct = probs.max(1), probs.argmax(1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
    return xs, ys


def plot_calibration(probs_raw, probs_cal, labels, rc: dict, path: str | Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    a.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    a.plot(*_reliability(probs_raw, labels), "o-", label="uncalibrated")
    a.plot(*_reliability(probs_cal, labels), "s-", label="temperature-scaled")
    a.set_xlabel("confidence")
    a.set_ylabel("accuracy")
    a.set_title("Reliability (test)")
    a.legend()
    b.plot(rc["coverage"], rc["risk"])
    b.set_xlabel("coverage (fraction of most-confident predictions kept)")
    b.set_ylabel("error rate")
    b.set_title(f"Risk-coverage (AURC {rc['aurc']:.4f})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_cam_gallery(tiles, rows_meta, class_names, path: str | Path) -> None:
    """tiles: [[PIL images] correct, [PIL images] wrong]; rows_meta: matching prediction rows."""
    n = max((len(t) for t in tiles), default=0)
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 6), squeeze=False)
    for r, (row_tiles, metas, title) in enumerate(zip(tiles, rows_meta, ("correct", "misclassified"))):
        for c in range(n):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(row_tiles):
                m = metas[c]
                ax.imshow(row_tiles[c])
                ax.set_title(f"true {class_names[int(m['label'])]}\npred {class_names[int(m['pred'])]}", fontsize=8)
        axes[r][0].text(-0.15, 0.5, title, transform=axes[r][0].transAxes, rotation=90, va="center", ha="right")
    fig.suptitle("Grad-CAM on test slices (red = most influential)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
