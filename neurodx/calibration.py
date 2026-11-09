"""Temperature scaling, calibration error and selective prediction."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """Single scalar T minimising validation NLL of softmax(logits / T) (Guo et al., 2017)."""
    logits, labels = logits.float().cpu(), labels.long().cpu()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def risk_coverage(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Sort by confidence; error rate among the k most confident predictions.
    AURC = mean risk over all coverages (lower is better)."""
    order = np.argsort(-probs.max(1), kind="stable")
    errors = (probs.argmax(1) != labels)[order].astype(np.float64)
    k = np.arange(1, len(errors) + 1)
    risk = np.cumsum(errors) / k
    coverage = k / len(errors)
    out = {"aurc": float(risk.mean()), "coverage": coverage.tolist(), "risk": risk.tolist()}
    for target in (0.5, 0.8, 0.9):
        out[f"risk_at_{int(target * 100)}pct_coverage"] = float(risk[max(int(target * len(risk)) - 1, 0)])
    return out
