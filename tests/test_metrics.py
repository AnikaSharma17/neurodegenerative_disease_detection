import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix as sk_cm

from neurodx.metrics import classification_metrics, confusion_matrix, segmentation_metrics


def test_classification_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 500)
    p = np.where(rng.random(500) < 0.7, y, rng.integers(0, 4, 500))
    cm = confusion_matrix(torch.tensor(p), torch.tensor(y), 4)
    assert np.array_equal(cm.numpy(), sk_cm(y, p, labels=range(4)))
    m = classification_metrics(cm, ["a", "b", "c", "d"])
    assert m["n"] == 500
    assert np.isclose(m["accuracy"], (p == y).mean())
    assert np.isclose(m["balanced_accuracy"], balanced_accuracy_score(y, p))
    assert np.isclose(m["macro_f1"], f1_score(y, p, average="macro"))
    assert np.isclose(m["macro_precision"], precision_score(y, p, average="macro", zero_division=0))
    assert np.isclose(m["macro_recall"], recall_score(y, p, average="macro"))


def test_segmentation_metrics_known_values():
    target = torch.tensor([[0, 0, 1, 1]])
    pred = torch.tensor([[0, 1, 1, 1]])
    s = segmentation_metrics(confusion_matrix(pred, target, 2))
    assert np.isclose(s["fg_iou"], 2 / 3)
    assert np.isclose(s["fg_dice"], 0.8)


def test_all_background_is_not_reported_as_perfect_foreground():
    z = torch.zeros(1, 8, dtype=torch.long)
    s = segmentation_metrics(confusion_matrix(z, z, 2))
    assert np.isnan(s["fg_iou"])  # the old pipeline reported 1.0 here
