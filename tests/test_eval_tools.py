import csv

import numpy as np
import torch
from PIL import Image

from neurodx.calibration import expected_calibration_error, fit_temperature, risk_coverage
from neurodx.config import DiseaseConfig
from neurodx.data import collect_samples
from neurodx.grouping import components, similarity_matrix, validate
from neurodx.metrics import aggregate_by_group, group_bootstrap_ci


def test_temperature_reduces_overconfidence():
    torch.manual_seed(0)
    labels = torch.randint(0, 3, (2000,))
    logits = torch.randn(2000, 3)
    logits[torch.arange(2000), labels] += 1.0
    logits *= 6  # same ranking, wildly overconfident
    t = fit_temperature(logits, labels)
    assert t > 2
    before = expected_calibration_error(torch.softmax(logits, 1).numpy(), labels.numpy())
    after = expected_calibration_error(torch.softmax(logits / t, 1).numpy(), labels.numpy())
    assert after < before


def test_risk_coverage_orders_by_confidence():
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.55, 0.45]])
    labels = np.array([0, 0, 1])  # only the least confident is wrong
    rc = risk_coverage(probs, labels)
    assert rc["risk"] == [0.0, 0.0, 1 / 3]


def test_patient_aggregation_and_grouped_bootstrap():
    probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]])
    labels = np.array([0, 0, 1, 1])
    groups = ["a", "a", "b", "b"]
    gp, gl, names = aggregate_by_group(probs, labels, groups)
    assert names == ["a", "b"]
    assert np.allclose(gp[0], [0.65, 0.35])
    assert gl.tolist() == [0, 1]
    ci = group_bootstrap_ci(labels, probs.argmax(1), groups, 2, n_boot=200)
    assert 0 <= ci["accuracy"][0] <= ci["accuracy"][1] <= 1


def _brain(patient: int, slice_noise_seed: int) -> np.ndarray:
    """Smooth patient-specific pattern + a little per-slice noise."""
    base = np.kron(np.random.default_rng(patient).normal(0, 1, (8, 8)), np.ones((8, 8)))
    return base + 0.1 * np.random.default_rng(1000 + slice_noise_seed).normal(0, 1, base.shape)


def test_similarity_grouping_recovers_patients(tmp_path):
    names = []
    for pid in (1, 2, 3):
        for sl in (26, 27, 28):
            arr = _brain(pid, pid * 100 + sl)
            arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
            name = f"{sl} ({pid}).png"
            Image.fromarray(arr).save(tmp_path / name)
            names.append(name)
    sim = similarity_matrix([tmp_path / n for n in names], size=(64, 64), device="cpu")
    labels = components(sim, 0.9)
    assert len(set(labels)) == 3
    v = validate(names, labels)
    assert v["same_patient_pairs_kept_together"] == 1.0
    assert v["co_clustered_pairs_same_patient"] == 1.0


def test_groups_file_overrides_regex(tiny_dataset):
    root, cfg = tiny_dataset
    gfile = root / "groups.csv"
    with gfile.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "group"])
        for cls in cfg.classes:
            for p in sorted((root / cfg.image_root / cls).iterdir()):
                w.writerow([f"{cls}/{p.name}", f"{cls}/patient"])
    cfg2 = DiseaseConfig(**{**cfg.__dict__, "groups_file": "groups.csv"})
    assert {s.group for s in collect_samples(cfg2, root)} == {"a/patient", "b/patient"}
    assert len({s.group for s in collect_samples(cfg2, root, split_level="image")}) == 24
