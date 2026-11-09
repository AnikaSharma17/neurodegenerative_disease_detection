import json

import pytest

import neurodx.config as config
from neurodx import train


@pytest.mark.parametrize("arch", ["multitask", "unet"])
def test_one_epoch_end_to_end(tiny_dataset, tmp_path, monkeypatch, arch):
    root, cfg = tiny_dataset
    monkeypatch.setitem(config.DISEASES, "alzheimer", cfg)
    monkeypatch.setitem(train.DISEASES, "alzheimer", cfg)
    out = tmp_path / "runs"
    result = train.main([
        "--disease", "alzheimer", "--arch", arch, "--backbone", "mobilenet_v2",
        "--epochs", "1", "--batch-size", "4", "--image-size", "64", "--num-workers", "0",
        "--val-frac", "0.3", "--test-frac", "0.3", "--no-pretrained", "--device", "cpu",
        "--data-root", str(root), "--out-dir", str(out), "--class-weighting", "--bootstrap", "50",
    ])
    run = out / "alzheimer" / f"{arch}_mobilenet_v2_seg1_group_seed42"
    for f in ("split.csv", "history.csv", "best.pt", "test_metrics.json", "test_predictions.csv",
              "confusion_matrix.png", "curves.png", "calibration.png"):
        assert (run / f).is_file(), f
    saved = json.loads((run / "test_metrics.json").read_text())
    n_test = result["split_sizes"]["test"]
    assert saved["test"]["n"] == n_test
    assert sum(map(sum, saved["test_confusion_matrix"])) == n_test
    assert saved["test_patient_level"]["n"] == result["split_groups"]["test"]
    lo, hi = saved["test_ci95"]["accuracy"]
    assert 0 <= lo <= hi <= 1
    assert saved["calibration"]["temperature"] > 0


def test_image_level_split_is_explicit(tiny_dataset, tmp_path, monkeypatch):
    root, cfg = tiny_dataset
    monkeypatch.setitem(train.DISEASES, "alzheimer", cfg)
    result = train.main([
        "--disease", "alzheimer", "--backbone", "mobilenet_v2", "--epochs", "1", "--batch-size", "8",
        "--image-size", "32", "--num-workers", "0", "--no-pretrained", "--device", "cpu", "--seg-weight", "0",
        "--split-level", "image", "--bootstrap", "0", "--data-root", str(root), "--out-dir", str(tmp_path),
    ])
    assert result["split_groups"]["test"] == result["split_sizes"]["test"]  # one group per image
