import json

import torch

import neurodx.config as config
from neurodx import mil, train
from neurodx.data import check_no_group_leakage, collect_samples, kfold_split


def test_kfold_tests_every_group_exactly_once(tiny_dataset):
    root, cfg = tiny_dataset
    samples = collect_samples(cfg, root)
    seen = []
    for fold in range(3):
        sp = kfold_split(samples, folds=3, fold=fold, val_frac=0.2, seed=0)
        check_no_group_leakage(sp)
        assert sum(len(v) for v in sp.values()) == len(samples)
        assert sp["train"] and sp["test"]
        seen += sorted({s.group for s in sp["test"]})
    assert sorted(seen) == sorted({s.group for s in samples})


def test_attention_mil_shapes():
    model = mil.AttentionMIL(3, "mobilenet_v2", pretrained=False).eval()
    with torch.no_grad():
        logits, attn = model(torch.randn(2, 5, 3, 64, 64))
    assert logits.shape == (2, 3)
    assert attn.shape == (2, 5)
    assert torch.allclose(attn.sum(1), torch.ones(2))


def test_mil_cv_fold_end_to_end(tiny_dataset, tmp_path, monkeypatch):
    root, cfg = tiny_dataset
    monkeypatch.setitem(config.DISEASES, "alzheimer", cfg)
    monkeypatch.setitem(mil.DISEASES, "alzheimer", cfg)
    out = tmp_path / "runs"
    result = mil.main([
        "--disease", "alzheimer", "--backbone", "mobilenet_v2", "--epochs", "1", "--bags-per-batch", "2",
        "--bag-size", "3", "--image-size", "64", "--num-workers", "0", "--no-pretrained", "--device", "cpu",
        "--cv-folds", "3", "--fold", "1", "--val-frac", "0.2", "--bootstrap", "20",
        "--data-root", str(root), "--out-dir", str(out),
    ])
    run = out / "alzheimer" / "mil_mobilenet_v2_seed42_cv3f1"
    for f in ("split.csv", "best.pt", "test_metrics.json", "test_predictions.csv", "curves.png"):
        assert (run / f).is_file(), f
    saved = json.loads((run / "test_metrics.json").read_text())
    assert saved["test_patient_level"]["n"] == result["split_groups"]["test"] == 2
    assert saved["fold"] == 1


def test_slice_model_cv_run_name(tiny_dataset, tmp_path, monkeypatch):
    root, cfg = tiny_dataset
    monkeypatch.setitem(train.DISEASES, "alzheimer", cfg)
    train.main([
        "--disease", "alzheimer", "--backbone", "mobilenet_v2", "--epochs", "1", "--batch-size", "8",
        "--image-size", "32", "--num-workers", "0", "--no-pretrained", "--device", "cpu", "--seg-weight", "0",
        "--cv-folds", "3", "--fold", "2", "--val-frac", "0.2", "--bootstrap", "0",
        "--data-root", str(root), "--out-dir", str(tmp_path),
    ])
    assert (tmp_path / "alzheimer" / "multitask_mobilenet_v2_seg0_group_seed42_cv3f2" / "test_metrics.json").is_file()
