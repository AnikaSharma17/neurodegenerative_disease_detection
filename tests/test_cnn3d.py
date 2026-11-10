import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
import prepare_volumes as pv  # noqa: E402

from neurodx import cnn3d  # noqa: E402
from neurodx.config import DISEASES, DiseaseConfig  # noqa: E402


def _subject(seed: int, brain_radius: float, shape=(40, 48, 36), zooms=(2.0, 2.0, 2.0)):
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(*[np.arange(n) - n / 2 for n in shape], indexing="ij"))
    r = np.sqrt(((grid * np.array(zooms)[:, None, None, None]) ** 2).sum(0))
    seg = np.where(r < brain_radius, 2, 0).astype(np.int16)
    seg[(r < brain_radius * 0.3)] = 4  # ventricle in the middle
    vol = rng.normal(5, 1, shape).astype(np.float32) + 100 * (seg > 0)
    return vol, seg, np.diag([*zooms, 1.0])


def test_fixed_grid_keeps_physical_size():
    small, seg_s, aff = _subject(0, 20)
    big, seg_b, _ = _subject(1, 30)
    a, ma = pv.to_fixed_grid(small, (2.0, 2.0, 2.0), seg_s, 32, 96)
    b, mb = pv.to_fixed_grid(big, (2.0, 2.0, 2.0), seg_b, 32, 96)
    assert a.shape == b.shape == ma.shape == (32, 32, 32) and a.dtype == np.uint8
    assert (mb > 0).sum() > 2 * (ma > 0).sum()  # no per-subject rescaling


def test_oasis_labels_from_txt(tmp_path):
    for sid, cdr, age in (("OAS1_0001_MR1", "0", "74"), ("OAS1_0002_MR1", "1", "80"), ("OAS1_0003_MR1", "", "22")):
        d = tmp_path / sid
        vol, _, aff = _subject(0, 20)
        (d / "PROCESSED/MPRAGE/T88_111").mkdir(parents=True)
        nib.save(nib.AnalyzeImage(vol, aff), str(d / "PROCESSED/MPRAGE/T88_111" / f"{sid}_mpr_n4_anon_111_t88_gfc.img"))
        nib.save(nib.AnalyzeImage(vol, aff),
                 str(d / "PROCESSED/MPRAGE/T88_111" / f"{sid}_mpr_n4_anon_111_t88_masked_gfc.img"))
        (d / f"{sid}.txt").write_text(f"SESSION ID:     {sid}\nAGE:            {age}\nM/F:            Female\n"
                                      f"CDR:            {cdr}\nnWBV:           0.74\n")
    scans = pv.oasis1_scans(tmp_path)
    assert [(s.subject, s.label, s.info["Age"]) for s in scans] == [
        ("OAS1_0001", "NonDemented", "74"), ("OAS1_0002", "MildDemented", "80")]
    assert scans[0].path.name.endswith("masked_gfc.img")
    assert pv.oasis1_scans(tmp_path, skull=True)[0].path.name.endswith("_t88_gfc.img")
    pv.main(["oasis1", "--src", str(tmp_path), "--dataset", "oasis", "--slices", "3", "--volume-size", "16",
             "--volume-fov", "80", "--out-root", str(tmp_path / "data")])
    with (tmp_path / "data/oasis/subjects.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [(r["group"], r["Age"], r["nWBV"]) for r in rows] == [("OAS1-0001", "74", "0.74"),
                                                                 ("OAS1-0002", "80", "0.74")]
    assert len(list((tmp_path / "data/oasis/volumes").rglob("*.npz"))) == 2


def test_cnn3d_cv_fold_end_to_end(tmp_path, monkeypatch):
    rows = []
    for i in range(12):
        label = "AD" if i % 2 else "CN"
        vol, seg, aff = _subject(i, 28 if label == "CN" else 22)
        v, s = tmp_path / "in" / f"v{i}.nii.gz", tmp_path / "in" / f"s{i}.nii.gz"
        v.parent.mkdir(exist_ok=True)
        nib.save(nib.Nifti1Image(vol, aff), str(v))
        nib.save(nib.Nifti1Image(seg, aff), str(s))
        rows.append({"subject": f"sub{i}", "scan": "a", "label": label, "path": v, "seg_path": s})
    manifest = tmp_path / "m.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pv.main(["manifest", "--csv", str(manifest), "--dataset", "toy3d", "--slices", "2", "--masks", "precomputed",
             "--structures", "ventricles", "--volume-size", "24", "--volume-fov", "96", "--workers", "2",
             "--out-root", str(tmp_path / "data")])
    with np.load(next((tmp_path / "data/toy3d/volumes/AD").glob("*.npz"))) as z:
        assert z["image"].shape == z["mask"].shape == (24, 24, 24) and z["mask"].max() == 1

    monkeypatch.setitem(DISEASES, "toy3d", DiseaseConfig("toy3d", "data/toy3d/mri", "data/toy3d/masks", ("CN", "AD"),
                                                         group_regex=r"^(.*?)__", group_across_classes=True))
    result = cnn3d.main(["--disease", "toy3d", "--data-root", str(tmp_path), "--out-dir", str(tmp_path / "runs"),
                         "--cv-folds", "3", "--fold", "1", "--epochs", "2", "--batch-size", "4", "--width", "4",
                         "--seg-weight", "0.5", "--num-workers", "0", "--bootstrap", "20", "--device", "cpu"])
    run = tmp_path / "runs/toy3d/cnn3d_seg_seed42_cv3f1"
    assert result["arch"] == "cnn3d" and result["fold"] == 1
    assert json.loads((run / "test_metrics.json").read_text())["split_sizes"]["test"] == 4
    with (run / "test_predictions.csv").open() as f:
        preds = list(csv.DictReader(f))
    assert len(preds) == 4 and {"p_CN", "p_AD", "group"} <= set(preds[0])
