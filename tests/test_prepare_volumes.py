import csv
import sys
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
import prepare_volumes as pv  # noqa: E402

from neurodx.config import DiseaseConfig  # noqa: E402
from neurodx.data import check_no_group_leakage, collect_samples, kfold_split  # noqa: E402


def _volume(seed: int, shape=(40, 48, 30)) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    vol = rng.normal(10, 2, shape).astype(np.float32)
    seg = np.zeros(shape, np.int16)
    seg[8:32, 8:40, 5:25] = 2          # white matter
    seg[15:20, 20:25, 10:20] = 17      # left hippocampus
    seg[22:26, 20:28, 10:20] = 4       # left lateral ventricle
    vol[seg > 0] += 100
    return vol, seg


def _save(arr, path: Path, analyze: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = (nib.AnalyzeImage if analyze else nib.Nifti1Image)(arr, np.eye(4))
    nib.save(img, str(path))
    return path


def test_manifest_with_precomputed_segmentation(tmp_path):
    rows = []
    for i, (subject, label) in enumerate([("sub1", "AD"), ("sub1", "AD"), ("sub2", "CN"), ("sub3", "CN")]):
        vol, seg = _volume(i)
        v = _save(vol, tmp_path / "in" / f"v{i}.nii.gz")
        s = _save(seg, tmp_path / "in" / f"seg{i}.nii.gz")
        rows.append({"subject": subject, "scan": f"scan{i}", "label": label, "path": v, "seg_path": s})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pv.main(["manifest", "--csv", str(manifest), "--dataset", "toy", "--slices", "6", "--masks", "precomputed",
             "--structures", "hippocampus+ventricles", "--out-root", str(tmp_path / "data")])

    images = sorted((tmp_path / "data/toy/mri").rglob("*.png"))
    masks = sorted((tmp_path / "data/toy/masks").rglob("*.png"))
    assert len(images) == len(masks) == 4 * 6
    assert images[0].name.startswith("sub1__scan0_s00")

    cfg = DiseaseConfig("toy", "data/toy/mri", "data/toy/masks", ("CN", "AD"),
                        group_regex=r"^(.*?)__", group_across_classes=True)
    samples = collect_samples(cfg, tmp_path)
    assert {s.group for s in samples} == {"sub1", "sub2", "sub3"}  # two scans of sub1 share one group
    for fold in range(3):
        check_no_group_leakage(kfold_split(samples, 3, fold, 0.2, 0))
    some_mask = np.array(__import__("PIL.Image", fromlist=["Image"]).open(masks[3]))
    assert set(np.unique(some_mask)) <= {0, 255}


def test_oasis1_layout_and_cdr_labels(tmp_path):
    src = tmp_path / "oasis"
    for i, sid in enumerate(["OAS1_0001_MR1", "OAS1_0002_MR1", "OAS1_0002_MR2", "OAS1_0003_MR1"]):
        vol, _ = _volume(i)
        _save(vol, src / sid / "PROCESSED/MPRAGE/T88_111" / f"{sid}_mpr_n4_anon_111_t88_masked_gfc.img", analyze=True)
    table = tmp_path / "oasis_cross-sectional.csv"
    table.write_text("ID,M/F,Age,CDR\nOAS1_0001_MR1,F,74,0\nOAS1_0002_MR1,M,80,0.5\n"
                     "OAS1_0002_MR2,M,80,0.5\nOAS1_0003_MR1,F,25,\n")
    scans = pv.oasis1_scans(src, table)
    assert [(s.subject, s.label) for s in scans] == [("OAS1_0001", "NonDemented"), ("OAS1_0002", "VeryMildDemented")]
    assert len(pv.oasis1_scans(src, table, include_young=True)) == 3
    pv.main(["oasis1", "--src", str(src), "--csv", str(table), "--dataset", "oasis", "--slices", "4",
             "--out-root", str(tmp_path / "data")])
    assert len(list((tmp_path / "data/oasis/mri/VeryMildDemented").glob("OAS1-0002__*.png"))) == 4


def test_ida_keeps_first_visit_per_subject(tmp_path):
    src = tmp_path / "ida"
    for image_id in ("I100", "I200"):  # I300 is listed in the CSV but was not downloaded
        vol, _ = _volume(int(image_id[1:]))
        _save(vol, src / "002_S_0001" / "MPRAGE" / "2010-01-01" / image_id / f"ADNI_002_S_0001_MR_{image_id}.nii")
    table = tmp_path / "collection.csv"
    table.write_text("Image Data ID,Subject,Group,Acq Date\nI200,002_S_0001,AD,5/01/2012\n"
                     "I100,002_S_0001,MCI,3/15/2010\nI300,002_S_0009,CN,1/01/2011\n")
    scans = pv.ida_scans(src, table, {"AD": "AD", "MCI": "MCI", "CN": "CN"})
    assert [(s.subject, s.scan, s.label) for s in scans] == [("002_S_0001", "I100", "MCI")]  # I300 has no file
    assert len(pv.ida_scans(src, table, {"AD": "AD", "MCI": "MCI"}, all_visits=True)) == 2


def test_mask_structures():
    seg = np.array([[0, 2, 17], [4, 24, 53]])
    assert pv.mask_from_labels(seg, "brain").tolist() == [[False, True, True], [True, False, True]]
    assert pv.mask_from_labels(seg, "hippocampus").tolist() == [[False, False, True], [False, False, True]]
