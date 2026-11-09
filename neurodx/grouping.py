"""Recover patient groups from image similarity.

Slices of the same patient (adjacent 2D cuts from one volume) are far more alike
than slices of different patients. We link two images when the Pearson correlation
of their downsampled pixels exceeds ``threshold`` and take connected components as
pseudo-patients. Linking only happens within a class folder.

For the Alzheimer dataset, ~20% of files are named ``<slice> (<patient>).jpg``.
Those known IDs are used to validate the grouping (see ``validate``).

    python -m neurodx.grouping --disease alzheimer      # writes data/alzheimer/groups.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .config import DISEASES, IMAGE_EXTS

KNOWN_ID = re.compile(r"^(\d+) \((\d+)\)\.\w+$")  # "<slice> (<patient>).jpg"


def similarity_matrix(paths: list[Path], size: tuple[int, int] = (88, 104), device: str | None = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = np.stack([np.asarray(Image.open(p).convert("L").resize(size), np.float32).ravel() for p in paths])
    t = torch.tensor(x, device=device)
    t = t - t.mean(1, keepdim=True)
    t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-8)
    sim = (t @ t.T).cpu().numpy()
    np.fill_diagonal(sim, 1.0)
    return sim


def components(sim: np.ndarray, threshold: float) -> np.ndarray:
    _, labels = connected_components(csr_matrix(sim > threshold), directed=False)
    return labels


def validate(names: list[str], labels: np.ndarray) -> dict | None:
    """Compare components with the patient IDs embedded in some file names."""
    ids = np.array([int(m.group(2)) if (m := KNOWN_ID.match(n)) else -1 for n in names])
    known = ids >= 0
    if known.sum() < 2:
        return None
    lab, pid = labels[known], ids[known]
    off = ~np.eye(len(lab), dtype=bool)
    same_c = (lab[:, None] == lab[None, :]) & off
    same_p = (pid[:, None] == pid[None, :]) & off
    both = (same_c & same_p).sum()
    return {
        "known_files": int(known.sum()),
        "same_patient_pairs_kept_together": float(both / max(same_p.sum(), 1)),
        "co_clustered_pairs_same_patient": float(both / max(same_c.sum(), 1)),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--disease", choices=sorted(DISEASES), required=True)
    p.add_argument("--data-root", default=".")
    p.add_argument("--threshold", type=float, default=0.97)
    p.add_argument("--out", default=None, help="Default: the disease's groups_file.")
    args = p.parse_args(argv)

    cfg = DISEASES[args.disease]
    out = Path(args.data_root) / (args.out or cfg.groups_file or f"data/{cfg.name}/groups.csv")
    rows = []
    for cls in cfg.classes:
        d = Path(args.data_root) / cfg.image_root / cls
        names = sorted(f.name for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        labels = components(similarity_matrix([d / n for n in names]), args.threshold)
        sizes = np.bincount(labels)
        print(f"{cls:18s} images {len(names):5d}  groups {len(sizes):4d}  "
              f"size min/median/max {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}")
        if (v := validate(names, labels)) is not None:
            print(f"{'':18s} validation on {v['known_files']} files with known patient IDs: "
                  f"recall {v['same_patient_pairs_kept_together']:.3f}, "
                  f"precision {v['co_clustered_pairs_same_patient']:.3f}")
        rows += [(f"{cls}/{n}", f"{cls}/p{lab:04d}") for n, lab in zip(names, labels)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "group"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
