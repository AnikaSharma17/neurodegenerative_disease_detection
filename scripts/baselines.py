"""Non-deep-learning baselines on exactly the same splits as neurodx.train.

  majority    : always predict the most frequent training class
  histogram   : logistic regression on a 64-bin intensity histogram (no spatial info)
  thumbnail   : logistic regression on 32x32 downsampled pixels

If a histogram-only model scores very high, the classes differ in global intensity
statistics (e.g. scanner or sequence), not necessarily in disease-related anatomy.

    python scripts/baselines.py --disease alzheimer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neurodx.config import DISEASES  # noqa: E402
from neurodx.data import collect_samples, split_samples  # noqa: E402
from neurodx.metrics import (  # noqa: E402
    aggregate_by_group,
    classification_metrics,
    confusion_matrix,
    group_bootstrap_ci,
)


def features(paths: list[str]) -> dict[str, np.ndarray]:
    hist, thumb = [], []
    for p in paths:
        g = np.asarray(Image.open(p).convert("L"), np.float32)
        h, _ = np.histogram(g, bins=64, range=(0, 255))
        hist.append(h / h.sum())
        thumb.append(np.asarray(Image.fromarray(g.astype(np.uint8)).resize((32, 32)), np.float32).ravel() / 255.0)
    return {"histogram": np.stack(hist), "thumbnail": np.stack(thumb)}


def evaluate(pred: np.ndarray, probs: np.ndarray, y: np.ndarray, groups: list[str], classes) -> dict:
    nc = len(classes)
    cm = confusion_matrix(torch.from_numpy(pred), torch.from_numpy(y), nc)
    gp, gl, _ = aggregate_by_group(probs, y, groups)
    gcm = confusion_matrix(torch.from_numpy(gp.argmax(1)), torch.from_numpy(gl), nc)
    return {
        "test": classification_metrics(cm, classes),
        "test_ci95": group_bootstrap_ci(y, pred, groups, nc, 1000, 0),
        "test_patient_level": classification_metrics(gcm, classes),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--disease", choices=sorted(DISEASES), required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--split-level", choices=("group", "image"), default="group")
    p.add_argument("--out-dir", default="runs")
    args = p.parse_args()

    cfg = DISEASES[args.disease]
    samples = collect_samples(cfg, require_masks=False, split_level=args.split_level)
    print(f"extracting features from {len(samples)} images ...")
    feats = features([s.image_path for s in samples])
    index = {s.image_path: i for i, s in enumerate(samples)}

    results = []
    for seed in args.seeds:
        splits = split_samples(samples, 0.15, 0.15, seed)
        tr = np.array([index[s.image_path] for s in splits["train"]])
        te = np.array([index[s.image_path] for s in splits["test"]])
        y_tr = np.array([s.label for s in splits["train"]])
        y_te = np.array([s.label for s in splits["test"]])
        groups = [s.group for s in splits["test"]]
        nc = len(cfg.classes)

        majority = np.bincount(y_tr, minlength=nc).argmax()
        probs = np.zeros((len(te), nc))
        probs[:, majority] = 1
        results.append({"model": "majority", "seed": seed,
                        **evaluate(probs.argmax(1), probs, y_te, groups, cfg.classes)})

        for name in ("histogram", "thumbnail"):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0))
            clf.fit(feats[name][tr], y_tr)
            proba = np.zeros((len(te), nc))
            proba[:, clf.classes_] = clf.predict_proba(feats[name][te])
            results.append({"model": f"logreg_{name}", "seed": seed,
                            **evaluate(proba.argmax(1), proba, y_te, groups, cfg.classes)})
        for r in results[-3:]:
            print(f"seed {seed} {r['model']:18s} slice bal-acc {r['test']['balanced_accuracy']:.3f} "
                  f"macro-F1 {r['test']['macro_f1']:.3f}  patient acc {r['test_patient_level']['accuracy']:.3f}")

    out = Path(args.out_dir) / args.disease / f"baselines_{args.split_level}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
