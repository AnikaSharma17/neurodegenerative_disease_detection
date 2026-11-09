"""Grad-CAM gallery and a simple sanity metric for a finished run.

Picks test slices that were correctly and incorrectly classified, writes a gallery
PNG, and reports how much of the Grad-CAM mass falls inside the head (pixels
brighter than the background). Attention mostly *outside* the head would suggest
the model relies on borders, text or scanner artefacts.

    python -m neurodx.explain --run runs/alzheimer/multitask_efficientnet_b0_seg1_group_seed0
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from .inference import grad_cam, load_checkpoint, overlay, preprocess


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--n", type=int, default=6, help="Examples per row (correct / wrong).")
    p.add_argument("--head-threshold", type=int, default=15, help="Gray level separating head from background.")
    p.add_argument("--max-images", type=int, default=300, help="Test slices used for the attention metric.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    run = Path(args.run)
    lm = load_checkpoint(run / "best.pt", args.device)
    with (run / "test_predictions.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    inside = []
    for r in rows[: args.max_images]:
        img = Image.open(r["image_path"])
        cam, _ = grad_cam(lm.model, preprocess(img, lm.image_size).to(lm.device))
        head = np.asarray(img.convert("L").resize(cam.shape[::-1]), np.float32) > args.head_threshold
        inside.append(float((cam * head).sum() / (cam.sum() + 1e-8)))

    correct = [r for r in rows if r["label"] == r["pred"]][: args.n]
    wrong = [r for r in rows if r["label"] != r["pred"]][: args.n]
    tiles = []
    for group in (correct, wrong):
        row_tiles = []
        for r in group:
            img = Image.open(r["image_path"])
            cam, _ = grad_cam(lm.model, preprocess(img, lm.image_size).to(lm.device))
            row_tiles.append(overlay(img, cam).resize((224, 224)))
        tiles.append(row_tiles)

    from .plots import plot_cam_gallery
    plot_cam_gallery(tiles, [correct, wrong], lm.classes, run / "gradcam_gallery.png")
    summary = {
        "n_slices": len(inside),
        "cam_mass_inside_head_mean": float(np.mean(inside)) if inside else None,
        "cam_mass_inside_head_p10": float(np.percentile(inside, 10)) if inside else None,
    }
    (run / "gradcam_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), f"\n-> {run / 'gradcam_gallery.png'}")
    return summary


if __name__ == "__main__":
    main()
