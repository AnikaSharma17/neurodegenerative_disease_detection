"""Generate binary pseudo-masks from MRI slices with classic image processing.

The mask marks bright regions (CLAHE-enhanced intensity > threshold) after
median denoising and morphological opening. It is a heuristic target derived
from the image itself, NOT a clinical lesion annotation.

Writes data/<disease>/masks/<class>/mask_<stem>.png with values {0, 255}.

    python preprocessing/generate_masks.py --disease parkinson
    python preprocessing/generate_masks.py --disease alzheimer --overwrite
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neurodx.config import DISEASES, IMAGE_EXTS  # noqa: E402


def make_mask(gray: np.ndarray, threshold: int = 180) -> np.ndarray:
    denoised = cv2.medianBlur(gray, 5)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    _, mask = cv2.threshold(enhanced, threshold, 255, cv2.THRESH_BINARY)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--disease", choices=sorted(DISEASES), required=True)
    p.add_argument("--data-root", default=".")
    p.add_argument("--threshold", type=int, default=180)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    cfg = DISEASES[args.disease]
    root = Path(args.data_root)
    written = skipped = failed = 0
    for cls in cfg.classes:
        src, dst = root / cfg.image_root / cls, root / cfg.mask_root / cls
        dst.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(src.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            out = dst / f"mask_{img_path.stem}.png"
            if out.exists() and not args.overwrite:
                skipped += 1
                continue
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"Could not read {img_path}", file=sys.stderr)
                failed += 1
                continue
            cv2.imwrite(str(out), make_mask(gray, args.threshold))
            written += 1
    print(f"{args.disease}: wrote {written}, skipped {skipped} existing, failed {failed}")


if __name__ == "__main__":
    main()
