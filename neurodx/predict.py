"""Predict one or more MRI slices with a trained checkpoint.

    python -m neurodx.predict --checkpoint runs/alzheimer/<run>/best.pt path/to/slice.jpg [...]
    python -m neurodx.predict --checkpoint ... --cam-dir cams/ --abstain-below 0.6 slice.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .inference import load_checkpoint, overlay, predict


def main(argv: list[str] | None = None) -> list[dict]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--abstain-below", type=float, default=0.0,
                   help="Flag predictions whose calibrated confidence is below this value.")
    p.add_argument("--cam-dir", default=None, help="Write Grad-CAM overlays here.")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    lm = load_checkpoint(args.checkpoint, args.device)
    results = []
    for path in args.images:
        image = Image.open(path)
        r = predict(lm, image, args.abstain_below, explain=args.cam_dir is not None)
        if "cam" in r:
            out = Path(args.cam_dir)
            out.mkdir(parents=True, exist_ok=True)
            overlay(image, r.pop("cam")).save(out / f"cam_{Path(path).stem}.png")
        results.append({"image": path, **r})
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
