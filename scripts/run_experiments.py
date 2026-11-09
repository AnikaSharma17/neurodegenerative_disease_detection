"""Run the experiment matrix behind the README results table.

Each run is skipped if its test_metrics.json already exists, so the script can
be interrupted and resumed. Logs go to runs/logs/<disease>__<run-name>.log.

    python scripts/run_experiments.py                 # everything
    python scripts/run_experiments.py --only leakage  # one experiment group
    python scripts/run_experiments.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SEEDS = (0, 1, 2)
COMMON = ["--epochs", "25", "--patience", "8", "--batch-size", "32", "--num-workers", "8", "--class-weighting"]

# name -> list of (disease, extra args)
EXPERIMENTS: dict[str, list[tuple[str, list[str]]]] = {
    # Same model; only the split changes. Measures slice-level leakage.
    "leakage": [("alzheimer", ["--backbone", "efficientnet_b0", "--split-level", "image"])],
    # Backbone comparison on the patient-level split.
    "backbones": [
        ("alzheimer", ["--backbone", b]) for b in ("efficientnet_b0", "mobilenet_v2", "vgg16")
    ],
    # Does the pseudo-mask segmentation head help? (seg weight 0 = classification only)
    "ablation": [
        ("alzheimer", ["--backbone", "efficientnet_b0", "--seg-weight", "0"]),
        ("alzheimer", ["--backbone", "efficientnet_b0", "--arch", "unet"]),
    ],
    "parkinson": [
        ("parkinson", ["--backbone", "efficientnet_b0"]),
    ],
}


def run_name(extra: list[str], seed: int) -> str:
    opts = dict(zip(extra[::2], extra[1::2]))
    return (f"{opts.get('--arch', 'multitask')}_{opts.get('--backbone', 'efficientnet_b0')}"
            f"_seg{float(opts.get('--seg-weight', 1)):g}_{opts.get('--split-level', 'group')}_seed{seed}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=sorted(EXPERIMENTS), nargs="*")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    log_dir = Path(args.out_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for exp in args.only or EXPERIMENTS:
        for disease, extra in EXPERIMENTS[exp]:
            for seed in SEEDS:
                name = run_name(extra, seed)
                if (Path(args.out_dir) / disease / name / "test_metrics.json").exists():
                    continue
                if (disease, name) in {(d, n) for d, n, _ in jobs}:
                    continue
                cmd = [sys.executable, "-m", "neurodx.train", "--disease", disease, *COMMON, *extra,
                       "--seed", str(seed), "--out-dir", args.out_dir]
                jobs.append((disease, name, cmd))

    print(f"{len(jobs)} runs to do")
    failures = 0
    for i, (disease, name, cmd) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {disease}/{name}", flush=True)
        if args.dry_run:
            print("   ", " ".join(cmd))
            continue
        with (log_dir / f"{disease}__{name}.log").open("w", encoding="utf-8") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        if rc != 0:
            failures += 1
            print(f"    FAILED (exit {rc}); see {log_dir / f'{disease}__{name}.log'}", flush=True)
    print(f"done; {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
