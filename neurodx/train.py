"""Train and evaluate a multi-task model.

Example (from the repository root):
    python -m neurodx.train --disease alzheimer --backbone efficientnet_b0 --arch multitask

Outputs go to runs/<disease>/<run-name>/ where run-name defaults to
<arch>_<backbone>_seg<w>_<split-level>_seed<seed>:
    split.csv, history.csv, best.pt, test_metrics.json, test_predictions.csv,
    confusion_matrix.png, curves.png, calibration.png
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .calibration import expected_calibration_error, fit_temperature, risk_coverage
from .config import DISEASES, NUM_SEG_CLASSES
from .data import MRIDataset, check_no_group_leakage, collect_samples, save_split, split_samples
from .metrics import (
    aggregate_by_group,
    classification_metrics,
    confusion_matrix,
    group_bootstrap_ci,
    segmentation_metrics,
)
from .models import ARCHS, BACKBONES, build_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--disease", choices=sorted(DISEASES), required=True)
    p.add_argument("--backbone", choices=BACKBONES, default="efficientnet_b0")
    p.add_argument("--arch", choices=ARCHS, default="multitask")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seg-weight", type=float, default=1.0,
                   help="Weight of the segmentation loss; 0 trains classification only.")
    p.add_argument("--class-weighting", action="store_true",
                   help="Inverse-frequency class weights in the classification loss.")
    p.add_argument("--patience", type=int, default=0, help="Early-stopping patience (0 = off).")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--split-level", choices=("group", "image"), default="group",
                   help="'group' keeps each patient/series in one split. 'image' is the leaky "
                        "slice-level split, kept only to measure leakage.")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true", help="Random init (offline/smoke tests).")
    p.add_argument("--data-root", default=".", help="Directory that contains data/.")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--limit-per-class", type=int, default=None, help="Debug: cap images per class.")
    p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples for CIs (0 = off).")
    p.add_argument("--mlflow", action="store_true", help="Also log params/metrics to MLflow (if installed).")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, num_classes, seg_weight, cls_weights=None,
              optimizer=None, scaler=None, collect=False):
    """One pass over ``loader``; trains if ``optimizer`` is given. Returns a dict with
    loss, classification/segmentation confusion matrices and (if ``collect``) logits."""
    train = optimizer is not None
    model.train(train)
    total_loss, n = 0.0, 0
    cls_cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    seg_cm = torch.zeros(NUM_SEG_CLASSES, NUM_SEG_CLASSES, dtype=torch.long)
    all_logits, all_labels = [], []
    use_amp = scaler is not None or (not train and device.type == "cuda")
    with torch.set_grad_enabled(train):
        for images, masks, labels, has_mask in loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            has_mask = has_mask.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                seg_out, cls_out = model(images)
                loss = F.cross_entropy(cls_out, labels, weight=cls_weights)
                if seg_weight > 0 and has_mask.any():
                    loss = loss + seg_weight * F.cross_entropy(seg_out[has_mask], masks[has_mask])
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * images.size(0)
            n += images.size(0)
            cls_cm += confusion_matrix(cls_out.argmax(1), labels, num_classes)
            if has_mask.any():
                seg_cm += confusion_matrix(seg_out[has_mask].argmax(1), masks[has_mask], NUM_SEG_CLASSES)
            if collect:
                all_logits.append(cls_out.float().detach().cpu())
                all_labels.append(labels.cpu())
    out = {"loss": total_loss / max(n, 1), "cls_cm": cls_cm, "seg_cm": seg_cm}
    if collect:
        out["logits"] = torch.cat(all_logits) if all_logits else torch.empty(0, num_classes)
        out["labels"] = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long)
    return out


def default_run_name(args: argparse.Namespace) -> str:
    return f"{args.arch}_{args.backbone}_seg{args.seg_weight:g}_{args.split_level}_seed{args.seed}"


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    seed_everything(args.seed)
    cfg = DISEASES[args.disease]
    nc = len(cfg.classes)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = Path(args.out_dir) / args.disease / (args.run_name or default_run_name(args))
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    samples = collect_samples(cfg, args.data_root, require_masks=args.seg_weight > 0,
                              limit_per_class=args.limit_per_class, split_level=args.split_level)
    splits = split_samples(samples, args.val_frac, args.test_frac, args.seed)
    check_no_group_leakage(splits)
    save_split(splits, run_dir / "split.csv")
    for name in ("train", "val", "test"):
        counts = np.bincount([s.label for s in splits[name]], minlength=nc)
        n_groups = len({s.group for s in splits[name]})
        print(f"{name:5s}: {len(splits[name]):5d} images, {n_groups:4d} groups; "
              f"per class {dict(zip(cfg.classes, counts.tolist()))}")
        if len(splits[name]) == 0:
            raise ValueError(f"Empty {name} split; adjust --val-frac/--test-frac or add data.")

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(MRIDataset(splits["train"], args.image_size, train=True), shuffle=True, **loader_kw)
    val_loader = DataLoader(MRIDataset(splits["val"], args.image_size), shuffle=False, **loader_kw)
    test_loader = DataLoader(MRIDataset(splits["test"], args.image_size), shuffle=False, **loader_kw)

    # ---- model ----
    model = build_model(args.arch, args.backbone, nc, NUM_SEG_CLASSES, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    cls_weights = None
    if args.class_weighting:
        counts = np.bincount([s.label for s in splits["train"]], minlength=nc).astype(np.float64)
        w = counts.sum() / (nc * np.maximum(counts, 1))
        cls_weights = torch.tensor(w, dtype=torch.float32, device=device)

    # ---- train; select checkpoint on validation macro-F1 ----
    history, best_f1, best_epoch, stale = [], -1.0, 0, 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device, nc, args.seg_weight, cls_weights, optimizer, scaler)
        va = run_epoch(model, val_loader, device, nc, args.seg_weight, cls_weights)
        scheduler.step()
        trm, vam = classification_metrics(tr["cls_cm"], cfg.classes), classification_metrics(va["cls_cm"], cfg.classes)
        trs, vas = segmentation_metrics(tr["seg_cm"]), segmentation_metrics(va["seg_cm"])
        row = {
            "epoch": epoch, "lr": scheduler.get_last_lr()[0],
            "train_loss": tr["loss"], "train_acc": trm["accuracy"], "train_macro_f1": trm["macro_f1"],
            "train_fg_dice": trs["fg_dice"],
            "val_loss": va["loss"], "val_acc": vam["accuracy"], "val_bal_acc": vam["balanced_accuracy"],
            "val_macro_f1": vam["macro_f1"], "val_fg_iou": vas["fg_iou"], "val_fg_dice": vas["fg_dice"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        improved = vam["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, stale = vam["macro_f1"], epoch, 0
            torch.save({"model_state": model.state_dict(), "args": vars(args),
                        "classes": list(cfg.classes), "epoch": epoch}, run_dir / "best.pt")
        else:
            stale += 1
        print(f"[{epoch:3d}/{args.epochs}] loss {tr['loss']:.4f}/{va['loss']:.4f}  "
              f"val acc {vam['accuracy']:.4f} bal-acc {vam['balanced_accuracy']:.4f} "
              f"macro-F1 {vam['macro_f1']:.4f} fg-dice {vas['fg_dice']:.4f}"
              f"{'  *best*' if improved else ''}  ({row['seconds']:.0f}s)", flush=True)
        if args.patience and stale >= args.patience:
            print(f"Early stopping: no val macro-F1 improvement for {args.patience} epochs.")
            break
    train_seconds = time.time() - t_start

    with (run_dir / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    # ---- final, once-only evaluation of the selected checkpoint ----
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    va = run_epoch(model, val_loader, device, nc, args.seg_weight, collect=True)
    te = run_epoch(model, test_loader, device, nc, args.seg_weight, collect=True)

    # Calibration: temperature fitted on validation only, applied to test.
    temperature = fit_temperature(va["logits"], va["labels"])
    ckpt["temperature"] = temperature
    torch.save(ckpt, run_dir / "best.pt")
    labels = te["labels"].numpy()
    probs_raw = torch.softmax(te["logits"], 1).numpy()
    probs = torch.softmax(te["logits"] / temperature, 1).numpy()
    preds = probs.argmax(1)
    groups = [s.group for s in splits["test"]]

    # Patient-level: average calibrated slice probabilities per group.
    gp, gl, _ = aggregate_by_group(probs, labels, groups)
    g_cm = confusion_matrix(torch.from_numpy(gp.argmax(1)), torch.from_numpy(gl), nc)
    rc = risk_coverage(probs, labels)

    result = {
        "disease": args.disease, "arch": args.arch, "backbone": args.backbone, "seed": args.seed,
        "seg_weight": args.seg_weight, "split_level": args.split_level, "pretrained": not args.no_pretrained,
        "class_weighting": args.class_weighting, "image_size": args.image_size,
        "best_epoch": best_epoch, "epochs_run": len(history), "best_val_macro_f1": best_f1,
        "train_seconds": train_seconds, "test_loss": te["loss"],
        "test": classification_metrics(te["cls_cm"], cfg.classes),
        "test_ci95": (group_bootstrap_ci(labels, preds, groups, nc, args.bootstrap, args.seed)
                      if args.bootstrap else None),
        "test_patient_level": classification_metrics(g_cm, cfg.classes),
        "test_patient_confusion_matrix": g_cm.tolist(),
        "test_segmentation": segmentation_metrics(te["seg_cm"]),
        "test_confusion_matrix": te["cls_cm"].tolist(),
        "calibration": {
            "temperature": temperature,
            "ece_before": expected_calibration_error(probs_raw, labels),
            "ece_after": expected_calibration_error(probs, labels),
            "aurc": rc["aurc"],
            **{k: v for k, v in rc.items() if k.startswith("risk_at")},
        },
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_groups": {k: len({s.group for s in v}) for k, v in splits.items()},
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2))
    with (run_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "group", "label", "pred", *[f"p_{c}" for c in cfg.classes]])
        for s, y, p in zip(splits["test"], labels, probs):
            w.writerow([s.image_path, s.group, int(y), int(p.argmax()), *[f"{x:.5f}" for x in p]])

    from .plots import plot_calibration, plot_confusion_matrix, plot_history
    plot_confusion_matrix(te["cls_cm"], cfg.classes, run_dir / "confusion_matrix.png",
                          title=f"{args.disease} {args.arch}/{args.backbone}: test slices")
    plot_history(history, run_dir / "curves.png")
    plot_calibration(probs_raw, probs, labels, rc, run_dir / "calibration.png")

    if args.mlflow:
        _log_mlflow(args, result, run_dir)

    t, pl = result["test"], result["test_patient_level"]
    print(f"TEST (epoch {best_epoch}): slice acc {t['accuracy']:.4f} bal-acc {t['balanced_accuracy']:.4f} "
          f"macro-F1 {t['macro_f1']:.4f} | patient acc {pl['accuracy']:.4f} | "
          f"ECE {result['calibration']['ece_before']:.3f}->{result['calibration']['ece_after']:.3f} -> {run_dir}")
    return result


def _log_mlflow(args, result, run_dir: Path) -> None:
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed; skipping MLflow logging (pip install mlflow).")
        return
    mlflow.set_experiment(f"neurodx-{args.disease}")
    with mlflow.start_run(run_name=run_dir.name):
        mlflow.log_params({k: v for k, v in vars(args).items() if v is not None})
        mlflow.log_metrics({
            "test_accuracy": result["test"]["accuracy"],
            "test_balanced_accuracy": result["test"]["balanced_accuracy"],
            "test_macro_f1": result["test"]["macro_f1"],
            "test_patient_accuracy": result["test_patient_level"]["accuracy"],
            "ece_after": result["calibration"]["ece_after"],
        })
        mlflow.log_artifacts(str(run_dir), artifact_path="run")


if __name__ == "__main__":
    main()
