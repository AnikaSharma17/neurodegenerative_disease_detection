"""Whole-volume 3D CNN for patient-level classification.

Reads the ``<subject>__<scan>.npz`` volumes that ``preprocessing/prepare_volumes.py
--volume-size N`` writes (uint8 ``image``, optional binary ``mask`` of the chosen
structures). One volume is one sample, so every metric here is patient-level.

The network is a small 3D ResNet (four stages, 24-192 channels, ~4.5 M parameters)
trained from scratch. With an anatomical mask it can add an auxiliary segmentation
loss (``--seg-weight``) on a light decoder, the 3D analogue of the 2D multi-task model.

    python -m neurodx.cnn3d --disease oasis --cv-folds 5 --fold 0

Outputs go to runs/<disease>/cnn3d_seed<seed>[_cv<k>f<i>]/:
    split.csv, history.csv, best.pt, test_metrics.json, test_predictions.csv (one row per
    patient), confusion_matrix.png, curves.png, calibration.png
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .calibration import expected_calibration_error, fit_temperature, risk_coverage
from .config import DISEASES
from .data import Sample, check_no_group_leakage, kfold_split, save_split, split_samples
from .metrics import classification_metrics, confusion_matrix, group_bootstrap_ci


def volume_root(image_root: str) -> str:
    """``data/oasis/mri`` -> ``data/oasis/volumes``."""
    return str(Path(image_root).parent / "volumes")


def collect_volumes(disease: str, root: str | Path = ".") -> list[Sample]:
    cfg = DISEASES[disease]
    vol_dir = Path(root) / volume_root(cfg.image_root)
    group_re = re.compile(cfg.group_regex or r"^(.*?)__")
    samples = []
    for label, cls in enumerate(cfg.classes):
        d = vol_dir / cls
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.npz")):
            m = group_re.match(f.name)
            group = m.group(1) if m else f.stem
            if m and not cfg.group_across_classes:
                group = f"{cls}/{group}"
            samples.append(Sample(str(f), None, label, group))
    if not samples:
        raise FileNotFoundError(f"No volumes under {vol_dir}. Run preprocessing/prepare_volumes.py "
                                "with --volume-size.")
    return samples


class VolumeDataset(Dataset):
    """Returns ``(volume [1,D,H,W] float, mask [D,H,W] long, label, has_mask, index)``.

    Training augmentation: left-right flip, random shift of up to ``shift`` voxels, and
    intensity scale/offset. Flip and shift are applied to the mask too."""

    def __init__(self, samples: list[Sample], train: bool = False, shift: int = 4):
        self.samples, self.train, self.shift = samples, train, shift
        self.labels = [s.label for s in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        with np.load(s.image_path) as z:
            vol = z["image"].astype(np.float32) / 255.0
            mask = z["mask"].astype(np.int64) if "mask" in z.files else None
        has_mask = mask is not None
        if mask is None:
            mask = np.zeros(vol.shape, np.int64)
        if self.train:
            if random.random() < 0.5:  # axis 0 is left-right after as_closest_canonical
                vol, mask = vol[::-1], mask[::-1]
            if self.shift:
                sh = [random.randint(-self.shift, self.shift) for _ in range(3)]
                vol, mask = np.roll(vol, sh, (0, 1, 2)), np.roll(mask, sh, (0, 1, 2))
            vol = vol * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05)
        vol = (vol - 0.5) / 0.25
        return (torch.from_numpy(np.ascontiguousarray(vol))[None], torch.from_numpy(np.ascontiguousarray(mask)),
                s.label, has_mask, idx)


class ResBlock3d(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(cout)
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.BatchNorm3d(cout)))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(self.bn2(self.conv2(out)) + self.skip(x), inplace=True)


class ResNet3d(nn.Module):
    """Stem (stride 2) + 4 residual stages (strides 1, 2, 2, 2). Returns logits and, if
    ``seg``, a 2-class segmentation map at the input resolution."""

    def __init__(self, num_classes: int, width: int = 24, dropout: float = 0.3, seg: bool = False):
        super().__init__()
        ch = [width, width * 2, width * 4, width * 8]
        self.stem = nn.Sequential(nn.Conv3d(1, ch[0], 5, 2, 2, bias=False), nn.BatchNorm3d(ch[0]), nn.ReLU(True))
        self.stages = nn.ModuleList([
            nn.Sequential(ResBlock3d(ch[0], ch[0], 1), ResBlock3d(ch[0], ch[0], 1)),
            nn.Sequential(ResBlock3d(ch[0], ch[1], 2), ResBlock3d(ch[1], ch[1], 1)),
            nn.Sequential(ResBlock3d(ch[1], ch[2], 2), ResBlock3d(ch[2], ch[2], 1)),
            nn.Sequential(ResBlock3d(ch[2], ch[3], 2), ResBlock3d(ch[3], ch[3], 1)),
        ])
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(ch[3], num_classes))
        self.seg_head = None
        if seg:  # light decoder from the stage-2 features (1/4 resolution)
            self.seg_head = nn.Sequential(nn.Conv3d(ch[1], ch[0], 3, 1, 1, bias=False), nn.BatchNorm3d(ch[0]),
                                          nn.ReLU(True), nn.Conv3d(ch[0], 2, 1))

    def forward(self, x):
        size = x.shape[2:]
        x = self.stem(x)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        logits = self.head(x)
        if self.seg_head is None:
            return logits, None
        seg = F.interpolate(self.seg_head(feats[1]), size=size, mode="trilinear", align_corners=False)
        return logits, seg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--disease", choices=sorted(DISEASES), required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--seg-weight", type=float, default=0.0, help="Auxiliary segmentation loss weight (needs masks).")
    p.add_argument("--class-weighting", action="store_true")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--cv-folds", type=int, default=0)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--data-root", default=".")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def run_epoch(model, loader, device, num_classes, seg_weight=0.0, cls_weights=None, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    logits_all, labels_all, idx_all = [], [], []
    use_amp = device.type == "cuda"
    with torch.set_grad_enabled(train):
        for vol, mask, labels, has_mask, idx in loader:
            vol, labels = vol.to(device, non_blocking=True), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits, seg = model(vol)
                loss = F.cross_entropy(logits.float(), labels, weight=cls_weights)
                if seg is not None and seg_weight > 0 and has_mask.any():
                    keep = has_mask.to(device)
                    loss = loss + seg_weight * F.cross_entropy(seg.float()[keep], mask.to(device)[keep])
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total += loss.item() * labels.size(0)
            n += labels.size(0)
            logits_all.append(logits.float().detach().cpu())
            labels_all.append(labels.cpu())
            idx_all.append(idx)
    logits, labels = torch.cat(logits_all), torch.cat(labels_all)
    return {"loss": total / max(n, 1), "cm": confusion_matrix(logits.argmax(1), labels, num_classes),
            "logits": logits, "labels": labels, "idx": torch.cat(idx_all)}


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = DISEASES[args.disease]
    nc = len(cfg.classes)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    name = args.run_name or f"cnn3d{'_seg' if args.seg_weight > 0 else ''}_seed{args.seed}" + (
        f"_cv{args.cv_folds}f{args.fold}" if args.cv_folds else "")
    run_dir = Path(args.out_dir) / args.disease / name
    run_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_volumes(args.disease, args.data_root)
    splits = (kfold_split(samples, args.cv_folds, args.fold, args.val_frac, args.seed) if args.cv_folds
              else split_samples(samples, args.val_frac, args.test_frac, args.seed))
    check_no_group_leakage(splits)
    save_split(splits, run_dir / "split.csv")
    ds = {k: VolumeDataset(v, train=(k == "train")) for k, v in splits.items()}
    for k, d in ds.items():
        print(f"{k:5s}: {len(d):4d} volumes, per class {np.bincount(d.labels, minlength=nc).tolist()}")
        if len(d) == 0:
            raise ValueError(f"Empty {k} split")
    kw = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    drop_last = len(ds["train"]) > args.batch_size  # avoid a size-1 BatchNorm batch
    train_loader = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True, drop_last=drop_last, **kw)
    val_loader = DataLoader(ds["val"], batch_size=args.batch_size, shuffle=False, **kw)
    test_loader = DataLoader(ds["test"], batch_size=args.batch_size, shuffle=False, **kw)

    model = ResNet3d(nc, args.width, seg=args.seg_weight > 0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    cls_weights = None
    if args.class_weighting:
        counts = np.bincount(ds["train"].labels, minlength=nc).astype(np.float64)
        cls_weights = torch.tensor(counts.sum() / (nc * np.maximum(counts, 1)), dtype=torch.float32, device=device)

    history, best_f1, best_epoch, stale = [], -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device, nc, args.seg_weight, cls_weights, optimizer, scaler)
        va = run_epoch(model, val_loader, device, nc, args.seg_weight, cls_weights)
        scheduler.step()
        trm, vam = classification_metrics(tr["cm"], cfg.classes), classification_metrics(va["cm"], cfg.classes)
        history.append({"epoch": epoch, "lr": scheduler.get_last_lr()[0], "train_loss": tr["loss"],
                        "train_acc": trm["accuracy"], "train_macro_f1": trm["macro_f1"], "val_loss": va["loss"],
                        "val_acc": vam["accuracy"], "val_bal_acc": vam["balanced_accuracy"],
                        "val_macro_f1": vam["macro_f1"], "seconds": time.time() - t0})
        improved = vam["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, stale = vam["macro_f1"], epoch, 0
            torch.save({"model_state": model.state_dict(), "args": vars(args), "classes": list(cfg.classes),
                        "epoch": epoch, "kind": "cnn3d"}, run_dir / "best.pt")
        else:
            stale += 1
        print(f"[{epoch:3d}/{args.epochs}] loss {tr['loss']:.4f}/{va['loss']:.4f} val acc {vam['accuracy']:.4f} "
              f"macro-F1 {vam['macro_f1']:.4f}{'  *best*' if improved else ''} ({time.time() - t0:.0f}s)", flush=True)
        if args.patience and stale >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    with (run_dir / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    va = run_epoch(model, val_loader, device, nc)
    te = run_epoch(model, test_loader, device, nc)
    temperature = fit_temperature(va["logits"], va["labels"])
    ckpt["temperature"] = temperature
    torch.save(ckpt, run_dir / "best.pt")

    labels = te["labels"].numpy()
    probs_raw = torch.softmax(te["logits"], 1).numpy()
    probs = torch.softmax(te["logits"] / temperature, 1).numpy()
    preds = probs.argmax(1)
    test_samples = ds["test"].samples
    groups = [test_samples[i].group for i in te["idx"].tolist()]
    cm = confusion_matrix(torch.from_numpy(preds), torch.from_numpy(labels), nc)
    rc = risk_coverage(probs, labels)
    patient = classification_metrics(cm, cfg.classes)
    result = {
        "disease": args.disease, "arch": "cnn3d", "backbone": f"resnet3d_w{args.width}", "seed": args.seed,
        "seg_weight": args.seg_weight, "split_level": "group", "cv_folds": args.cv_folds,
        "fold": args.fold if args.cv_folds else None, "best_epoch": best_epoch, "epochs_run": len(history),
        "best_val_macro_f1": best_f1, "test": patient, "test_patient_level": patient,
        "test_confusion_matrix": cm.tolist(),
        "test_ci95": group_bootstrap_ci(labels, preds, groups, nc, args.bootstrap, args.seed) if args.bootstrap
        else None,
        "calibration": {"temperature": temperature, "ece_before": expected_calibration_error(probs_raw, labels),
                        "ece_after": expected_calibration_error(probs, labels), "aurc": rc["aurc"],
                        **{k: v for k, v in rc.items() if k.startswith("risk_at")}},
        "split_sizes": {k: len(v) for k, v in splits.items()},
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2))
    with (run_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "group", "label", "pred", *[f"p_{c}" for c in cfg.classes]])
        for i, g, y, p in zip(te["idx"].tolist(), groups, labels, probs):
            w.writerow([test_samples[i].image_path, g, int(y), int(p.argmax()), *[f"{x:.5f}" for x in p]])

    from .plots import plot_calibration, plot_confusion_matrix, plot_history
    plot_confusion_matrix(cm, cfg.classes, run_dir / "confusion_matrix.png",
                          title=f"{args.disease} 3D CNN: test patients")
    plot_history(history, run_dir / "curves.png")
    plot_calibration(probs_raw, probs, labels, rc, run_dir / "calibration.png")
    print(f"TEST (epoch {best_epoch}): patient acc {patient['accuracy']:.4f} bal-acc "
          f"{patient['balanced_accuracy']:.4f} macro-F1 {patient['macro_f1']:.4f} -> {run_dir}")
    return result


if __name__ == "__main__":
    main()
