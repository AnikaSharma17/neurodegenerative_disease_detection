"""Patient-level model: attention multiple-instance learning over all slices of a patient.

Each patient (or Parkinson series) is a bag of 2D slices. A shared CNN encodes every
slice, a gated attention module (Ilse et al., 2018) weights the slices, and the weighted
average is classified once per patient. The attention weights show which slices drove
the decision.

    python -m neurodx.mil --disease alzheimer --cv-folds 5 --fold 0

Outputs go to runs/<disease>/mil_<backbone>_seed<seed>[_cv<k>f<i>]/:
    split.csv, history.csv, best.pt, test_metrics.json, test_predictions.csv (one row per
    patient, with the top-attended slice), confusion_matrix.png, curves.png, calibration.png
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .calibration import expected_calibration_error, fit_temperature, risk_coverage
from .config import DISEASES
from .data import MRIDataset, Sample, check_no_group_leakage, collect_samples, kfold_split, save_split, split_samples
from .metrics import classification_metrics, confusion_matrix, group_bootstrap_ci
from .models import BACKBONES, Encoder


class PatientBags(Dataset):
    """One item per group: ``(slices [N,3,H,W], label, index)``. During training a random
    subset of ``bag_size`` slices is drawn (with replacement if the group is smaller), so
    bags stack into a batch; evaluation uses every slice in path order."""

    def __init__(self, samples: list[Sample], image_size: int = 224, train: bool = False, bag_size: int = 16):
        by_group: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            by_group[s.group].append(i)
        self.groups = sorted(by_group)
        self.members = [sorted(by_group[g], key=lambda i: samples[i].image_path) for g in self.groups]
        self.labels = [samples[m[0]].label for m in self.members]
        self.samples = samples
        self.slices = MRIDataset(samples, image_size, train=train)
        self.train, self.bag_size = train, bag_size

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int):
        members = self.members[idx]
        if self.train and self.bag_size:
            pick = (random.sample(members, self.bag_size) if len(members) >= self.bag_size
                    else random.choices(members, k=self.bag_size))
        else:
            pick = members
        return torch.stack([self.slices[i][0] for i in pick]), self.labels[idx], idx


class AttentionMIL(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "efficientnet_b0", pretrained: bool = True,
                 hidden: int = 128, dropout: float = 0.25):
        super().__init__()
        self.encoder = Encoder(backbone, pretrained)
        dim = self.encoder.channels[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.attn_v = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh())
        self.attn_u = nn.Sequential(nn.Linear(dim, hidden), nn.Sigmoid())
        self.attn_w = nn.Linear(hidden, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, num_classes))

    def forward(self, bags: torch.Tensor):
        """``bags``: [B, N, 3, H, W] -> (logits [B, C], attention [B, N])."""
        b, n = bags.shape[:2]
        feats = self.pool(self.encoder(bags.flatten(0, 1))[-1]).flatten(1).view(b, n, -1)
        scores = self.attn_w(self.attn_v(feats) * self.attn_u(feats)).squeeze(-1)
        attn = torch.softmax(scores.float(), dim=1)
        bag = (attn.unsqueeze(-1) * feats.float()).sum(1)
        return self.classifier(bag), attn


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--disease", choices=sorted(DISEASES), required=True)
    p.add_argument("--backbone", choices=BACKBONES, default="efficientnet_b0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bags-per-batch", type=int, default=4)
    p.add_argument("--bag-size", type=int, default=16, help="Slices sampled per patient during training.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--class-weighting", action="store_true")
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--cv-folds", type=int, default=0)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--data-root", default=".")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--limit-per-class", type=int, default=None)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def run_epoch(model, loader, device, num_classes, cls_weights=None, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    logits_all, labels_all, idx_all, attn_all = [], [], [], []
    use_amp = device.type == "cuda"
    with torch.set_grad_enabled(train):
        for bags, labels, idx in loader:
            bags, labels = bags.to(device, non_blocking=True), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits, attn = model(bags)
                loss = F.cross_entropy(logits.float(), labels, weight=cls_weights)
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
            attn_all.extend(a.detach().cpu() for a in attn)
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    cm = confusion_matrix(logits.argmax(1), labels, num_classes)
    return {"loss": total / max(n, 1), "cm": cm, "logits": logits, "labels": labels,
            "idx": torch.cat(idx_all), "attn": attn_all}


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = DISEASES[args.disease]
    nc = len(cfg.classes)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    name = args.run_name or f"mil_{args.backbone}_seed{args.seed}" + (
        f"_cv{args.cv_folds}f{args.fold}" if args.cv_folds else "")
    run_dir = Path(args.out_dir) / args.disease / name
    run_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(cfg, args.data_root, require_masks=False, limit_per_class=args.limit_per_class)
    splits = (kfold_split(samples, args.cv_folds, args.fold, args.val_frac, args.seed) if args.cv_folds
              else split_samples(samples, args.val_frac, args.test_frac, args.seed))
    check_no_group_leakage(splits)
    save_split(splits, run_dir / "split.csv")
    ds = {k: PatientBags(v, args.image_size, train=(k == "train"), bag_size=args.bag_size)
          for k, v in splits.items()}
    for k, d in ds.items():
        print(f"{k:5s}: {len(d):4d} patients, per class {np.bincount(d.labels, minlength=nc).tolist()}")
        if len(d) == 0:
            raise ValueError(f"Empty {k} split")
    kw = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(ds["train"], batch_size=args.bags_per_batch, shuffle=True, drop_last=False, **kw)
    val_loader = DataLoader(ds["val"], batch_size=1, shuffle=False, **kw)
    test_loader = DataLoader(ds["test"], batch_size=1, shuffle=False, **kw)

    model = AttentionMIL(nc, args.backbone, pretrained=not args.no_pretrained).to(device)
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
        tr = run_epoch(model, train_loader, device, nc, cls_weights, optimizer, scaler)
        va = run_epoch(model, val_loader, device, nc, cls_weights)
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
                        "epoch": epoch, "kind": "mil"}, run_dir / "best.pt")
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
    test_ds = ds["test"]
    groups = [test_ds.groups[i] for i in te["idx"].tolist()]
    cm = confusion_matrix(torch.from_numpy(preds), torch.from_numpy(labels), nc)
    rc = risk_coverage(probs, labels)
    patient = classification_metrics(cm, cfg.classes)
    result = {
        "disease": args.disease, "arch": "mil", "backbone": args.backbone, "seed": args.seed,
        "seg_weight": 0.0, "split_level": "group", "cv_folds": args.cv_folds,
        "fold": args.fold if args.cv_folds else None, "bag_size": args.bag_size,
        "best_epoch": best_epoch, "epochs_run": len(history), "best_val_macro_f1": best_f1,
        "test": patient, "test_patient_level": patient, "test_confusion_matrix": cm.tolist(),
        "test_ci95": group_bootstrap_ci(labels, preds, groups, nc, args.bootstrap, args.seed) if args.bootstrap
        else None,
        "calibration": {"temperature": temperature, "ece_before": expected_calibration_error(probs_raw, labels),
                        "ece_after": expected_calibration_error(probs, labels), "aurc": rc["aurc"],
                        **{k: v for k, v in rc.items() if k.startswith("risk_at")}},
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_groups": {k: len(d) for k, d in ds.items()},
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2))
    with (run_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "label", "pred", *[f"p_{c}" for c in cfg.classes], "top_slice", "top_attention"])
        for i, g, y, p, a in zip(te["idx"].tolist(), groups, labels, probs, te["attn"]):
            top = int(a.argmax())
            w.writerow([g, int(y), int(p.argmax()), *[f"{x:.5f}" for x in p],
                        test_ds.samples[test_ds.members[i][top]].image_path, f"{float(a[top]):.4f}"])

    from .plots import plot_calibration, plot_confusion_matrix, plot_history
    plot_confusion_matrix(cm, cfg.classes, run_dir / "confusion_matrix.png",
                          title=f"{args.disease} attention-MIL/{args.backbone}: test patients")
    plot_history(history, run_dir / "curves.png")
    plot_calibration(probs_raw, probs, labels, rc, run_dir / "calibration.png")
    print(f"TEST (epoch {best_epoch}): patient acc {patient['accuracy']:.4f} bal-acc "
          f"{patient['balanced_accuracy']:.4f} macro-F1 {patient['macro_f1']:.4f} -> {run_dir}")
    return result


if __name__ == "__main__":
    main()
