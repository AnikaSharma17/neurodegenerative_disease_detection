"""Patient-level k-fold cross-validation, pooled into one out-of-fold estimate.

With 200 Alzheimer patients a single 15% test split holds only ~32 patients, so test
metrics swing by +/-0.1 between seeds. Here every patient is tested exactly once
(5 folds), and metrics are computed on the pooled out-of-fold predictions:

  slice CNN   : multitask EfficientNet-B0, slice probabilities averaged per patient
  MIL         : attention-MIL over all slices of a patient (neurodx.mil)
  3D CNN      : whole-volume 3D ResNet (neurodx.cnn3d), if prepare_volumes wrote volumes
  baselines   : majority, logistic regression on histogram / thumbnail features, and
                for datasets with data/<d>/subjects.csv (OASIS): logistic regression on
                age alone (a confound check) and on age + sex + nWBV + eTIV

Resumable: folds that already have test_metrics.json are skipped. Writes
results/CV_RESULTS.md and runs/<disease>/cv<k>_summary.json.

    python scripts/run_cv.py                      # both diseases, 5 folds
    python scripts/run_cv.py --diseases oasis --models cnn mil cnn3d
    python scripts/run_cv.py --report-only
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import features  # noqa: E402
from neurodx.config import DISEASES  # noqa: E402
from neurodx.data import collect_samples, kfold_split  # noqa: E402
from neurodx.metrics import aggregate_by_group, classification_metrics, confusion_matrix  # noqa: E402

COMMON = ["--epochs", "25", "--patience", "8", "--num-workers", "8", "--class-weighting"]


def run(cmd: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(cmd), flush=True)
    with log.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, cwd=ROOT)


CNN3D_ARGS = ["--epochs", "60", "--patience", "15", "--num-workers", "4", "--class-weighting",
              "--batch-size", "8", "--seg-weight", "0.5"]


def train_folds(disease: str, folds: int, seed: int, runs: Path, models=("cnn", "mil")) -> None:
    for fold in range(folds):
        jobs = {
            "cnn": (f"multitask_efficientnet_b0_seg1_group_seed{seed}_cv{folds}f{fold}",
                    [sys.executable, "-m", "neurodx.train", "--disease", disease, "--batch-size", "32", *COMMON]),
            "mil": (f"mil_efficientnet_b0_seed{seed}_cv{folds}f{fold}",
                    [sys.executable, "-m", "neurodx.mil", "--disease", disease, "--bags-per-batch", "4",
                     "--bag-size", "16", *COMMON]),
            "cnn3d": (f"cnn3d_seg_seed{seed}_cv{folds}f{fold}",
                      [sys.executable, "-m", "neurodx.cnn3d", "--disease", disease, *CNN3D_ARGS]),
        }
        for name, cmd in (jobs[m] for m in models):
            if (runs / disease / name / "test_metrics.json").exists():
                continue
            run(cmd + ["--seed", str(seed), "--cv-folds", str(folds), "--fold", str(fold)],
                runs / "logs" / f"{disease}_{name}.log")


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pooled_patient_probs(disease: str, prefix: str, folds: int, runs: Path, classes) -> tuple:
    """Concatenate out-of-fold predictions and average slice probabilities per patient."""
    probs, labels, groups, fold_ids = [], [], [], []
    for fold in range(folds):
        path = runs / disease / f"{prefix}_cv{folds}f{fold}" / "test_predictions.csv"
        if not path.exists():
            return None
        for r in read_rows(path):
            probs.append([float(r[f"p_{c}"]) for c in classes])
            labels.append(int(r["label"]))
            groups.append(r["group"])
            fold_ids.append(fold)
    probs, labels = np.array(probs), np.array(labels)
    gp, gl, names = aggregate_by_group(probs, labels, groups)
    fold_of = dict(zip(groups, fold_ids))
    slice_cm = confusion_matrix(torch.from_numpy(probs.argmax(1)), torch.from_numpy(labels), len(classes))
    slice_m = classification_metrics(slice_cm, classes) if len(groups) > len(names) else None
    return gp, gl, names, np.array([fold_of[n] for n in names]), slice_m


def baseline_patient_probs(disease: str, folds: int, seed: int, classes) -> dict:
    cfg = DISEASES[disease]
    samples = collect_samples(cfg, ROOT, require_masks=False)
    feats = features([s.image_path for s in samples])
    index = {s.image_path: i for i, s in enumerate(samples)}
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    nc = len(classes)
    tabular = subject_features(ROOT / Path(cfg.image_root).parent / "subjects.csv")
    out: dict[str, dict] = defaultdict(lambda: {"probs": [], "labels": [], "groups": [], "folds": []})
    for fold in range(folds):
        sp = kfold_split(samples, folds, fold, 0.15, seed)
        tr = np.array([index[s.image_path] for s in sp["train"] + sp["val"]])
        te = np.array([index[s.image_path] for s in sp["test"]])
        y_tr = np.array([samples[i].label for i in tr])
        models = {"majority": None, "logreg_histogram": "histogram", "logreg_thumbnail": "thumbnail"}
        for name, feat in models.items():
            p = np.zeros((len(te), nc))
            if feat is None:
                p[:, np.bincount(y_tr, minlength=nc).argmax()] = 1
            else:
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
                clf.fit(feats[feat][tr], y_tr)
                p[:, clf.classes_] = clf.predict_proba(feats[feat][te])
            o = out[name]
            o["probs"].append(p)
            o["labels"] += [samples[i].label for i in te]
            o["groups"] += [samples[i].group for i in te]
            o["folds"] += [fold] * len(te)
        if tabular:  # one row per patient, same folds
            tr_g = sorted({samples[i].group for i in tr})
            te_g = sorted({samples[i].group for i in te})
            lab = {s.group: s.label for s in samples}
            for name, cols in TABULAR.items():
                X_tr = np.array([tabular[g][cols] for g in tr_g])
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
                clf.fit(X_tr, [lab[g] for g in tr_g])
                p = np.zeros((len(te_g), nc))
                p[:, clf.classes_] = clf.predict_proba(np.array([tabular[g][cols] for g in te_g]))
                o = out[name]
                o["probs"].append(p)
                o["labels"] += [lab[g] for g in te_g]
                o["groups"] += te_g
                o["folds"] += [fold] * len(te_g)
    result = {}
    for name, o in out.items():
        probs, labels = np.concatenate(o["probs"]), np.array(o["labels"])
        gp, gl, names = aggregate_by_group(probs, labels, o["groups"])
        fold_of = dict(zip(o["groups"], o["folds"]))
        cm = confusion_matrix(torch.from_numpy(probs.argmax(1)), torch.from_numpy(labels), nc)
        slice_m = classification_metrics(cm, classes) if len(o["groups"]) > len(names) else None
        result[name] = (gp, gl, names, np.array([fold_of[n] for n in names]), slice_m)
    return result


# Tabular baselines (column indices into subject_features rows): age, sex, nWBV, eTIV.
TABULAR = {"logreg_age": [0], "logreg_age_sex_nWBV_eTIV": [0, 1, 2, 3]}


def subject_features(path: Path) -> dict[str, np.ndarray]:
    """``group -> [age, female, nWBV, eTIV]`` from prepare_volumes' subjects.csv; empty if absent.
    Missing values are imputed with the column median."""
    if not path.is_file():
        return {}
    rows = read_rows(path)

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    X = np.array([[num(r.get("Age")), float(str(r.get("M/F", "")).upper().startswith("F")),
                   num(r.get("nWBV")), num(r.get("eTIV"))] for r in rows])
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), med, X)
    return {r["group"]: x for r, x in zip(rows, X)}


def patient_metrics(gp, gl, classes) -> dict:
    cm = confusion_matrix(torch.from_numpy(gp.argmax(1)), torch.from_numpy(gl), len(classes))
    return classification_metrics(cm, classes)


def bootstrap(gp, gl, classes, n_boot=2000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    keys = ("accuracy", "balanced_accuracy", "macro_f1")
    stats = {k: [] for k in keys}
    for _ in range(n_boot):
        i = rng.integers(0, len(gl), len(gl))
        m = patient_metrics(gp[i], gl[i], classes)
        for k in keys:
            stats[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in stats.items()}


def paired_diff(a, b, gl, names_a, names_b, classes, n_boot=2000, seed=0) -> dict:
    """Bootstrap the macro-F1 difference a - b on the same resampled patients."""
    order = {n: i for i, n in enumerate(names_b)}
    b = b[[order[n] for n in names_a]]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(gl), len(gl))
        fa, fb = patient_metrics(a[i], gl[i], classes), patient_metrics(b[i], gl[i], classes)
        diffs.append(fa["macro_f1"] - fb["macro_f1"])
    diffs = np.array(diffs)
    return {"mean": float(np.nanmean(diffs)), "ci95": [float(np.nanpercentile(diffs, 2.5)),
            float(np.nanpercentile(diffs, 97.5))], "p_a_better": float(np.nanmean(diffs > 0))}


def per_fold_f1(gp, gl, folds_arr, k, classes) -> list[float]:
    return [patient_metrics(gp[folds_arr == f], gl[folds_arr == f], classes)["macro_f1"] for f in range(k)]


CNN3D = "3D CNN (whole volume, + seg loss)"


def report(diseases, folds: int, seed: int, runs: Path, out: Path, report_name: str = "CV_RESULTS.md") -> None:
    md = ["# Cross-validated results", "",
          f"Generated by `python scripts/run_cv.py`. {folds}-fold patient-level cross-validation (seed {seed}):",
          "every patient (Alzheimer) or MRI series (Parkinson) is tested exactly once, and metrics are",
          "computed on the pooled out-of-fold predictions. CIs: 2,000 patient bootstrap resamples.", ""]
    for disease in diseases:
        classes = DISEASES[disease].classes
        methods = dict(baseline_patient_probs(disease, folds, seed, classes))
        for label, prefix in (("slice CNN (EfficientNet-B0, mean of slices)",
                               f"multitask_efficientnet_b0_seg1_group_seed{seed}"),
                              ("attention-MIL (EfficientNet-B0)", f"mil_efficientnet_b0_seed{seed}"),
                              (CNN3D, f"cnn3d_seg_seed{seed}")):
            r = pooled_patient_probs(disease, prefix, folds, runs, classes)
            if r is not None:
                methods[label] = r
        md += [f"## {disease.capitalize()} ({len(next(iter(methods.values()))[1])} patients/series)", "",
               "| Method | Patient acc. | Patient bal. acc. | Patient macro-F1 | 95% CI macro-F1 "
               "| Per-fold macro-F1 | Slice macro-F1 |",
               "|---|---|---|---|---|---|---|"]
        summary = {}
        for name, (gp, gl, names, fa, slice_m) in methods.items():
            m = patient_metrics(gp, gl, classes)
            ci = bootstrap(gp, gl, classes)
            pf = per_fold_f1(gp, gl, fa, folds, classes)
            summary[name] = {"patient": m, "ci95": ci, "per_fold_macro_f1": pf, "slice": slice_m}
            slice_txt = f"{slice_m['macro_f1']:.3f}" if slice_m else "–"
            md.append(f"| {name} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | {m['macro_f1']:.3f} | "
                      f"{ci['macro_f1'][0]:.2f}–{ci['macro_f1'][1]:.2f} | "
                      f"{np.nanmean(pf):.3f} ± {np.nanstd(pf, ddof=1):.3f} | {slice_txt} |")
        md.append("")
        md += ["Per-class patient recall:", "", "| Method | " + " | ".join(classes) + " |",
               "|---|" + "---|" * len(classes)]
        for name, s in summary.items():
            md.append(f"| {name} | " + " | ".join(f"{s['patient']['per_class'][c]['recall']:.2f} "
                                                  f"(n={s['patient']['per_class'][c]['support']})"
                                                  for c in classes) + " |")
        md.append("")
        cnn = "slice CNN (EfficientNet-B0, mean of slices)"
        mil = "attention-MIL (EfficientNet-B0)"
        tab = "logreg_age_sex_nWBV_eTIV"
        comparisons = [(mil, cnn), (cnn, "logreg_histogram"), (cnn, "logreg_thumbnail"), (mil, "logreg_histogram"),
                       (CNN3D, cnn), (CNN3D, "logreg_histogram"), (CNN3D, tab), (cnn, tab), (cnn, "logreg_age")]
        lines = []
        for a, b in comparisons:
            if a in methods and b in methods:
                d = paired_diff(methods[a][0], methods[b][0], methods[a][1], methods[a][2], methods[b][2], classes)
                summary[f"{a} - {b}"] = d
                lines.append(f"- {a} minus {b}: {d['mean']:+.3f} macro-F1 (95% CI {d['ci95'][0]:+.2f} to "
                             f"{d['ci95'][1]:+.2f}; better in {100 * d['p_a_better']:.0f}% of resamples)")
        if lines:
            md += ["Paired bootstrap differences (same resampled patients):", "", *lines, ""]
        (runs / disease / f"cv{folds}_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    out.mkdir(parents=True, exist_ok=True)
    (out / report_name).write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--diseases", nargs="+", default=["alzheimer", "parkinson"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    p.add_argument("--models", nargs="+", choices=("cnn", "mil", "cnn3d"), default=["cnn", "mil"])
    p.add_argument("--report-name", default="CV_RESULTS.md")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    runs = ROOT / args.runs
    if not args.report_only:
        for d in args.diseases:
            train_folds(d, args.folds, args.seed, runs, args.models)
    report(args.diseases, args.folds, args.seed, runs, ROOT / args.out, args.report_name)


if __name__ == "__main__":
    main()
