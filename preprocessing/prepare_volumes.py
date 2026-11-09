"""Turn 3D MRI volumes with real subject IDs into the 2D slice layout used by neurodx.

Supported sources (all need you to accept the dataset's data-use agreement and
download the data yourself; see data/README.md):

  oasis1    OASIS-1 cross-sectional: subject folders OAS1_xxxx_MR1 plus
            oasis_cross-sectional.csv (CDR 0 / 0.5 / 1 / 2 -> 4 classes)
  ida       ADNI or PPMI downloads from LONI IDA: the image folder plus the
            collection CSV (Image Data ID, Subject, Group, ...)
  manifest  any CSV with columns subject,label,path[,seg_path]

For every volume it writes ``--slices`` axial slices from inside the brain, named
``<subject>__<scan>_s<k>.png``. The subject ID before ``__`` becomes the split group,
so all scans and visits of one person stay in one split.

Masks (``--masks``):
  none         no masks (train with --seg-weight 0)
  synthseg     run FreeSurfer's ``mri_synthseg`` (FreeSurfer >= 7.3) on each volume
  fastsurfer   run the FastSurfer Docker image in segmentation-only mode
  precomputed  use the manifest's seg_path column (any FreeSurfer-style label volume)

``--structures`` picks which labels become the foreground: brain (all tissue),
hippocampus, ventricles, or hippocampus+ventricles (atrophy markers in Alzheimer's).

    python preprocessing/prepare_volumes.py oasis1 --src ~/oasis1 --csv oasis_cross-sectional.csv \
        --dataset oasis --masks synthseg
    python preprocessing/prepare_volumes.py ida --src ~/ppmi --csv PPMI_collection.csv --dataset ppmi \
        --label-map PD=PD Control=Control --masks fastsurfer
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# FreeSurfer aseg labels.
STRUCTURES = {
    "hippocampus": {17, 53},
    "ventricles": {4, 5, 14, 15, 43, 44},
}
STRUCTURES["hippocampus+ventricles"] = STRUCTURES["hippocampus"] | STRUCTURES["ventricles"]
NON_BRAIN = {0, 24}  # background, CSF
OASIS_CDR = {0.0: "NonDemented", 0.5: "VeryMildDemented", 1.0: "MildDemented", 2.0: "ModerateDemented"}
VOLUME_EXTS = (".nii", ".nii.gz", ".img", ".mgz")


@dataclass
class Scan:
    subject: str
    scan: str
    label: str
    path: Path
    seg_path: Path | None = None


def _nib():
    try:
        import nibabel
    except ImportError as e:  # pragma: no cover
        raise SystemExit("nibabel is required: pip install -r requirements-3d.txt") from e
    return nibabel


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", name).strip("-")


# ---------------------------------------------------------------- manifests
def oasis1_scans(src: Path, csv_path: Path, include_young: bool = False) -> list[Scan]:
    """OASIS-1: one MR1 session per subject; MR2 reliability rescans are skipped so
    each person counts once. Subjects without a CDR (mostly under 60) are skipped unless
    ``include_young``, which labels them NonDemented."""
    scans = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sid = row["ID"].strip()
            if not sid.endswith("_MR1"):
                continue
            cdr = row.get("CDR", "").strip()
            if cdr == "" and not include_young:
                continue
            label = OASIS_CDR.get(float(cdr) if cdr else 0.0)
            if label is None:
                continue
            hits = sorted(src.rglob(f"{sid}*masked_gfc.img")) or sorted(src.rglob(f"{sid}*_t88_*.img"))
            if not hits:
                continue
            scans.append(Scan(sid.rsplit("_MR", 1)[0], sid, label, hits[0]))
    return scans


def ida_scans(src: Path, csv_path: Path, label_map: dict[str, str], all_visits: bool = False) -> list[Scan]:
    """ADNI/PPMI IDA download: match each Image Data ID to a file whose path contains
    ``I<id>`` as a folder or as a ``_I<id>`` filename token. By default only each
    subject's earliest scan is kept, so every person has one label (diagnoses can change
    between visits)."""
    index: dict[str, Path] = {}
    for p in src.rglob("*"):
        if p.is_file() and p.name.lower().endswith(VOLUME_EXTS):
            for m in re.finditer(r"(?:^|[_/\\])(I\d+)(?=[_./\\]|$)", str(p.relative_to(src))):
                index.setdefault(m.group(1), p)
    rows = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            group = row.get("Group", "").strip()
            if group not in label_map:
                continue
            image_id = row.get("Image Data ID", "").strip()
            image_id = image_id if image_id.startswith("I") else f"I{image_id}"
            if image_id in index:
                rows.append((_date_key(row.get("Acq Date", "")), row["Subject"].strip(), image_id, group))
    rows.sort()
    scans, seen = [], set()
    for _, subject, image_id, group in rows:
        if subject in seen and not all_visits:
            continue
        seen.add(subject)
        scans.append(Scan(subject, image_id, label_map[group], index[image_id]))
    return scans


def _date_key(text: str) -> tuple:
    """IDA dates look like 9/27/2010; unknown formats sort last."""
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", text.strip())
    return (int(m.group(3)), int(m.group(1)), int(m.group(2))) if m else (9999, 0, 0)


def manifest_scans(csv_path: Path) -> list[Scan]:
    scans = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f)):
            path = Path(row["path"])
            seg = Path(row["seg_path"]) if row.get("seg_path") else None
            scans.append(Scan(row["subject"], row.get("scan") or path.name.split(".")[0] or str(i),
                              row["label"], path, seg))
    return scans


# ---------------------------------------------------------------- segmentation
def run_synthseg(vol: Path, work: Path) -> tuple[Path, Path]:
    exe = shutil.which("mri_synthseg")
    if not exe:
        raise SystemExit("mri_synthseg not found. Install FreeSurfer >= 7.3 and source its setup script.")
    seg, res = work / "seg.nii.gz", work / "resampled.nii.gz"
    subprocess.run([exe, "--i", str(vol), "--o", str(seg), "--resample", str(res), "--cpu"], check=True)
    return res, seg  # both on SynthSeg's 1 mm grid


def run_fastsurfer(vol: Path, work: Path, image: str, license_file: str | None) -> tuple[Path, Path]:
    if not shutil.which("docker"):
        raise SystemExit("Docker is required for --masks fastsurfer.")
    cmd = ["docker", "run", "--rm", "-v", f"{vol.parent.resolve()}:/in:ro", "-v", f"{work.resolve()}:/out"]
    if license_file:
        cmd += ["-v", f"{Path(license_file).resolve()}:/fs_license.txt:ro"]
    cmd += [image, "--t1", f"/in/{vol.name}", "--sid", "s", "--sd", "/out", "--seg_only"]
    if license_file:
        cmd += ["--fs_license", "/fs_license.txt"]
    subprocess.run(cmd, check=True)
    mri = work / "s" / "mri"
    return mri / "orig.mgz", mri / "aparc.DKTatlas+aseg.deep.mgz"  # conformed 1 mm 256^3 grid


def mask_from_labels(seg: np.ndarray, structures: str) -> np.ndarray:
    if structures == "brain":
        return ~np.isin(seg, list(NON_BRAIN))
    if structures not in STRUCTURES:
        raise ValueError(f"unknown --structures {structures!r}")
    return np.isin(seg, list(STRUCTURES[structures]))


# ---------------------------------------------------------------- slicing
def load_canonical(path: Path) -> np.ndarray:
    nib = _nib()
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data[..., 0] if data.ndim == 4 else data


def pick_slices(volume: np.ndarray, n: int, margin: float = 0.1) -> list[int]:
    """Evenly spaced axial slices inside the brain extent, skipping ``margin`` at each end
    (the top and bottom slices are mostly skull and noise)."""
    occupied = np.where((volume > np.percentile(volume, 60)).any(axis=(0, 1)))[0]
    lo, hi = (occupied[0], occupied[-1]) if len(occupied) else (0, volume.shape[2] - 1)
    span = hi - lo
    lo, hi = lo + int(margin * span), hi - int(margin * span)
    return sorted(set(np.linspace(lo, hi, n).round().astype(int).tolist()))


def to_uint8(sl: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(sl, [1, 99])
    return (np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def write_slices(scan: Scan, vol: np.ndarray, seg: np.ndarray | None, out_img: Path, out_mask: Path | None,
                 n: int, structures: str) -> int:
    if seg is not None and seg.shape != vol.shape:
        raise ValueError(f"{scan.scan}: segmentation shape {seg.shape} != volume shape {vol.shape}")
    base = f"{safe(scan.subject)}__{safe(scan.scan)}"
    (out_img / scan.label).mkdir(parents=True, exist_ok=True)
    if out_mask is not None:
        (out_mask / scan.label).mkdir(parents=True, exist_ok=True)
    idx = pick_slices(vol, n)
    for k, z in enumerate(idx):
        # Axial slice in RAS: rotate so anterior is up, as radiologists view it.
        Image.fromarray(np.rot90(to_uint8(vol[:, :, z]))).save(out_img / scan.label / f"{base}_s{k:02d}.png")
        if seg is not None and out_mask is not None:
            m = np.rot90(mask_from_labels(seg[:, :, z], structures)).astype(np.uint8) * 255
            Image.fromarray(m).save(out_mask / scan.label / f"mask_{base}_s{k:02d}.png")
    return len(idx)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", choices=("oasis1", "ida", "manifest"))
    p.add_argument("--src", type=Path, help="Folder with the downloaded volumes (oasis1, ida).")
    p.add_argument("--csv", type=Path, required=True, help="OASIS CSV, IDA collection CSV, or manifest.")
    p.add_argument("--dataset", required=True, help="Output name: data/<dataset>/mri and data/<dataset>/masks.")
    p.add_argument("--label-map", nargs="+", default=[], metavar="GROUP=CLASS",
                   help="ida: map IDA Group values to class folders, e.g. AD=AD CN=CN MCI=MCI.")
    p.add_argument("--all-visits", action="store_true",
                   help="ida: keep every scan, not just each subject's first (all stay in one split group).")
    p.add_argument("--include-young", action="store_true", help="oasis1: keep subjects without CDR as NonDemented.")
    p.add_argument("--slices", type=int, default=32)
    p.add_argument("--masks", choices=("none", "synthseg", "fastsurfer", "precomputed"), default="none")
    p.add_argument("--structures", choices=("brain", *STRUCTURES), default="brain")
    p.add_argument("--fastsurfer-image", default="deepmi/fastsurfer:latest")
    p.add_argument("--fs-license", default=None, help="FreeSurfer license file, if your FastSurfer version needs it.")
    p.add_argument("--out-root", type=Path, default=Path("data"))
    p.add_argument("--limit", type=int, default=None, help="Debug: process only the first N scans.")
    args = p.parse_args(argv)

    if args.source == "oasis1":
        scans = oasis1_scans(args.src, args.csv, args.include_young)
    elif args.source == "ida":
        label_map = dict(kv.split("=", 1) for kv in args.label_map)
        if not label_map:
            p.error("ida needs --label-map, e.g. --label-map AD=AD CN=CN")
        scans = ida_scans(args.src, args.csv, label_map, args.all_visits)
    else:
        scans = manifest_scans(args.csv)
    if args.limit:
        scans = scans[: args.limit]
    if not scans:
        raise SystemExit("No scans matched. Check --src, --csv and --label-map.")
    if args.masks == "precomputed" and any(s.seg_path is None for s in scans):
        raise SystemExit("--masks precomputed needs a seg_path for every row of the manifest.")

    out_img = args.out_root / args.dataset / "mri"
    out_mask = None if args.masks == "none" else args.out_root / args.dataset / "masks"
    written, counts = 0, {}
    for i, scan in enumerate(scans, 1):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            vol_path, seg_path = scan.path, scan.seg_path
            if args.masks == "synthseg":
                vol_path, seg_path = run_synthseg(scan.path, work)
            elif args.masks == "fastsurfer":
                vol_path, seg_path = run_fastsurfer(scan.path, work, args.fastsurfer_image, args.fs_license)
            vol = load_canonical(vol_path)
            seg = load_canonical(seg_path).round().astype(np.int32) if seg_path else None
            written += write_slices(scan, vol, seg, out_img, out_mask, args.slices, args.structures)
        counts[scan.label] = counts.get(scan.label, 0) + 1
        print(f"[{i}/{len(scans)}] {scan.subject} {scan.scan} -> {scan.label}", flush=True)
    print(f"{written} slices from {len(scans)} scans of {len({s.subject for s in scans})} subjects; "
          f"scans per class {counts} -> {out_img}")


if __name__ == "__main__":
    main()
