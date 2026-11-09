"""Sample discovery, leak-aware splitting and the image+mask dataset."""
from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

from .config import IMAGE_EXTS, IMAGENET_MEAN, IMAGENET_STD, DiseaseConfig

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    image_path: str
    mask_path: str | None
    label: int
    group: str


def find_mask(mask_dir: Path, image_name: str) -> Path | None:
    """Masks are stored as ``mask_<stem>.png`` (current generator, preferred) or
    ``mask_<image file name>`` (older JPEG masks). Return the first that exists."""
    stem = Path(image_name).stem
    for candidate in (f"mask_{stem}.png", f"mask_{image_name}"):
        path = mask_dir / candidate
        if path.is_file():
            return path
    return None


def collect_samples(
    cfg: DiseaseConfig,
    root: str | Path = ".",
    require_masks: bool = True,
    limit_per_class: int | None = None,
    split_level: str = "group",
) -> list[Sample]:
    """``split_level="image"`` gives every image its own group. That reproduces
    the leaky slice-level split, and exists only to measure the leakage."""
    if split_level not in ("group", "image"):
        raise ValueError(f"split_level must be 'group' or 'image', not {split_level!r}")
    root = Path(root)
    group_re = re.compile(cfg.group_regex) if cfg.group_regex else None
    group_map: dict[str, str] = {}
    if cfg.groups_file and split_level == "group":
        gpath = root / cfg.groups_file
        if not gpath.is_file():
            raise FileNotFoundError(
                f"Patient groups file {gpath} not found. Run `python -m neurodx.grouping "
                f"--disease {cfg.name}`, or pass --split-level image (leaky)."
            )
        with gpath.open(newline="", encoding="utf-8") as f:
            group_map = {row["image"]: row["group"] for row in csv.DictReader(f)}
    samples: list[Sample] = []
    missing: list[str] = []
    for label, cls in enumerate(cfg.classes):
        img_dir = root / cfg.image_root / cls
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {img_dir}")
        names = sorted(f.name for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if limit_per_class:
            names = names[:limit_per_class]
        for name in names:
            mask = find_mask(root / cfg.mask_root / cls, name)
            if mask is None:
                missing.append(f"{cls}/{name}")
            key = f"{cls}/{name}"
            if split_level == "image":
                group = key
            elif group_map:
                if key not in group_map:
                    raise KeyError(f"{key} missing from {cfg.groups_file}; regenerate it.")
                group = group_map[key]
            else:
                match = group_re.match(name) if group_re else None
                group = f"{cls}/{match.group(1)}" if match else key
            samples.append(Sample(str(img_dir / name), str(mask) if mask else None, label, group))
    if not samples:
        raise FileNotFoundError(f"No images found under {root / cfg.image_root}")
    if missing and require_masks:
        raise FileNotFoundError(
            f"{len(missing)} of {len(samples)} images have no mask under "
            f"{root / cfg.mask_root} (first: {missing[0]}). Run "
            "preprocessing/generate_masks.py, or train with --seg-weight 0."
        )
    return samples


def split_samples(
    samples: list[Sample], val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
) -> dict[str, list[Sample]]:
    """Stratified, group-aware split. Whole groups are assigned to one split, so
    slices from the same series never appear in both train and evaluation."""
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        groups[s.group].append(s)
    by_class: dict[int, list[str]] = defaultdict(list)
    for g, members in groups.items():
        majority = Counter(m.label for m in members).most_common(1)[0][0]
        by_class[majority].append(g)

    rng = random.Random(seed)
    out: dict[str, list[Sample]] = {k: [] for k in SPLITS}
    for label in sorted(by_class):
        gs = sorted(by_class[label])
        rng.shuffle(gs)
        total = sum(len(groups[g]) for g in gs)
        targets = {"test": test_frac * total, "val": val_frac * total}
        counts = {"test": 0, "val": 0}
        for i, g in enumerate(gs):
            remaining_groups = len(gs) - i
            dest = "train"
            if remaining_groups > 1:  # always leave at least one group for training
                for split in ("test", "val"):
                    if counts[split] < targets[split]:
                        dest = split
                        break
            if dest != "train":
                counts[dest] += len(groups[g])
            out[dest].extend(groups[g])
    for k in SPLITS:
        out[k].sort(key=lambda s: s.image_path)
    return out


def save_split(splits: dict[str, list[Sample]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "image_path", "mask_path", "label", "group"])
        for split in SPLITS:
            for s in splits[split]:
                w.writerow([split, s.image_path, s.mask_path or "", s.label, s.group])


def load_split(path: str | Path) -> dict[str, list[Sample]]:
    out: dict[str, list[Sample]] = {k: [] for k in SPLITS}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["split"]].append(
                Sample(row["image_path"], row["mask_path"] or None, int(row["label"]), row["group"])
            )
    return out


def check_no_group_leakage(splits: dict[str, list[Sample]]) -> None:
    seen: dict[str, str] = {}
    for split in SPLITS:
        for s in splits[split]:
            if seen.setdefault(s.group, split) != split:
                raise ValueError(f"Group {s.group!r} appears in {seen[s.group]} and {split}")


class MRIDataset(Dataset):
    """Returns ``(image, mask, label, has_mask)``.

    Geometric augmentation is applied identically to image and mask; the mask
    is binarised (>127 -> 1) and resized with nearest-neighbour so labels stay
    in {0, 1}. Evaluation datasets (``train=False``) are deterministic.
    """

    def __init__(self, samples: list[Sample], image_size: int = 224, train: bool = False):
        self.samples = samples
        self.size = [image_size, image_size]
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = Image.open(s.image_path).convert("RGB")
        if s.mask_path:
            mask_arr = (np.array(Image.open(s.mask_path).convert("L")) > 127).astype(np.uint8)
            mask = Image.fromarray(mask_arr)
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.NEAREST)
        else:
            mask = Image.new("L", image.size, 0)

        image = TF.resize(image, self.size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        if self.train:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            angle = random.uniform(-15.0, 15.0)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        image_t = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image_t, mask_t, s.label, s.mask_path is not None
