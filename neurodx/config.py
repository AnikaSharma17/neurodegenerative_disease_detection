"""Dataset definitions and shared constants."""
from __future__ import annotations

from dataclasses import dataclass

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
NUM_SEG_CLASSES = 2  # background / foreground of the binary pseudo-mask
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DiseaseConfig:
    name: str
    image_root: str
    mask_root: str
    classes: tuple[str, ...]
    # Regex whose first capture group identifies a group of related images
    # (e.g. one MRI series). Images in one group never cross split boundaries.
    group_regex: str | None = None
    # CSV (columns: image, group) mapping "<class>/<file>" to a patient group.
    # Takes precedence over group_regex. Produced by ``python -m neurodx.grouping``.
    groups_file: str | None = None


DISEASES: dict[str, DiseaseConfig] = {
    "alzheimer": DiseaseConfig(
        name="alzheimer",
        image_root="data/alzheimer/mri/OriginalDataset",
        mask_root="data/alzheimer/masks/OriginalDataset",
        classes=("NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"),
        # Filenames are mostly shuffled (nonDem123.jpg), so patients are recovered by
        # image similarity: 200 patients x 32 slices. See neurodx/grouping.py.
        groups_file="metadata/alzheimer_groups.csv",
    ),
    "parkinson": DiseaseConfig(
        name="parkinson",
        image_root="data/parkinson/mri",
        mask_root="data/parkinson/masks",
        classes=("normal", "parkinson"),
        # Files look like <series-name>_<slice>.png; keep each series in one split.
        group_regex=r"^(.*)_\d+\.\w+$",
    ),
}
