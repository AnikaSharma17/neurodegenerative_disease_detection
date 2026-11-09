from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from neurodx.config import DiseaseConfig


def _write(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


@pytest.fixture
def tiny_dataset(tmp_path):
    """Two classes, 3 series x 4 slices each, with PNG masks in {0,255}."""
    cfg = DiseaseConfig(
        name="tiny",
        image_root="data/tiny/mri",
        mask_root="data/tiny/masks",
        classes=("a", "b"),
        group_regex=r"^(.*)_\d+\.\w+$",
    )
    rng = np.random.default_rng(0)
    for label, cls in enumerate(cfg.classes):
        for series in range(3):
            for sl in range(4):
                name = f"s{series}_{sl:03d}"
                img = rng.integers(0, 255, (40, 48), dtype=np.uint8)
                img[:, :24] = 40 + 150 * label  # class-dependent signal
                mask = np.zeros((40, 48), np.uint8)
                mask[10:20, 10:30] = 255
                _write(tmp_path / cfg.image_root / cls / f"{name}.png", img)
                _write(tmp_path / cfg.mask_root / cls / f"mask_{name}.png", mask)
    return tmp_path, cfg
