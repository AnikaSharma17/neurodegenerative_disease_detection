import numpy as np
import pytest
from PIL import Image

from neurodx.data import (
    MRIDataset,
    check_no_group_leakage,
    collect_samples,
    find_mask,
    load_split,
    save_split,
    split_samples,
)


def test_every_image_is_paired_with_its_mask(tiny_dataset):
    root, cfg = tiny_dataset
    samples = collect_samples(cfg, root)
    assert len(samples) == 24
    for s in samples:
        assert s.mask_path is not None
        assert s.mask_path.endswith(f"mask_{s.image_path.split('/')[-1].split(chr(92))[-1]}")


def test_legacy_jpeg_mask_name_is_found(tmp_path):
    (tmp_path / "mask_26 (19).jpg").write_bytes(b"x")
    assert find_mask(tmp_path, "26 (19).jpg").name == "mask_26 (19).jpg"


def test_missing_mask_raises_unless_masks_optional(tiny_dataset):
    root, cfg = tiny_dataset
    next((root / cfg.mask_root / "a").glob("*.png")).unlink()
    with pytest.raises(FileNotFoundError, match="have no mask"):
        collect_samples(cfg, root)
    samples = collect_samples(cfg, root, require_masks=False)
    assert sum(s.mask_path is None for s in samples) == 1


def test_groups_come_from_series_name(tiny_dataset):
    root, cfg = tiny_dataset
    groups = {s.group for s in collect_samples(cfg, root)}
    assert groups == {f"{c}/s{i}" for c in "ab" for i in range(3)}


def test_split_is_disjoint_grouped_stratified_and_deterministic(tiny_dataset, tmp_path):
    root, cfg = tiny_dataset
    samples = collect_samples(cfg, root)
    a = split_samples(samples, 0.3, 0.3, seed=1)
    b = split_samples(samples, 0.3, 0.3, seed=1)
    assert a == b
    check_no_group_leakage(a)
    paths = [s.image_path for k in a for s in a[k]]
    assert len(paths) == len(set(paths)) == len(samples)
    for k in a:  # every class present in every split
        assert {s.label for s in a[k]} == {0, 1}
    save_split(a, tmp_path / "split.csv")
    assert load_split(tmp_path / "split.csv") == a


def test_mask_is_binary_after_resize_and_augmentation(tiny_dataset):
    root, cfg = tiny_dataset
    samples = collect_samples(cfg, root)
    for train in (False, True):
        image, mask, label, has_mask = MRIDataset(samples, image_size=64, train=train)[0]
        assert image.shape == (3, 64, 64)
        assert mask.shape == (64, 64)
        assert set(mask.unique().tolist()) <= {0, 1}
        assert has_mask


def test_jpeg_artifacts_are_binarised(tmp_path):
    from neurodx.data import Sample

    img = np.full((32, 32), 100, np.uint8)
    mask = np.zeros((32, 32), np.uint8)
    mask[8:24, 8:24] = 255
    Image.fromarray(img).save(tmp_path / "x.png")
    Image.fromarray(mask).save(tmp_path / "mask_x.jpg", quality=50)
    _, m, _, _ = MRIDataset([Sample(str(tmp_path / "x.png"), str(tmp_path / "mask_x.jpg"), 0, "g")], 32)[0]
    assert set(m.unique().tolist()) == {0, 1}


def test_eval_dataset_is_deterministic(tiny_dataset):
    root, cfg = tiny_dataset
    ds = MRIDataset(collect_samples(cfg, root), image_size=64, train=False)
    assert ds[3][0].equal(ds[3][0])
