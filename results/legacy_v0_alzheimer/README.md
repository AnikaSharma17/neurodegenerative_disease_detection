# Legacy results (v0, invalid)

These files are kept only as the "before" picture.

These files were produced by the original `train_alzheimer.py` (removed in v0.2.0).
They are kept for comparison, but the audit found defects that invalidate them:

- **Segmentation metrics are meaningless.** No mask file was ever matched, so every
  target was all-zero, and `val_miou = val_dice = 1.0` is an artefact.
- **The confusion matrices add up all 30 epochs** (they total 38,400 = 30 × 1,280),
  rather than showing the final model.
- **Validation images were augmented**, there was no held-out test set, and the
  checkpoint was picked on the broken mIoU.
- **The split was unseeded**, so each backbone was scored on a different validation set.
- **"acc" is torchmetrics' macro-averaged accuracy**, not plain accuracy.

Re-run with `python -m neurodx.train` and replace these with the outputs from
`runs/<disease>/<arch>_<backbone>_seed<seed>/`.
