# Data

Images are not stored in git. Place the datasets here with this layout. Scripts expect
to be run from the repository root.

```
data/
├── alzheimer/
│   ├── mri/
│   │   ├── OriginalDataset/{NonDemented,VeryMildDemented,MildDemented,ModerateDemented}/*.jpg
│   │   └── AugmentedAlzheimerDataset/...      (not used: augmented copies of the originals)
│   └── masks/OriginalDataset/<class>/mask_*   (python preprocessing/generate_masks.py --disease alzheimer)
└── parkinson/
    ├── mri/{normal,parkinson}/*.png
    └── masks/{normal,parkinson}/mask_*.png
```

## Alzheimer: patient groups

The Alzheimer folders match the public Kaggle "Augmented Alzheimer MRI Dataset"
(OriginalDataset). Most filenames are shuffled (`nonDem123.jpg`), so there is no
patient ID to split on. `python -m neurodx.grouping --disease alzheimer` recovers
patients by linking slices whose pixel correlation exceeds 0.97. It writes
`metadata/alzheimer_groups.csv`, which is committed.

Results on 2026-11-09:

| Class | Images | Recovered patients | Slices per patient |
|---|---|---|---|
| NonDemented | 3,200 | 100 | 32 |
| VeryMildDemented | 2,240 | 70 | 32 |
| MildDemented | 896 | 28 | 32 |
| ModerateDemented | 64 | **2** | 32 |

1,252 files carry an explicit patient ID (`<slice> (<patient>).jpg`). On those files the
grouping keeps every same-patient pair together and never merges two patients
(precision and recall both 1.000). The dataset is therefore 200 patients, not 6,400
independent samples, and ModerateDemented is two people.

## Parkinson

The source is undocumented, so record its origin and licence here. Series names suggest
different scanner vendors per class: Siemens-style names in `normal`, Philips-style names
in `parkinson`. Splits are grouped by series (`<series>_<slice>.png`).
