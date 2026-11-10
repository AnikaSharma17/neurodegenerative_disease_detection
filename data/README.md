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

## Larger datasets with real subject IDs (OASIS, ADNI, PPMI)

The Kaggle Alzheimer set has only 200 patients and shuffled filenames. The datasets below
come with subject IDs, clinical labels and full 3D volumes. Each one needs a free data-use
agreement or application, so you have to register and download the data yourself.

| Dataset | Use for | Labels | Access |
|---|---|---|---|
| [OASIS-1](https://sites.wustl.edu/oasisbrains/) | Alzheimer staging | CDR 0 / 0.5 / 1 / 2 (416 subjects) | Accept the OASIS data-use terms |
| [ADNI](https://adni.loni.usc.edu) | Alzheimer (CN / MCI / AD) | Diagnosis per visit | Apply on LONI IDA |
| [PPMI](https://www.ppmi-info.org) | Parkinson's (PD / Control) | Cohort | Apply on LONI IDA |

`preprocessing/prepare_volumes.py` converts the downloads into the slice layout
`data/<dataset>/mri/<class>/<subject>__<scan>_s<k>.png`. The subject ID before `__` is the
split group, so every scan of a person stays in one split, even across classes.

```bash
pip install -r requirements-3d.txt

# OASIS-1: download oasis_cross-sectional_disc1..12.tar.gz (16.2 GB) from
# https://download.nrg.wustl.edu/data/ and extract them into one folder. Labels are read
# from each session's OAS1_xxxx_MR1.txt; --csv oasis_cross-sectional.csv also works.
python preprocessing/prepare_volumes.py oasis1 --src data/raw/oasis1/extracted --dataset oasis \
    --masks fastsurfer --fastsurfer-image deepmi/fastsurfer:cu132-v2.6.0 \
    --structures hippocampus+ventricles --volume-size 96 --workers 2

# ADNI / PPMI (IDA image folder + the collection CSV exported from IDA)
python preprocessing/prepare_volumes.py ida --src /path/to/ADNI --csv ADNI_collection.csv \
    --dataset adni --label-map CN=CN MCI=MCI AD=AD --masks synthseg --structures hippocampus+ventricles
python preprocessing/prepare_volumes.py ida --src /path/to/PPMI --csv PPMI_collection.csv \
    --dataset ppmi --label-map Control=Control PD=PD --masks fastsurfer

python -m neurodx.mil --disease oasis --cv-folds 5 --fold 0
python -m neurodx.cnn3d --disease oasis --cv-folds 5 --fold 0 --seg-weight 0.5
python scripts/run_cv.py --diseases oasis --models cnn mil cnn3d --report-name OASIS_CV_RESULTS.md
```

For OASIS, `prepare_volumes.py` also writes `data/oasis/subjects.csv` with age, sex,
education, SES, CDR, MMSE, eTIV and nWBV. `run_cv.py` uses it for two tabular baselines:

- Age alone, which shows how much of the signal is just age.
- Age + sex + nWBV + eTIV, the classic whole-brain atrophy measure.

MMSE and CDR are never used as features.

`--volume-size 96` also saves each scan as a 96³ volume under `data/<dataset>/volumes/`, for
the 3D CNN. Voxels are 2 mm, giving a 192 mm field of view. Each volume is cropped around
the brain but never rescaled, so differences in brain size and atrophy are preserved.

### Anatomical masks

`--masks` replaces the threshold pseudo-masks with real brain segmentations:

- `synthseg`: FreeSurfer's `mri_synthseg` (FreeSurfer 7.3 or later). It works on any
  contrast and resolution and runs on CPU.
- `fastsurfer`: the `deepmi/fastsurfer` Docker image in segmentation-only mode.
  - It runs on the GPU through `docker run --gpus all`. Pass `--fastsurfer-gpus ""` to use
    the CPU instead.
  - OASIS Analyze files are converted to NIfTI first.
  - The input is the unmasked image, skull included.
- `precomputed`: label volumes you already have, given as a `seg_path` column in a manifest.

`--structures` chooses the foreground: `brain`, `hippocampus`, `ventricles` or
`hippocampus+ventricles`. The last two are the classic atrophy markers in Alzheimer's.
