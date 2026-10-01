# Dataset Setup — Carinthia-S

The dataset is **not bundled** in this ZIP (139 MB, CC BY 4.0). Download it once and place it as shown.

**Source:** C. Kofler and V. Hasić, *Carinthia-S dataset*, Zenodo, 2025. DOI: [10.5281/zenodo.16895427](https://doi.org/10.5281/zenodo.16895427)
File to download: `data.zip`

## Required layout

```
Semiconductor_Risk_Calibrated_Visual_Screening/
└── data/
    └── raw/
        └── carinthia_s/
            └── data/
                ├── images/          <- 4,591 SEM images (.jpg)
                ├── masks/           <- 4,591 binary masks (.png)
                └── carinthia-s.csv  <- columns: image_path, mask_path, filename, label
```

## PowerShell

```powershell
# from the project root, after downloading data.zip into the project root
Expand-Archive -Path .\data.zip -DestinationPath .\data\raw\carinthia_s -Force
Get-ChildItem .\data\raw\carinthia_s\data | Select-Object Name
python experiments\01_verify_dataset.py
```

If the archive extracts to a different folder name, move its contents so that `images\`, `masks\`
and `carinthia-s.csv` sit directly under `data\raw\carinthia_s\data\`.

Relative paths in the CSV are resolved automatically (relative to the CSV folder, its parents, or
`images\` / `masks\` by file name).

**Expected verification** (computed from disk, never hard-coded): 4,591 images, 4,591 masks,
480×480 grayscale, 224 empty masks (REFERENCE), 4,367 non-empty masks (ANOMALY).
