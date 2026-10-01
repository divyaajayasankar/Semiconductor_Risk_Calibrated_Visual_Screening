"""Create dist/Risk_Calibrated_Semiconductor_Visual_Screening.zip

    python scripts/package_project.py                 # code + models + results (+ dataset if present)
    python scripts/package_project.py --no-dataset    # exclude Carinthia-S images/masks
    python scripts/package_project.py --check         # readiness check only
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_NAME = "Risk_Calibrated_Semiconductor_Visual_Screening.zip"
TOP = "Semiconductor_Risk_Calibrated_Visual_Screening"

EXCLUDE_DIRS = {".venv", "venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".idea", ".vscode",
                ".git", "dist", ".mypy_cache", "cache", "_minted", "paper_build"}
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".tmp", ".log", ".aux", ".out", ".synctex.gz", ".fls", ".fdb_latexmk"}

REQUIRED_RESULTS = [
    "results/dataset/verification.json", "results/dataset/image_metadata.csv",
    "results/dataset/split_manifest.csv", "results/dataset/leakage_report.json",
    "results/phase1/metrics.json", "results/phase1/predictions.csv",
    "results/phase2/metrics.json", "results/phase2/predictions.csv",
    "results/ablation/ablation.csv", "results/drift/drift_metrics.csv",
    "reports/final_metrics.csv", "reports/localization_summary.csv", "reports/experiment_summary.md",
    "models/convae_best.pt", "models/phase1_policy.json", "models/phase2/rcc_policy.json",
]
REQUIRED_CODE = ["README.md", "RUN_INSTRUCTIONS.md", "requirements.txt", "experiment_config.yaml", "run_all.py",
                 "app/streamlit_app.py", "DATASET_SETUP.md"]


def readiness_check(root: Path = ROOT, need_results: bool = True, code_root: Path | None = None) -> list[str]:
    code_root = code_root or root
    missing = [r for r in REQUIRED_CODE if not (code_root / r).exists()]
    if need_results:
        missing += [r for r in REQUIRED_RESULTS if not (root / r).exists()]
    return missing


def _skip(p: Path, root: Path, include_dataset: bool) -> bool:
    rel = p.relative_to(root)
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return True
    if any(str(p).endswith(s) for s in EXCLUDE_SUFFIX):
        return True
    if not include_dataset and rel.parts[:2] == ("data", "raw") and p.name != ".gitkeep":
        return True
    return False


def build(root: Path = ROOT, include_dataset: bool = True) -> Path:
    out = root / "dist" / ZIP_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    data_present = (root / "data/raw/carinthia_s/data/carinthia-s.csv").exists()
    include_dataset = include_dataset and data_present
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file() and not _skip(p, root, include_dataset):
                z.write(p, f"{TOP}/{p.relative_to(root).as_posix()}")
                n += 1
        if not include_dataset:  # keep the empty dataset folder structure
            for d in ["images", "masks"]:
                z.writestr(f"{TOP}/data/raw/carinthia_s/data/{d}/.gitkeep", "")
    size = out.stat().st_size / 1e6
    print(f"ZIP created: {out}  ({n} files, {size:.1f} MB, dataset included: {include_dataset})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dataset", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--allow-missing-results", action="store_true",
                    help="package code even if experiments have not been run yet")
    a = ap.parse_args()
    missing = readiness_check(ROOT, need_results=not a.allow_missing_results)
    if missing:
        print("Readiness check - missing files:\n  " + "\n  ".join(missing))
        if not a.allow_missing_results:
            print("Run `python run_all.py` first (or pass --allow-missing-results).")
            sys.exit(1)
    elif a.allow_missing_results and readiness_check(ROOT, need_results=True):
        print("Readiness check (code only): PASS - experiment results not generated yet; run `python run_all.py`.")
    else:
        print("Readiness check: PASS")
    if not a.check:
        build(ROOT, include_dataset=not a.no_dataset)
