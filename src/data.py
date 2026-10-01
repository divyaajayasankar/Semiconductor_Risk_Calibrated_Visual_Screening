"""Carinthia-S dataset handling: path resolution, mask-based ground truth,
verification, and deterministic leakage-free splitting.

Ground-truth rule (never inferred from the class number):
    EMPTY mask      -> REFERENCE / NORMAL (is_anomaly = 0)
    NON-EMPTY mask  -> ANOMALY / DEFECT   (is_anomaly = 1)
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .config import ROLES
from .utils import log

REQUIRED_COLUMNS = ["image_path", "mask_path", "filename", "label"]


# ---------------------------------------------------------------- paths
def resolve_path(value: str, data_dir: Path, kind: str) -> Path:
    """Resolve a CSV path robustly (CSV paths may be relative to several roots)."""
    value = str(value).replace("\\", "/").strip()
    p = Path(value)
    candidates = [
        p,
        data_dir / p,
        data_dir.parent / p,
        data_dir.parent.parent / p,
        data_dir / kind / p.name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return data_dir / kind / p.name  # best guess (reported as missing by verification)


def load_index(data_dir: Path, csv_path: Path) -> pd.DataFrame:
    """Read carinthia-s.csv and attach absolute image/mask paths."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found: {csv_path}\nSee DATASET_SETUP.md for where to place Carinthia-S."
        )
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df = df.copy()
    df["filename"] = df["filename"].astype(str)
    df["image_abs"] = [str(resolve_path(v, data_dir, "images")) for v in df["image_path"]]
    df["mask_abs"] = [str(resolve_path(v, data_dir, "masks")) for v in df["mask_path"]]
    return df


def mask_foreground(mask_path: str | Path, threshold: int) -> tuple[int, int, tuple[int, int], str]:
    """Return (#pixels>0, #pixels>=threshold, size, mode) for a mask."""
    with Image.open(mask_path) as m:
        size, mode = m.size, m.mode
        arr = np.asarray(m.convert("L"))
    return int((arr > 0).sum()), int((arr >= threshold).sum()), size, mode


# --------------------------------------------------------- verification
def verify_dataset(data_dir: Path, csv_path: Path, mask_threshold: int = 128) -> tuple[dict, pd.DataFrame]:
    """Compute every dataset statistic from the files on disk (nothing hard-coded)."""
    df = load_index(data_dir, csv_path)
    rows = []
    n_img_missing = n_mask_missing = n_misaligned = 0
    for r in df.itertuples(index=False):
        rec: dict[str, Any] = {"filename": r.filename, "label": r.label,
                               "image_path": r.image_abs, "mask_path": r.mask_abs}
        img_ok, msk_ok = Path(r.image_abs).exists(), Path(r.mask_abs).exists()
        n_img_missing += not img_ok
        n_mask_missing += not msk_ok
        if img_ok:
            with Image.open(r.image_abs) as im:
                rec.update(width=im.size[0], height=im.size[1], mode=im.mode)
        if msk_ok:
            fg_any, fg_thr, msize, mmode = mask_foreground(r.mask_abs, mask_threshold)
            rec.update(mask_width=msize[0], mask_height=msize[1], mask_mode=mmode,
                       mask_fg_pixels=fg_any, mask_fg_pixels_thr=fg_thr)
            if img_ok and (rec["width"], rec["height"]) != msize:
                n_misaligned += 1
        rows.append(rec)
    meta = pd.DataFrame(rows)
    meta["is_anomaly"] = (meta.get("mask_fg_pixels", pd.Series(0, index=meta.index)).fillna(0) > 0).astype(int)
    meta["mask_coverage"] = meta.get("mask_fg_pixels", 0) / (meta.get("width", 1) * meta.get("height", 1))

    sizes = meta.groupby(["width", "height"]).size() if "width" in meta else pd.Series(dtype=int)
    anom = meta[meta.is_anomaly == 1]
    report = {
        "csv_path": str(csv_path),
        "csv_exists": True,
        "csv_rows": int(len(df)),
        "csv_columns": list(df.columns),
        "missing_values": {c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS},
        "duplicate_filenames": int(df["filename"].duplicated().sum()),
        "images_found": int(len(df) - n_img_missing),
        "masks_found": int(len(df) - n_mask_missing),
        "images_missing": int(n_img_missing),
        "masks_missing": int(n_mask_missing),
        "image_mask_size_mismatch": int(n_misaligned),
        "image_sizes": {f"{w}x{h}": int(c) for (w, h), c in sizes.items()},
        "image_modes": meta["mode"].value_counts().to_dict() if "mode" in meta else {},
        "mask_modes": meta["mask_mode"].value_counts().to_dict() if "mask_mode" in meta else {},
        "shape_consistent": bool(len(sizes) == 1),
        "empty_masks_reference": int((meta.is_anomaly == 0).sum()),
        "non_empty_masks_anomaly": int((meta.is_anomaly == 1).sum()),
        "class_distribution": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "class_vs_ground_truth": {
            str(k): {"reference": int((g.is_anomaly == 0).sum()), "anomaly": int((g.is_anomaly == 1).sum())}
            for k, g in meta.groupby("label")
        },
        "mask_coverage_anomalies": {
            "mean": float(anom["mask_coverage"].mean()) if len(anom) else 0.0,
            "median": float(anom["mask_coverage"].median()) if len(anom) else 0.0,
            "min": float(anom["mask_coverage"].min()) if len(anom) else 0.0,
            "max": float(anom["mask_coverage"].max()) if len(anom) else 0.0,
        },
        "ground_truth_rule": "EMPTY mask -> REFERENCE; NON-EMPTY mask -> ANOMALY",
        "mask_threshold_for_pixels": mask_threshold,
    }
    report["status"] = "PASS" if (
        report["images_missing"] == 0 and report["masks_missing"] == 0
        and report["duplicate_filenames"] == 0 and report["image_mask_size_mismatch"] == 0
        and report["empty_masks_reference"] > 0 and report["non_empty_masks_anomaly"] > 0
    ) else "FAIL"
    return report, meta


# --------------------------------------------------------------- split
def _allocate(ids: list, fractions: dict[str, float], rng: np.random.Generator) -> dict[str, str]:
    ids = list(ids)
    rng.shuffle(ids)
    names = list(fractions)
    fr = np.array([fractions[k] for k in names], dtype=float)
    fr = fr / fr.sum()
    counts = np.floor(fr * len(ids)).astype(int)
    for i in np.argsort(-(fr * len(ids) - counts))[: len(ids) - counts.sum()]:
        counts[i] += 1
    out, start = {}, 0
    for name, c in zip(names, counts):
        for x in ids[start:start + c]:
            out[x] = name
        start += c
    return out


def make_split(meta: pd.DataFrame, split_cfg: dict, seed: int = 42) -> pd.DataFrame:
    """Deterministic split. Reference images over 5 roles; anomalies only over
    VALIDATION / POLICY_VALIDATION / TEST, stratified by metadata class."""
    rng = np.random.default_rng(seed)
    role_map: dict[str, str] = {}
    ref_ids = sorted(meta.loc[meta.is_anomaly == 0, "filename"])
    role_map.update(_allocate(ref_ids, {k.upper(): v for k, v in split_cfg["reference"].items()}, rng))
    anom = meta[meta.is_anomaly == 1]
    for _, g in sorted(anom.groupby("label"), key=lambda t: str(t[0])):
        role_map.update(_allocate(sorted(g["filename"]),
                                  {k.upper(): v for k, v in split_cfg["anomaly"].items()}, rng))
    out = meta[["filename", "label", "is_anomaly", "image_path", "mask_path"]].copy()
    out["role"] = out["filename"].map(role_map)
    return out.sort_values(["role", "filename"]).reset_index(drop=True)


def validate_split(split: pd.DataFrame, meta: pd.DataFrame | None = None) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if split["filename"].duplicated().any():
        problems.append("duplicate filenames in split manifest")
    bad_roles = set(split["role"].dropna()) - set(ROLES)
    if bad_roles or split["role"].isna().any():
        problems.append(f"invalid/missing roles: {bad_roles}")
    if (split.loc[split.role == "TRAIN", "is_anomaly"] == 1).any():
        problems.append("TRAIN contains anomalous images")
    if (split.loc[split.role == "CALIBRATION", "is_anomaly"] == 1).any():
        problems.append("CALIBRATION contains anomalous images")
    for r in ROLES:
        if (split.role == r).sum() == 0:
            problems.append(f"role {r} is empty")
    if meta is not None:
        if set(meta["filename"]) != set(split["filename"]):
            problems.append("split manifest does not match dataset files")
        else:
            m = meta.set_index("filename")["is_anomaly"]
            if (split.set_index("filename")["is_anomaly"] != m.reindex(split["filename"]).values).any():
                problems.append("ground-truth labels in manifest disagree with masks")
    return len(problems) == 0, problems


def leakage_report(split: pd.DataFrame) -> dict:
    sets = {r: set(split.loc[split.role == r, "filename"]) for r in ROLES}
    test = sets["TEST"]
    inter = {f"{r.lower()}_intersect_test": len(sets[r] & test) for r in ROLES if r != "TEST"}
    counts = {r: {"total": len(sets[r]),
                  "reference": int(((split.role == r) & (split.is_anomaly == 0)).sum()),
                  "anomaly": int(((split.role == r) & (split.is_anomaly == 1)).sum())} for r in ROLES}
    ok = all(v == 0 for v in inter.values()) and counts["TRAIN"]["anomaly"] == 0 \
        and counts["CALIBRATION"]["anomaly"] == 0
    return {
        **inter,
        "train_contains_only_reference": counts["TRAIN"]["anomaly"] == 0,
        "calibration_contains_only_reference": counts["CALIBRATION"]["anomaly"] == 0,
        "role_counts": counts,
        "test_used_for": ["final evaluation only (after policy freeze)"],
        "status": "PASS" if ok else "FAIL",
    }


def manifest_hash(split: pd.DataFrame) -> str:
    s = "\n".join(f"{f},{r}" for f, r in sorted(zip(split["filename"], split["role"])))
    return hashlib.sha256(s.encode()).hexdigest()[:16]


MANIFEST_COLUMNS = ["filename", "label", "is_anomaly", "role"]


def save_split(split: pd.DataFrame, path: Path) -> None:
    """Persist the manifest WITHOUT machine-specific absolute paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    split[MANIFEST_COLUMNS].to_csv(path, index=False)


def load_split(path: Path, data_dir: Path | None = None, csv_path: Path | None = None) -> pd.DataFrame:
    """Load the manifest; if data_dir/csv given, attach absolute image/mask paths."""
    df = pd.read_csv(path)
    df["filename"] = df["filename"].astype(str)
    if data_dir is not None and csv_path is not None:
        idx = load_index(data_dir, csv_path).set_index("filename")
        df["image_path"] = df["filename"].map(idx["image_abs"])
        df["mask_path"] = df["filename"].map(idx["mask_abs"])
    return df


def role(split: pd.DataFrame, name: str, anomaly: int | None = None) -> pd.DataFrame:
    d = split[split.role == name]
    if anomaly is not None:
        d = d[d.is_anomaly == anomaly]
    return d.reset_index(drop=True)


def get_or_create_split(meta: pd.DataFrame, path: Path, split_cfg: dict, seed: int) -> tuple[pd.DataFrame, str]:
    """Preserve an existing valid split manifest; otherwise create a new one."""
    if path.exists():
        existing = load_split(path)
        ok, problems = validate_split(existing, meta)
        if ok:
            # refresh absolute paths (manifest may come from another machine)
            p = meta.set_index("filename")[["image_path", "mask_path"]]
            existing["image_path"] = existing["filename"].map(p["image_path"])
            existing["mask_path"] = existing["filename"].map(p["mask_path"])
            log(f"Existing split manifest is valid - preserved ({path.name}).")
            return existing, "preserved"
        log(f"Existing split manifest invalid ({problems}) - regenerating.")
    return make_split(meta, split_cfg, seed), "created"
