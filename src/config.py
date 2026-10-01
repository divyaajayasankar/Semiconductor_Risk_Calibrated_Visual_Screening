"""Central configuration loading and canonical project paths."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "experiment_config.yaml"

RESEARCH_TITLE = (
    "Risk-Calibrated and Cost-Aware Anomaly Detection Visual Screening for "
    "Semiconductor Manufacturing Under Distribution Drift"
)
APP_NAME = "SemiVision AI – Semiconductor Defect Intelligence Platform"

ROLES = ["TRAIN", "VALIDATION", "CALIBRATION", "POLICY_VALIDATION", "TEST"]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config. The env var SEMIVISION_CONFIG overrides the default path."""
    path = Path(path or os.environ.get("SEMIVISION_CONFIG", DEFAULT_CONFIG))
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path)
    return cfg


@dataclass(frozen=True)
class Paths:
    root: Path
    data_dir: Path
    csv: Path
    models: Path
    results: Path
    reports: Path

    @property
    def checkpoint(self) -> Path:
        return self.models / "convae_best.pt"

    @property
    def phase1_policy(self) -> Path:
        return self.models / "phase1_policy.json"

    @property
    def phase2_dir(self) -> Path:
        return self.models / "phase2"

    @property
    def split_manifest(self) -> Path:
        return self.results / "dataset" / "split_manifest.csv"

    @property
    def figures(self) -> Path:
        return self.results / "figures"

    @property
    def cache(self) -> Path:
        return self.results / "cache"

    def ensure(self) -> "Paths":
        for p in [self.models, self.phase2_dir, self.results, self.reports, self.figures, self.cache,
                  self.results / "dataset", self.results / "phase1", self.results / "phase2",
                  self.results / "ablation", self.results / "drift"]:
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(cfg: dict[str, Any], root: Path | None = None) -> Paths:
    root = Path(root or os.environ.get("SEMIVISION_ROOT", PROJECT_ROOT))
    p = cfg["paths"]

    def _abs(x: str) -> Path:
        q = Path(x)
        return q if q.is_absolute() else root / q

    data_dir = _abs(p["data_dir"])
    return Paths(
        root=root,
        data_dir=data_dir,
        csv=data_dir / p["csv_name"],
        models=_abs(p["models_dir"]),
        results=_abs(p["results_dir"]),
        reports=_abs(p["reports_dir"]),
    ).ensure()


def deep_copy(cfg: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(cfg)
