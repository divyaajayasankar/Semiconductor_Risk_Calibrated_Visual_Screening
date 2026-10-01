"""Frozen-policy inference used by the dashboard and batch screening."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .config import get_paths, load_config
from .conformal import conformal_pvalues
from .localization import predict_mask
from .models_convae import count_parameters, load_checkpoint
from .preprocessing import load_image
from .scoring import apply_robust, error_maps, image_scores, local_col
from .utils import file_sha256, load_json


class ScreeningEngine:
    """Loads the frozen ConvAE + Phase-1/Phase-2 policies and screens SEM images."""

    def __init__(self, cfg: dict | None = None, root: Path | None = None):
        self.cfg = cfg or load_config()
        self.paths = get_paths(self.cfg, root)
        self.size = self.cfg["data"]["image_size"]
        self.device = torch.device("cpu")
        self.model, self.ckpt = load_checkpoint(self.paths.checkpoint, self.device)
        self.p1 = load_json(self.paths.phase1_policy)
        self.p2 = load_json(self.paths.phase2_dir / "rcc_policy.json")
        self.p2["normalization"] = load_json(self.paths.phase2_dir / "normalization.json")
        self.p2["calibration_scores"] = np.load(self.paths.phase2_dir / "calibration_scores.npy").tolist()
        self.loc = load_json(self.paths.phase2_dir / "localization_threshold.json")

    @staticmethod
    def available(cfg: dict | None = None, root: Path | None = None) -> tuple[bool, list[str]]:
        cfg = cfg or load_config()
        p = get_paths(cfg, root)
        need = [p.checkpoint, p.phase1_policy, p.phase2_dir / "rcc_policy.json",
                p.phase2_dir / "normalization.json", p.phase2_dir / "calibration_scores.npy",
                p.phase2_dir / "localization_threshold.json"]
        missing = [str(x.relative_to(p.root)) for x in need if not x.exists()]
        return len(missing) == 0, missing

    def info(self) -> dict:
        return {
            "checkpoint": str(self.paths.checkpoint.name),
            "checkpoint_sha256": file_sha256(self.paths.checkpoint),
            "policy_sha256": file_sha256(self.paths.phase2_dir / "rcc_policy.json"),
            "parameters": count_parameters(self.model),
            "input": f"1 x {self.size} x {self.size} (grayscale, [0,1])",
            "best_epoch": self.ckpt.get("epoch"), "val_loss": self.ckpt.get("val_loss"),
            "seed": self.ckpt.get("seed", self.cfg["seed"]),
            "phase1_threshold": self.p1["threshold"], "phase1_calibration_size": self.p1["calibration_size"],
            "topk": self.p2["topk"], "lambda": self.p2["lambda"], "alpha": self.p2["alpha"],
            "cost_fn": self.p2["cost_fn"], "cost_fp": self.p2["cost_fp"],
            "calibration_size": self.p2["calibration_size"],
            "loc_phase1": self.loc["phase1"], "loc_phase2": self.loc["phase2"],
        }

    def screen(self, image: Image.Image | str | Path, phase: int = 2) -> dict:
        x = load_image(image if isinstance(image, Image.Image) else Image.open(image), self.size)
        recon, raw, sm = error_maps(self.model, x[None], self.device, self.p2["smoothing_sigma"])
        sc = image_scores(raw, sm, [self.p2["topk"]])
        g = float(sc["global"][0])
        l = float(sc[local_col(self.p2["topk"])][0])
        out = {"image": x, "recon": recon[0], "raw_map": raw[0], "smooth_map": sm[0],
               "global_score": g, "local_score": l, "phase": phase}
        if phase == 1:
            out.update(
                score=g, threshold=self.p1["threshold"],
                p_value=float(conformal_pvalues(self.p1["calibration_global_scores"], [g])[0]),
                decision="ANOMALY" if g >= self.p1["threshold"] else "REFERENCE",
                fused_score=None, rule=(f"global {'≥' if g >= self.p1['threshold'] else '<'} threshold {self.p1['threshold']:.4g}"),
                heatmap=raw[0], heat_vmax=2.0 * self.loc["phase1"]["threshold"],
                pred_mask=predict_mask(raw[0], self.loc["phase1"]["threshold"], 0),
            )
        else:
            n = self.p2["normalization"]
            zg = float(apply_robust([g], n["global"])[0])
            zl = float(apply_robust([l], n[local_col(self.p2["topk"])])[0])
            fused = zg + self.p2["lambda"] * zl
            p = float(conformal_pvalues(self.p2["calibration_scores"], [fused])[0])
            out.update(
                score=fused, fused_score=fused, z_global=zg, z_local=zl, p_value=p,
                threshold=self.p2["alpha"],
                rule=f"p = {p:.4f} {'≤' if p <= self.p2['alpha'] else '>'} α* = {self.p2['alpha']:g}",
                decision="ANOMALY" if p <= self.p2["alpha"] else "REFERENCE",
                heatmap=sm[0], heat_vmax=2.0 * self.loc["phase2"]["threshold"],
                pred_mask=predict_mask(sm[0], self.loc["phase2"]["threshold"], self.loc["phase2"]["morph_radius"]),
            )
        return out

    def screen_batch(self, items: list[tuple[str, Image.Image]], phase: int = 2) -> pd.DataFrame:
        rows = []
        for name, img in items:
            try:
                r = self.screen(img, phase)
                rows.append({"filename": name, "decision": r["decision"], "global_score": r["global_score"],
                             "local_score": r["local_score"] if phase == 2 else None,
                             "fused_score": r["fused_score"], "p_value": r["p_value"],
                             "threshold": r["threshold"], "defect_area_pct": 100 * float(r["pred_mask"].mean()),
                             "status": "OK"})
            except Exception as e:  # keep screening other files
                rows.append({"filename": name, "decision": None, "status": f"ERROR: {e}"})
        return pd.DataFrame(rows)
