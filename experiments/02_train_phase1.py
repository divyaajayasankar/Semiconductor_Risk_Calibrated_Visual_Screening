"""Step 02 - Train the Phase-1 2-D ConvAE on TRAIN reference images (skipped if a valid checkpoint exists).

Usage: python experiments/02_train_phase1.py [--retrain]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import data as D  # noqa: E402
from src.config import get_paths, load_config  # noqa: E402
from src.models_convae import checkpoint_is_valid  # noqa: E402
from src.training import train_phase1  # noqa: E402
from src.utils import log, save_json  # noqa: E402


def main(cfg=None, retrain: bool = False) -> dict:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    if not retrain and checkpoint_is_valid(paths.checkpoint, cfg["data"]["image_size"]):
        log(f"Valid Phase-1 checkpoint found ({paths.checkpoint.name}) - reusing, NOT retraining.")
        return {"status": "reused"}
    split = D.load_split(paths.split_manifest, paths.data_dir, paths.csv)
    train_refs = D.role(split, "TRAIN", 0)
    val_refs = D.role(split, "VALIDATION", 0)
    info = train_phase1(train_refs, val_refs, cfg, paths.checkpoint)
    out = paths.results / "phase1"
    pd.DataFrame(info.pop("history")).to_csv(out / "training_history.csv", index=False)
    info.update(n_train_reference=len(train_refs), n_val_reference=len(val_refs), status="trained")
    save_json(info, out / "training.json")
    log(f"Best checkpoint: epoch {info['best_epoch']}  val_loss={info['best_val_loss']:.6f} -> {paths.checkpoint}")
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true", help="force retraining even if a valid checkpoint exists")
    main(retrain=ap.parse_args().retrain)
