"""Master runner for the complete research pipeline.

    python run_all.py                 # steps 01-10
    python run_all.py --from 04       # resume (valid Phase-1 checkpoint is never retrained)
    python run_all.py --only 08       # single step
    python run_all.py --retrain       # force Phase-1 retraining
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import RESEARCH_TITLE, load_config  # noqa: E402
from src.seed import set_seed  # noqa: E402
from src.utils import log  # noqa: E402


def _mod(rel: str):
    p = ROOT / rel
    spec = importlib.util.spec_from_file_location(p.stem, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def step_readiness(cfg):
    pk = _mod("scripts/package_project.py")
    from src.config import get_paths
    missing = pk.readiness_check(get_paths(cfg).root, True, code_root=ROOT)
    if missing:
        raise RuntimeError(f"Packaging readiness FAILED - missing: {missing}")
    log("Packaging readiness: PASS (all required result/model files present)")


STEPS = [
    ("01", "Dataset verification + leakage-free split", lambda c, a: _mod("experiments/01_verify_dataset.py").main(c)),
    ("02", "Phase-1 training (if required)", lambda c, a: _mod("experiments/02_train_phase1.py").main(c, a.retrain)),
    ("03", "Phase-1 calibration + TEST evaluation", lambda c, a: _mod("experiments/03_evaluate_phase1.py").evaluate(c)),
    ("04", "Freeze Phase-1 model + policy", lambda c, a: _mod("experiments/03_evaluate_phase1.py").freeze(c)),
    ("05", "Phase-2 RCC-ConvAE policy fitting (no TEST)", lambda c, a: _mod("experiments/04_phase2_rcc.py").fit(c)),
    ("06", "Phase-2 final TEST evaluation + localisation", lambda c, a: _mod("experiments/04_phase2_rcc.py").evaluate(c)),
    ("07", "Ablation study", lambda c, a: _mod("experiments/06_ablation.py").main(c)),
    ("08", "Distribution drift + recalibration", lambda c, a: _mod("experiments/05_distribution_drift.py").main(c)),
    ("09", "Reports and figures", lambda c, a: _mod("experiments/07_generate_reports.py").main(c)),
    ("10", "Packaging-readiness validation", lambda c, a: step_readiness(c)),
]


def main():
    ap = argparse.ArgumentParser(description=RESEARCH_TITLE)
    ap.add_argument("--from", dest="start", default="01", help="first step to run (01-10)")
    ap.add_argument("--to", dest="end", default="10", help="last step to run (01-10)")
    ap.add_argument("--only", default=None, help="run a single step")
    ap.add_argument("--retrain", action="store_true", help="force Phase-1 retraining")
    ap.add_argument("--config", default=None, help="alternative YAML config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    start, end = (a.only, a.only) if a.only else (a.start.zfill(2), a.end.zfill(2))
    print("=" * 78 + f"\n{RESEARCH_TITLE}\n" + "=" * 78)
    t0 = time.time()
    for sid, name, fn in STEPS:
        if not (start <= sid <= end):
            continue
        log(f"STEP {sid} | {name}")
        ts = time.time()
        fn(cfg, a)
        log(f"STEP {sid} done in {time.time() - ts:.1f}s")
    log(f"Pipeline finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
