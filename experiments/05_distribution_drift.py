"""Step 08 - Distribution-drift evaluation (frozen vs recalibrated policies).

For every imaging perturbation the CALIBRATION references and TEST images are perturbed.
  P1-frozen : frozen Phase-1 threshold
  P1-recal  : threshold re-estimated from perturbed CALIBRATION references (no labels, no TEST)
  P2-frozen : frozen RCC normalisation + calibration scores + alpha*
  P2-recal  : conformal calibration scores re-computed on perturbed CALIBRATION references;
              normalisation, top-k, lambda and alpha* stay frozen
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import data as D  # noqa: E402
from src import evaluation as E  # noqa: E402
from src.conformal import empirical_threshold  # noqa: E402
from src.config import get_paths, load_config  # noqa: E402
from src.drift import make_perturb  # noqa: E402
from src.scoring import compute_scores  # noqa: E402
from src.utils import load_json, log  # noqa: E402

METRICS = ["roc_auc", "pr_auc", "balanced_accuracy", "f1", "fpr", "fnr", "cost_per_image"]


def _p2():
    spec = importlib.util.spec_from_file_location("p2", Path(__file__).with_name("04_phase2_rcc.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _subsample(test: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if not n or n >= len(test):
        return test
    ref = test[test.is_anomaly == 0]
    anom = test[test.is_anomaly == 1]
    n_anom = max(1, n - len(ref))
    return pd.concat([ref, anom.sample(n=min(n_anom, len(anom)), random_state=seed)]).reset_index(drop=True)


def main(cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    ctx = E.load_context(cfg, paths)
    split = ctx["split"]
    p1 = load_json(paths.phase1_policy)
    p2 = _p2().load_policy(paths)
    cal_recs = D.role(split, "CALIBRATION", 0)
    test_recs = _subsample(D.role(split, "TEST"), cfg["drift"].get("max_test_images"), cfg["seed"])
    q = cfg["phase1"]["threshold_quantile"]
    rows = []
    for cond in cfg["drift"]["conditions"]:
        log(f"Drift condition: {cond['name']}")
        cal = compute_scores(ctx["model"], cal_recs, cfg, ctx["device"], perturb=make_perturb(cond, cfg["seed"], 0))
        tst = compute_scores(ctx["model"], test_recs, cfg, ctx["device"],
                             perturb=make_perturb(cond, cfg["seed"], 1_000_000))
        # Phase 1
        pr = E.apply_phase1(tst, p1)
        rows.append({"condition": cond["name"], "policy": "Phase1-frozen", **_m(pr, cfg)})
        p1r = dict(p1, threshold=empirical_threshold(cal["global"], q),
                   calibration_global_scores=cal["global"].tolist())
        rows.append({"condition": cond["name"], "policy": "Phase1-recalibrated", **_m(E.apply_phase1(tst, p1r), cfg)})
        # Phase 2
        rows.append({"condition": cond["name"], "policy": "Phase2-frozen", **_m(E.apply_phase2(tst, p2), cfg)})
        cal2 = E.phase2_score(cal, p2)
        rows.append({"condition": cond["name"], "policy": "Phase2-recalibrated",
                     **_m(E.apply_phase2(tst, p2, cal_scores=cal2), cfg)})
        last = pd.DataFrame(rows[-4:])
        log(last[["policy", "roc_auc", "balanced_accuracy", "fpr", "fnr"]].round(4).to_string(index=False))

    df = pd.DataFrame(rows)
    out = paths.results / "drift"
    df.to_csv(out / "drift_metrics.csv", index=False)
    summ = summarize(df)
    summ.to_csv(out / "drift_summary.csv", index=False)
    summ.to_csv(paths.reports / "drift_summary.csv", index=False)
    log("Drift summary:\n" + summ.round(4).to_string(index=False))
    return df


def _m(pred, cfg) -> dict:
    m = E.evaluate_predictions(pred, cfg)
    return {k: m[k] for k in METRICS} | {"fp": m["fp"], "fn": m["fn"], "n": m["n"]}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pol, g in df.groupby("policy", sort=False):
        clean = g[g.condition == "clean"].iloc[0]
        shifted = g[g.condition != "clean"]
        rows.append({
            "policy": pol,
            "clean_roc_auc": clean.roc_auc, "clean_balanced_accuracy": clean.balanced_accuracy,
            "mean_roc_auc": shifted.roc_auc.mean(),
            "mean_balanced_accuracy": shifted.balanced_accuracy.mean(),
            "mean_fpr": shifted.fpr.mean(), "mean_fnr": shifted.fnr.mean(),
            "mean_cost_per_image": shifted.cost_per_image.mean(),
            "worst_balanced_accuracy": shifted.balanced_accuracy.min(),
            "worst_condition": shifted.loc[shifted.balanced_accuracy.idxmin(), "condition"],
            "mean_delta_roc_auc_vs_clean": (shifted.roc_auc - clean.roc_auc).mean(),
            "mean_delta_balanced_accuracy_vs_clean": (shifted.balanced_accuracy - clean.balanced_accuracy).mean(),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
