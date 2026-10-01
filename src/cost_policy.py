"""Cost-aware selection of the conformal significance level alpha*.

Total Cost = C_FN * FN + C_FP * FP   (defaults C_FN = 10, C_FP = 1)
alpha* = argmin over the alpha grid of POLICY-VALIDATION cost
Ties -> the smallest alpha (fewest false alarms).
Optional yield constraint: policy-validation flag rate <= max_flag_rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def total_cost(y_true, y_pred, c_fn: float = 10.0, c_fp: float = 1.0) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    cost = c_fn * fn + c_fp * fp
    return {"FN": fn, "FP": fp, "fn_cost": c_fn * fn, "fp_cost": c_fp * fp,
            "total_cost": float(cost), "cost_per_image": float(cost / max(len(y_true), 1))}


def select_alpha(p_values, y_true, alpha_grid, c_fn: float = 10.0, c_fp: float = 1.0,
                 max_flag_rate: float | None = None) -> tuple[float, pd.DataFrame]:
    p = np.asarray(p_values, dtype=float)
    y = np.asarray(y_true).astype(int)
    rows = []
    for a in alpha_grid:
        pred = (p <= a).astype(int)
        c = total_cost(y, pred, c_fn, c_fp)
        flag_rate = float(pred.mean())
        feasible = max_flag_rate is None or flag_rate <= max_flag_rate
        rows.append({"alpha": float(a), **c, "flag_rate": flag_rate, "feasible": feasible})
    table = pd.DataFrame(rows)
    feas = table[table.feasible] if table.feasible.any() else table
    best = feas.sort_values(["total_cost", "alpha"], ascending=[True, True]).iloc[0]
    return float(best["alpha"]), table
