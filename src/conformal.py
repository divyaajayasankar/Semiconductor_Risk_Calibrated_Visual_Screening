"""Split-conformal anomaly p-values from reference-only calibration scores.

p(x) = (1 + #{i : alpha_i >= alpha(x)}) / (n + 1)

Under exchangeability of a new reference image with the calibration references,
P(p <= a) <= a, i.e. the false-alarm rate on reference images is controlled at a.
"""
from __future__ import annotations

import numpy as np


def conformal_pvalues(cal_scores, scores) -> np.ndarray:
    cal = np.sort(np.asarray(cal_scores, dtype=float))
    s = np.atleast_1d(np.asarray(scores, dtype=float))
    n = len(cal)
    if n == 0:
        raise ValueError("empty calibration set")
    n_ge = n - np.searchsorted(cal, s, side="left")  # count of cal >= s
    return (1.0 + n_ge) / (n + 1.0)


def empirical_threshold(cal_scores, quantile: float) -> float:
    """Finite-sample-corrected empirical quantile of reference calibration scores."""
    cal = np.sort(np.asarray(cal_scores, dtype=float))
    n = len(cal)
    k = int(np.ceil((n + 1) * quantile)) - 1
    return float(cal[min(max(k, 0), n - 1)])


def pvalue_diagnostics(p_ref: np.ndarray, p_anom: np.ndarray, alphas) -> dict:
    """Empirical false-alarm rate vs nominal alpha (validity check) on held-out references."""
    out = {"n_reference": int(len(p_ref)), "n_anomaly": int(len(p_anom)), "per_alpha": []}
    for a in alphas:
        out["per_alpha"].append({
            "alpha": float(a),
            "reference_flag_rate": float(np.mean(p_ref <= a)) if len(p_ref) else None,
            "anomaly_detection_rate": float(np.mean(p_anom <= a)) if len(p_anom) else None,
        })
    out["median_p_reference"] = float(np.median(p_ref)) if len(p_ref) else None
    out["median_p_anomaly"] = float(np.median(p_anom)) if len(p_anom) else None
    out["min_attainable_p"] = None
    return out
