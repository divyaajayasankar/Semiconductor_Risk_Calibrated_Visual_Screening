"""Image-level evaluation metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve


def _safe(fn, *a):
    try:
        return float(fn(*a))
    except ValueError:
        return None


def image_metrics(y_true, score, y_pred, c_fn: float = 10.0, c_fp: float = 1.0) -> dict:
    """ROC-AUC/PR-AUC use the continuous anomaly score; the rest use the frozen decision."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    s = np.asarray(score, dtype=float)
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    div = lambda a, b: float(a / b) if b else 0.0
    rec = div(tp, tp + fn)
    spec = div(tn, tn + fp)
    prec = div(tp, tp + fp)
    return {
        "roc_auc": _safe(roc_auc_score, y, s),
        "pr_auc": _safe(average_precision_score, y, s),
        "accuracy": div(tp + tn, len(y)),
        "balanced_accuracy": (rec + spec) / 2,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": div(2 * prec * rec, prec + rec),
        "fpr": div(fp, fp + tn),
        "fnr": div(fn, fn + tp),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "n": int(len(y)), "n_reference": int((y == 0).sum()), "n_anomaly": int((y == 1).sum()),
        "total_cost": float(c_fn * fn + c_fp * fp),
        "cost_per_image": div(c_fn * fn + c_fp * fp, len(y)),
        "fn_cost": float(c_fn * fn), "fp_cost": float(c_fp * fp),
    }


def curves(y_true, score) -> dict:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    fpr, tpr, _ = roc_curve(y, s)
    pr, rc, _ = precision_recall_curve(y, s)
    return {"fpr": fpr, "tpr": tpr, "precision": pr, "recall": rc}
