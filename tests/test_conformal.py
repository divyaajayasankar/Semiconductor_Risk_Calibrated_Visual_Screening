import numpy as np
import pandas as pd

from src.conformal import conformal_pvalues, empirical_threshold
from src.cost_policy import select_alpha, total_cost
from src.localization import StreamingPixelAUC, dice_iou, predict_mask
from src.scoring import apply_robust, fit_robust, topk_mean


def test_pvalue_formula():
    cal = np.array([1, 2, 3, 4, 5], float)
    p = conformal_pvalues(cal, [0.5, 3, 10])
    assert np.allclose(p, [6 / 6, 4 / 6, 1 / 6])   # (1 + #cal>=s)/(n+1)


def test_pvalue_validity_under_exchangeability():
    rng = np.random.default_rng(0)
    cal, new = rng.normal(size=500), rng.normal(size=20000)
    p = conformal_pvalues(cal, new)
    for a in [0.05, 0.1, 0.2]:
        assert np.mean(p <= a) <= a + 0.01


def test_threshold_quantile():
    assert empirical_threshold(np.arange(1, 101), 0.95) == 96


def test_cost_policy():
    c = total_cost([1, 1, 0, 0], [0, 1, 1, 0], 10, 1)
    assert c["FN"] == 1 and c["FP"] == 1 and c["total_cost"] == 11
    p = np.array([0.01, 0.02, 0.5, 0.9])
    y = np.array([1, 1, 0, 0])
    a, t = select_alpha(p, y, [0.005, 0.03, 0.6], 10, 1)
    assert a == 0.03 and t.total_cost.min() == 0


def test_robust_and_topk():
    prm = fit_robust(np.array([1, 2, 3, 4, 100.0]))
    assert prm["median"] == 3
    assert abs(apply_robust([3], prm)[0]) < 1e-12
    flat = np.arange(100, dtype=float)[None]
    assert topk_mean(flat, 0.05)[0] == np.mean(np.arange(95, 100))


def test_localization_shapes_and_metrics():
    m = np.zeros((32, 32)); m[10:20, 10:20] = 1
    pred = predict_mask(m, 0.5, 1)
    assert pred.shape == (32, 32)
    d, i = dice_iou(pred, m > 0.5)
    assert d > 0.9 and i > 0.8
    auc = StreamingPixelAUC(1.0, 100)
    auc.update(m, m > 0.5)
    assert auc.auroc() > 0.99 and auc.average_precision() > 0.99
