import json

import pandas as pd
import pytest

from src import data as D
from src import evaluation as E
from src.config import ROLES


def test_leakage_report_pass(pipeline):
    rep = json.loads((pipeline.results / "dataset" / "leakage_report.json").read_text())
    assert rep["status"] == "PASS"
    for r in ROLES[:-1]:
        assert rep[f"{r.lower()}_intersect_test"] == 0
    assert rep["train_contains_only_reference"] and rep["calibration_contains_only_reference"]


def test_policy_fitting_never_uses_test(pipeline, cfg):
    """Changing every TEST score must not change any fitted policy component."""
    scores = pd.read_csv(pipeline.cache / "clean_scores.csv")
    scores["filename"] = scores["filename"].astype(str)
    pol_a, _ = E.fit_phase2_policy(scores[scores.role != "TEST"], cfg)
    tampered = scores.copy()
    num = [c for c in tampered.columns if c == "global" or c.startswith("local_")]
    tampered.loc[tampered.role == "TEST", num] *= 1000.0
    pol_b, _ = E.fit_phase2_policy(tampered[tampered.role != "TEST"], cfg)
    pol_c, _ = E.fit_phase2_policy(tampered, cfg)   # even if TEST rows are passed in
    for k in ["topk", "lambda", "alpha", "calibration_scores"]:
        assert pol_a[k] == pol_b[k] == pol_c[k]
    assert E.fit_phase1_policy(scores, cfg)["threshold"] == E.fit_phase1_policy(tampered, cfg)["threshold"]


def test_normalization_rejects_test(cfg):
    from src.scoring import fit_normalization
    bad = pd.DataFrame({"global": [1.0], "local_0.001": [1.0], "is_anomaly": [0], "role": ["TEST"]})
    with pytest.raises(AssertionError):
        fit_normalization(bad, [0.001], "mad")


def test_manifest_has_no_absolute_paths(pipeline):
    df = pd.read_csv(pipeline.split_manifest)
    assert list(df.columns) == D.MANIFEST_COLUMNS
