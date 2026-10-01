import json

import numpy as np
from PIL import Image

from src.inference import ScreeningEngine

REQUIRED = ["results/phase1/metrics.json", "results/phase2/metrics.json", "results/ablation/ablation.csv",
            "results/drift/drift_metrics.csv", "reports/final_metrics.csv", "reports/localization_summary.csv",
            "reports/experiment_summary.md", "models/phase2/rcc_policy.json", "results/experiment_manifest.json"]


def test_result_generation(pipeline):
    for r in REQUIRED:
        assert (pipeline.root / r).exists(), r
    m2 = json.loads((pipeline.results / "phase2" / "metrics.json").read_text())
    for k in ["roc_auc", "pr_auc", "f1", "balanced_accuracy", "fpr", "fnr", "cost_per_image",
              "selected_topk", "selected_lambda", "selected_alpha"]:
        assert k in m2
    assert set(["dice", "iou", "pixel_auroc"]) <= set(m2["localization"])
    assert (pipeline.figures / "roc_curve.png").exists()


def test_engine_single_and_batch(pipeline, cfg):
    eng = ScreeningEngine(cfg)
    img = Image.fromarray((np.random.default_rng(0).random((80, 80)) * 255).astype(np.uint8))
    for phase in (1, 2):
        r = eng.screen(img, phase)
        assert r["decision"] in ("REFERENCE", "ANOMALY")
        assert 0 < r["p_value"] <= 1
        assert r["pred_mask"].shape == (cfg["data"]["image_size"],) * 2
    df = eng.screen_batch([("a.png", img), ("b.png", img)], 2)
    assert len(df) == 2 and (df.status == "OK").all()
    assert eng.info()["parameters"] > 0
