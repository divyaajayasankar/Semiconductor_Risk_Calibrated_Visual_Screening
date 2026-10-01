"""Step 09 - Figures, report tables, experiment summary and reproducibility manifest.

Everything is read from the result files produced by steps 01-08 (nothing is hard-coded).
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import data as D  # noqa: E402
from src import plots  # noqa: E402
from src.config import RESEARCH_TITLE, get_paths, load_config  # noqa: E402
from src.utils import file_sha256, get_device, load_json, log, python_version, save_json  # noqa: E402

IMG_METRICS = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1",
               "fpr", "fnr", "tn", "fp", "fn", "tp", "cost_per_image", "total_cost", "fn_cost", "fp_cost"]
LOC_METRICS = ["dice", "iou", "pixel_auroc", "pixel_ap"]


def _fmt(v, nd=4):
    return "n/a" if v is None or (isinstance(v, float) and pd.isna(v)) else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main(cfg=None) -> None:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    R, F = paths.results, paths.figures
    m1 = load_json(R / "phase1" / "metrics.json")
    m2 = load_json(R / "phase2" / "metrics.json")
    p1 = pd.read_csv(R / "phase1" / "predictions.csv")
    p2 = pd.read_csv(R / "phase2" / "predictions.csv")
    pol1 = load_json(paths.phase1_policy)
    pol2 = load_json(paths.phase2_dir / "rcc_policy.json")

    # ---------------- figures
    log("Generating figures ...")
    plots.roc_pr(p1, p2, F, m1, m2)
    plots.confusion(m1, "Phase 1 ConvAE (TEST)", F / "confusion_phase1.png")
    plots.confusion(m2, "Phase 2 RCC-ConvAE (TEST)", F / "confusion_phase2.png")
    plots.score_distribution(p1, pol1["threshold"], "Phase-1 global score", "Global reconstruction MSE",
                             F / "score_distribution_phase1.png", log=True)
    thr_fused = p2.loc[p2.pred == 1, "score"].min() if (p2.pred == 1).any() else p2.score.max()
    plots.score_distribution(p2, thr_fused, "Phase-2 fused score", "Fused robust score (Z_g + lambda Z_l)",
                             F / "score_distribution_phase2.png")
    plots.pvalue_hist(p2, pol2["alpha"], F / "pvalue_distribution.png")
    at = R / "phase2" / "selection_alpha_policy_validation.csv"
    if at.exists():
        plots.alpha_cost(pd.read_csv(at), pol2["alpha"], F / "alpha_cost_curve.png")
    th = R / "phase1" / "training_history.csv"
    if th.exists():
        plots.training_curve(pd.read_csv(th), F / "training_curve.png")
    ex = R / "phase2" / "localization_examples.npz"
    if ex.exists():
        plots.localization_examples(ex, F / "localization_examples.png")
    ab = R / "ablation" / "ablation.csv"
    if ab.exists():
        plots.ablation_bars(pd.read_csv(ab), F / "ablation.png")
    dr = R / "drift" / "drift_metrics.csv"
    if dr.exists():
        ddf = pd.read_csv(dr)
        plots.drift_plot(ddf, F / "drift_balanced_accuracy.png", "balanced_accuracy")
        plots.drift_plot(ddf, F / "drift_roc_auc.png", "roc_auc")
    meta = pd.read_csv(R / "dataset" / "image_metadata.csv")
    split = D.load_split(paths.split_manifest)
    plots.dataset_overview(meta, split, F / "dataset_overview.png")

    # ---------------- tables
    rows = []
    for name, m in [("Phase 1 - 2-D ConvAE (global error)", m1), ("Phase 2 - RCC-ConvAE (proposed)", m2)]:
        r = {"model": name, **{k: m.get(k) for k in IMG_METRICS}}
        for k in LOC_METRICS:
            r[k] = (m.get("localization") or {}).get(k)
        rows.append(r)
    final = pd.DataFrame(rows)
    final.to_csv(paths.reports / "final_metrics.csv", index=False)
    loc = pd.DataFrame([{"model": r["model"], **{k: r[k] for k in LOC_METRICS},
                         "threshold": (m.get("localization") or {}).get("threshold"),
                         "morph_radius": (m.get("localization") or {}).get("morph_radius"),
                         "n_anomalous_images": (m.get("localization") or {}).get("n_anomalous_images")}
                        for r, m in zip(rows, [m1, m2])])
    loc.to_csv(paths.reports / "localization_summary.csv", index=False)

    # ---------------- manifest
    pkgs = {}
    for p in ["torch", "numpy", "pandas", "scikit-learn", "scipy", "Pillow", "matplotlib", "streamlit", "plotly",
              "PyYAML"]:
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pkgs[p] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.root, capture_output=True, text=True,
                                timeout=5).stdout.strip() or None
    except Exception:
        commit = None
    ver = load_json(R / "dataset" / "verification.json")
    manifest = {
        "title": RESEARCH_TITLE, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": python_version(), "platform": platform.platform(), "packages": pkgs,
        "device": str(get_device(cfg["training"].get("device", "auto"))), "seed": cfg["seed"],
        "dataset_counts": {k: ver.get(k) for k in ["csv_rows", "images_found", "masks_found",
                                                    "empty_masks_reference", "non_empty_masks_anomaly"]},
        "split_manifest_sha256": file_sha256(paths.split_manifest),
        "checkpoint_sha256": file_sha256(paths.checkpoint),
        "rcc_policy_sha256": file_sha256(paths.phase2_dir / "rcc_policy.json"),
        "hyperparameters": {"data": cfg["data"], "model": cfg["model"], "training": cfg["training"],
                            "phase1": cfg["phase1"], "phase2": cfg["phase2"], "localization": cfg["localization"]},
        "selected": {"topk": pol2["topk"], "lambda": pol2["lambda"], "alpha": pol2["alpha"],
                     "phase1_threshold": pol1["threshold"]},
        "metric_files": [str(p.relative_to(paths.root)) for p in sorted(R.rglob("*.json")) + sorted(R.rglob("*.csv"))
                         if "cache" not in p.parts],
        "git_commit": commit,
    }
    save_json(manifest, R / "experiment_manifest.json")

    # ---------------- summary markdown
    lines = [f"# Experiment Summary\n\n**{RESEARCH_TITLE}**\n", f"Generated: {manifest['timestamp']}  ",
             f"Seed: {cfg['seed']} | Device: {manifest['device']} | Python {manifest['python']}\n",
             "## Dataset (Carinthia-S)\n",
             f"- Images: {ver['images_found']} | Masks: {ver['masks_found']} | CSV rows: {ver['csv_rows']}",
             f"- Reference (empty mask): {ver['empty_masks_reference']} | Anomaly (non-empty): {ver['non_empty_masks_anomaly']}",
             f"- Image sizes: {ver['image_sizes']} | modes: {ver['image_modes']}",
             f"- Split roles: " + ", ".join(f"{k}={v['total']} ({v['reference']} ref / {v['anomaly']} anom)"
                                           for k, v in ver['split']['role_counts'].items()) + "\n",
             "## Frozen policies\n",
             f"- Phase 1 threshold: {pol1['threshold']:.6g} ({pol1['threshold_source']})",
             f"- Phase 2: top-k = {pol2['topk']}, lambda = {pol2['lambda']}, alpha* = {pol2['alpha']}, "
             f"calibration n = {pol2['calibration_size']}, costs FN:FP = {pol2['cost_fn']}:{pol2['cost_fp']}\n",
             "## Image-level results (TEST)\n",
             "| Metric | Phase 1 ConvAE | Phase 2 RCC-ConvAE |", "|---|---|---|"]
    for k in ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1",
              "fpr", "fnr", "tn", "fp", "fn", "tp", "cost_per_image"]:
        lines.append(f"| {k} | {_fmt(m1.get(k))} | {_fmt(m2.get(k))} |")
    lines += ["\n## Pixel-level localisation (TEST)\n", "| Metric | Phase 1 | Phase 2 |", "|---|---|---|"]
    for k in LOC_METRICS:
        lines.append(f"| {k} | {_fmt((m1.get('localization') or {}).get(k))} | "
                     f"{_fmt((m2.get('localization') or {}).get(k))} |")
    if ab.exists():
        a = pd.read_csv(ab)
        lines += ["\n## Ablation (TEST)\n", a.drop(columns=["decision_rule"]).round(4).to_markdown(index=False)
                  if _has_tabulate() else a.round(4).to_string(index=False)]
    ds = R / "drift" / "drift_summary.csv"
    if ds.exists():
        d = pd.read_csv(ds)
        lines += ["\n## Distribution drift summary\n",
                  d.round(4).to_markdown(index=False) if _has_tabulate() else d.round(4).to_string(index=False)]
    lines += ["\n## Interpretation notes\n",
              "- The base paper (Gorman et al., IEEE TSM 2023) reports AUC = 1.00 on 1-D batch-process data; it is "
              "methodological inspiration, not a directly comparable benchmark.",
              "- RCC-ConvAE is a risk-calibrated, cost-aware decision framework on the same frozen ConvAE backbone "
              "(not a new CNN). Image-level parity with Phase 1 is reported as-is.",
              "- TEST data was used only after every policy component had been frozen."]
    (paths.reports / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")

    # ---------------- paper tables
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import export_paper_tables
        export_paper_tables.main(paths.root)
    except Exception as e:  # pragma: no cover
        log(f"Paper table export skipped: {e}")
    log(f"Reports written to {paths.reports} ; figures to {F}")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
