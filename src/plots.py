"""Publication figures (matplotlib, headless)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import curves  # noqa: E402

C1, C2, C3, C4, C5 = "#64748B", "#2563EB", "#0D9488", "#D97706", "#DC2626"
POLICY_COLORS = {"Phase1-frozen": "#94A3B8", "Phase1-recalibrated": C1,
                 "Phase2-frozen": "#93C5FD", "Phase2-recalibrated": C2}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "legend.frameon": False, "font.family": "DejaVu Sans",
})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def roc_pr(p1: pd.DataFrame, p2: pd.DataFrame, out: Path, m1: dict, m2: dict):
    c1, c2 = curves(p1.is_anomaly, p1.score), curves(p2.is_anomaly, p2.score)
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.plot(c1["fpr"], c1["tpr"], color=C1, lw=1.8, label=f"Phase 1 ConvAE (AUC={m1['roc_auc']:.4f})")
    ax.plot(c2["fpr"], c2["tpr"], color=C2, lw=1.8, ls="--", label=f"RCC-ConvAE (AUC={m2['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], color="#CBD5E1", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (TEST)")
    ax.legend(loc="lower right", fontsize=7)
    _save(fig, out / "roc_curve.png")
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.plot(c1["recall"], c1["precision"], color=C1, lw=1.8, label=f"Phase 1 (AP={m1['pr_auc']:.4f})")
    ax.plot(c2["recall"], c2["precision"], color=C2, lw=1.8, ls="--", label=f"RCC-ConvAE (AP={m2['pr_auc']:.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (TEST)")
    ax.legend(loc="lower left", fontsize=7)
    _save(fig, out / "pr_curve.png")


def confusion(m: dict, title: str, path: Path):
    cm = np.array(m["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(3.2, 2.9))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "#0F172A", fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1], ["REFERENCE", "ANOMALY"])
    ax.set_yticks([0, 1], ["REFERENCE", "ANOMALY"])
    ax.set(xlabel="Predicted", ylabel="Ground truth (mask)", title=title)
    _save(fig, path)


def score_distribution(pred: pd.DataFrame, thr: float, title: str, xlabel: str, path: Path, log: bool = False):
    fig, ax = plt.subplots(figsize=(4.4, 3))
    ref, an = pred.loc[pred.is_anomaly == 0, "score"], pred.loc[pred.is_anomaly == 1, "score"]
    lo, hi = pred.score.min(), pred.score.max()
    bins = np.geomspace(max(lo, 1e-8), hi, 50) if log and lo > 0 else np.linspace(lo, hi, 50)
    ax.hist(an, bins=bins, color=C5, alpha=0.55, density=True, label=f"Anomaly (n={len(an)})")
    ax.hist(ref, bins=bins, color=C3, alpha=0.7, density=True, label=f"Reference (n={len(ref)})")
    ax.axvline(thr, color="#0F172A", ls="--", lw=1.2, label="Decision threshold")
    if log and lo > 0:
        ax.set_xscale("log")
    ax.set(xlabel=xlabel, ylabel="Density", title=title)
    ax.legend(fontsize=7)
    _save(fig, path)


def pvalue_hist(pred: pd.DataFrame, alpha: float, path: Path):
    fig, ax = plt.subplots(figsize=(4.4, 3))
    bins = np.linspace(0, 1, 26)
    ax.hist(pred.loc[pred.is_anomaly == 1, "p_value"], bins=bins, color=C5, alpha=0.55, density=True, label="Anomaly")
    ax.hist(pred.loc[pred.is_anomaly == 0, "p_value"], bins=bins, color=C3, alpha=0.7, density=True, label="Reference")
    ax.axvline(alpha, color="#0F172A", ls="--", lw=1.2, label=f"alpha* = {alpha:g}")
    ax.set(xlabel="Conformal p-value", ylabel="Density", title="Conformal p-values (TEST)")
    ax.legend(fontsize=7)
    _save(fig, path)


def alpha_cost(table: pd.DataFrame, alpha: float, path: Path):
    fig, ax = plt.subplots(figsize=(4.4, 3))
    ax.plot(table.alpha, table.total_cost, marker="o", color=C2, lw=1.6, label="Total cost (10FN + FP)")
    ax.bar(table.alpha, table.fn_cost, width=0.006, color=C5, alpha=0.5, label="FN cost")
    ax.axvline(alpha, color="#0F172A", ls="--", lw=1.1, label=f"alpha* = {alpha:g}")
    ax.set(xlabel="alpha", ylabel="Cost (policy-validation)", title="Cost-aware alpha selection")
    ax.legend(fontsize=7)
    _save(fig, path)


def training_curve(hist: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(4.4, 3))
    ax.plot(hist.epoch, hist.train_loss, color=C1, lw=1.5, label="Train (reference)")
    ax.plot(hist.epoch, hist.val_loss, color=C2, lw=1.5, label="Validation (reference)")
    b = hist.val_loss.idxmin()
    ax.scatter(hist.epoch[b], hist.val_loss[b], color=C5, zorder=3, s=20, label=f"Best epoch {hist.epoch[b]}")
    ax.set_yscale("log")
    ax.set(xlabel="Epoch", ylabel="MSE reconstruction loss", title="Phase-1 ConvAE training")
    ax.legend(fontsize=7)
    _save(fig, path)


def localization_examples(npz_path: Path, path: Path, n: int = 4):
    z = np.load(npz_path, allow_pickle=True)
    idx = sorted({int(k.split("_")[-1]) for k in z.files if k.startswith("image_")})[:n]
    if not idx:
        return
    cols = ["Input SEM", "Reconstruction", "Error heatmap (P2)", "Ground-truth mask", "Pred. mask P1",
            "Pred. mask P2", "Overlay (P2)"]
    fig, axes = plt.subplots(len(idx), len(cols), figsize=(1.75 * len(cols), 1.8 * len(idx)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(idx):
        img, rec, mp = z[f"image_{i}"].astype(float), z[f"recon_{i}"].astype(float), z[f"map_{i}"].astype(float)
        gt, pr = z[f"gt_{i}"], z[f"pred_{i}"]
        p1 = z[f"p1pred_{i}"] if f"p1pred_{i}" in z.files else np.zeros_like(pr)
        panels = [(img, "gray"), (rec, "gray"), (mp, "inferno"), (gt, "gray"), (p1, "gray"), (pr, "gray")]
        for c, (arr, cmap) in enumerate(panels):
            axes[r, c].imshow(arr, cmap=cmap)
        axes[r, 6].imshow(img, cmap="gray")
        axes[r, 6].imshow(np.ma.masked_where(~pr.astype(bool), pr), cmap="autumn", alpha=0.55)
        axes[r, 6].contour(gt.astype(float), levels=[0.5], colors="#22D3EE", linewidths=0.7)
        axes[r, 0].set_ylabel(f"Dice {float(z[f'dice_{i}']):.2f}", fontsize=7)
    for c, t in enumerate(cols):
        axes[0, c].set_title(t, fontsize=7)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    _save(fig, path)


def ablation_bars(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.8))
    for ax, (col, t) in zip(axes, [("balanced_accuracy", "Balanced accuracy"), ("cost_per_image", "Cost / image"),
                                   ("dice", "Dice (localisation)")]):
        vals = df[col].astype(float).fillna(0)
        colors = [C2 if v == "D" else "#94A3B8" for v in df.variant]
        ax.bar(df.variant, vals, color=colors)
        for x, v in zip(df.variant, vals):
            ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(t)
    fig.suptitle("Ablation study (TEST); D = RCC-ConvAE", fontsize=9)
    _save(fig, path)


def drift_plot(df: pd.DataFrame, path: Path, metric: str = "balanced_accuracy"):
    conds = list(dict.fromkeys(df.condition))
    fig, ax = plt.subplots(figsize=(7.5, 3))
    w = 0.2
    for j, pol in enumerate(POLICY_COLORS):
        g = df[df.policy == pol].set_index("condition").reindex(conds)
        ax.bar(np.arange(len(conds)) + (j - 1.5) * w, g[metric], w, color=POLICY_COLORS[pol], label=pol)
    ax.set_xticks(np.arange(len(conds)), conds, rotation=25, ha="right")
    ax.set(ylabel=metric.replace("_", " ").title())
    ax.set_title(f"Distribution drift: {metric.replace('_', ' ')}", pad=22)
    ax.set_ylim(max(0, df[metric].min() - 0.05), 1.005)
    ax.legend(fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _save(fig, path)


def dataset_overview(meta: pd.DataFrame, split: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
    vc = meta.groupby(["label", "is_anomaly"]).size().unstack(fill_value=0)
    vc.plot.bar(ax=axes[0], color=[C3, C5], width=0.7)
    axes[0].set(title="Metadata class vs. mask ground truth", xlabel="Class", ylabel="Images")
    axes[0].legend(["Reference (empty mask)", "Anomaly (non-empty)"], fontsize=7)
    axes[0].set_yscale("log")
    rc = split.groupby(["role", "is_anomaly"]).size().unstack(fill_value=0)
    rc.plot.barh(ax=axes[1], stacked=True, color=[C3, C5])
    axes[1].set(title="Leakage-free split roles", xlabel="Images", ylabel="")
    axes[1].legend(["Reference", "Anomaly"], fontsize=7)
    _save(fig, path)
