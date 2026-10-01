"""Helpers for the SemiVision AI Streamlit dashboard: result loading, styling, figures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import APP_NAME, RESEARCH_TITLE, get_paths, load_config  # noqa: E402

BLUE, SLATE, TEAL, AMBER, RED = "#2563EB", "#64748B", "#0D9488", "#D97706", "#DC2626"
POLICY_COLORS = {"Phase1-frozen": "#94A3B8", "Phase1-recalibrated": "#475569",
                 "Phase2-frozen": "#93C5FD", "Phase2-recalibrated": "#2563EB"}

CSS = """
<style>
:root { --ink:#0F172A; --muted:#64748B; --line:#E2E8F0; --card:#FFFFFF; --blue:#2563EB; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0B1220 0%,#111C33 100%); }
section[data-testid="stSidebar"] * { color:#E2E8F0 !important; }
section[data-testid="stSidebar"] .stRadio label { padding:6px 4px; border-radius:8px; }
.hero { background: radial-gradient(1200px 300px at 0% 0%, #1E3A8A 0%, #0F172A 60%);
        color:#F8FAFC; padding:28px 32px; border-radius:18px; margin-bottom:18px;
        box-shadow: 0 10px 30px rgba(15,23,42,.18); }
.hero h1 { font-size:1.9rem; margin:0 0 6px 0; color:#F8FAFC; letter-spacing:-.01em; }
.hero p { color:#CBD5E1; margin:0; font-size:.98rem; line-height:1.5; }
.hero .tag { display:inline-block; background:rgba(96,165,250,.18); color:#BFDBFE; border:1px solid rgba(147,197,253,.35);
        padding:3px 10px; border-radius:999px; font-size:.75rem; margin-right:6px; margin-bottom:10px; }
.kpi { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:14px 16px;
       box-shadow:0 1px 2px rgba(15,23,42,.04); height:100%; }
.kpi .label { color:var(--muted); font-size:.74rem; text-transform:uppercase; letter-spacing:.06em; font-weight:600; }
.kpi .value { color:var(--ink); font-size:1.55rem; font-weight:700; margin-top:2px; font-variant-numeric: tabular-nums; }
.kpi .sub { color:var(--muted); font-size:.78rem; margin-top:2px; }
.kpi.sm { padding:10px 12px; } .kpi.sm .value { font-size:1.12rem; word-break:break-all; }
.kpi.sm .label { font-size:.66rem; }
.kpi.accent { border-top:3px solid var(--blue); }
.section-title { font-size:1.05rem; font-weight:700; color:var(--ink); margin:18px 0 8px 0; }
.card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:16px 18px; }
.pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem; font-weight:600; }
.pill.ok { background:#DCFCE7; color:#166534; } .pill.warn { background:#FEF3C7; color:#92400E; }
.pill.bad { background:#FEE2E2; color:#991B1B; }
.decision { border-radius:16px; padding:18px 20px; text-align:center; font-weight:800; font-size:1.6rem; letter-spacing:.04em; }
.decision.ref { background:linear-gradient(135deg,#ECFDF5,#D1FAE5); color:#065F46; border:1px solid #6EE7B7; }
.decision.anom { background:linear-gradient(135deg,#FEF2F2,#FEE2E2); color:#991B1B; border:1px solid #FCA5A5; }
.decision small { display:block; font-size:.78rem; font-weight:500; letter-spacing:0; margin-top:4px; opacity:.85; }
.flow { display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
.flow span { background:#EFF6FF; color:#1E40AF; border:1px solid #BFDBFE; padding:5px 10px; border-radius:8px; font-size:.8rem; }
.flow b { color:#94A3B8; }
.flow.p1 span { background:#F1F5F9; color:#334155; border-color:#CBD5E1; }
div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
</style>
"""


def ctx():
    cfg = load_config()
    return cfg, get_paths(cfg)


def read_json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def read_csv(p: Path):
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def load_results(paths) -> dict:
    R = paths.results
    return {
        "verification": read_json(R / "dataset" / "verification.json"),
        "leakage": read_json(R / "dataset" / "leakage_report.json"),
        "m1": read_json(R / "phase1" / "metrics.json"),
        "m2": read_json(R / "phase2" / "metrics.json"),
        "p1_pred": read_csv(R / "phase1" / "predictions.csv"),
        "p2_pred": read_csv(R / "phase2" / "predictions.csv"),
        "ablation": read_csv(R / "ablation" / "ablation.csv"),
        "drift": read_csv(R / "drift" / "drift_metrics.csv"),
        "drift_summary": read_csv(R / "drift" / "drift_summary.csv"),
        "alpha_table": read_csv(R / "phase2" / "selection_alpha_policy_validation.csv"),
        "tl_table": read_csv(R / "phase2" / "selection_topk_lambda_validation.csv"),
        "history": read_csv(R / "phase1" / "training_history.csv"),
        "manifest": read_json(R / "experiment_manifest.json"),
    }


def kpi(label: str, value, sub: str = "", accent: bool = False, small: bool = False) -> str:
    return (f'<div class="kpi{" accent" if accent else ""}{" sm" if small else ""}"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')


def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return f"{float(v):.{nd}f}"


def colorize(arr: np.ndarray, cmap: str = "inferno", vmax: float | None = None) -> np.ndarray:
    import matplotlib
    a = np.asarray(arr, dtype=float)
    vmax = vmax or (np.quantile(a, 0.999) or 1.0)
    a = np.clip(a / (vmax + 1e-12), 0, 1)
    return (matplotlib.colormaps[cmap](a)[..., :3] * 255).astype(np.uint8)


def overlay(img: np.ndarray, heat: np.ndarray, mask: np.ndarray | None = None, alpha: float = 0.55,
            vmax: float | None = None) -> np.ndarray:
    base = np.stack([np.clip(img, 0, 1)] * 3, -1)
    h = colorize(heat, vmax=vmax) / 255.0
    w = alpha * np.clip(np.asarray(heat) / ((vmax or np.quantile(heat, 0.999)) + 1e-12), 0, 1)[..., None]
    out = (1 - w) * base + w * h
    if mask is not None and mask.any():
        from scipy import ndimage
        edge = mask & ~ndimage.binary_erosion(mask, iterations=1)
        out[edge] = [0.13, 0.83, 0.93]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def gray(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def mask_img(m: np.ndarray) -> np.ndarray:
    return (np.asarray(m, bool) * 255).astype(np.uint8)


# ------------------------------------------------------------- plotly
def plotly_layout(fig, height=340, title=None):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=44 if title else 10, b=10),
                      title=dict(text=title, x=0, xanchor="left", font=dict(size=15)) if title else None,
                      template="plotly_white", font=dict(family="Inter, Segoe UI, sans-serif", size=12),
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0, title_text=""),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def roc_fig(p1: pd.DataFrame, p2: pd.DataFrame):
    import plotly.graph_objects as go
    from sklearn.metrics import roc_curve, roc_auc_score
    fig = go.Figure()
    for df, name, col, dash in [(p1, "Phase 1 ConvAE", SLATE, "solid"), (p2, "RCC-ConvAE", BLUE, "dash")]:
        f, t, _ = roc_curve(df.is_anomaly, df.score)
        fig.add_scatter(x=f, y=t, name=f"{name} (AUC {roc_auc_score(df.is_anomaly, df.score):.4f})",
                        line=dict(color=col, width=2.4, dash=dash))
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color="#CBD5E1", dash="dot"), showlegend=False)
    fig.update_xaxes(title="False positive rate")
    fig.update_yaxes(title="True positive rate")
    return plotly_layout(fig, 360, "ROC curve · TEST")


def pr_fig(p1: pd.DataFrame, p2: pd.DataFrame):
    import plotly.graph_objects as go
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig = go.Figure()
    for df, name, col, dash in [(p1, "Phase 1 ConvAE", SLATE, "solid"), (p2, "RCC-ConvAE", BLUE, "dash")]:
        p, r, _ = precision_recall_curve(df.is_anomaly, df.score)
        fig.add_scatter(x=r, y=p, name=f"{name} (AP {average_precision_score(df.is_anomaly, df.score):.4f})",
                        line=dict(color=col, width=2.4, dash=dash))
    fig.update_xaxes(title="Recall")
    fig.update_yaxes(title="Precision")
    return plotly_layout(fig, 360, "Precision–recall · TEST")


def cm_fig(m: dict, title: str):
    import plotly.graph_objects as go
    cm = np.array(m["confusion_matrix"])
    fig = go.Figure(go.Heatmap(z=cm, x=["REFERENCE", "ANOMALY"], y=["REFERENCE", "ANOMALY"],
                               colorscale=[[0, "#EFF6FF"], [1, "#1D4ED8"]], showscale=False,
                               text=[[f"{v:,}" for v in r] for r in cm], texttemplate="%{text}",
                               textfont=dict(size=18)))
    fig.update_yaxes(autorange="reversed", title="Ground truth (mask)")
    fig.update_xaxes(title="Predicted")
    return plotly_layout(fig, 300, title)


def score_hist(pred: pd.DataFrame, title: str, xcol: str = "score", log: bool = False, vline=None):
    import plotly.graph_objects as go
    fig = go.Figure()
    x = pred[xcol]
    if log:
        x = np.log10(np.clip(x, 1e-12, None))
    for a, name, col in [(0, "Reference", TEAL), (1, "Anomaly", RED)]:
        fig.add_histogram(x=x[pred.is_anomaly == a], name=name, marker_color=col, opacity=0.65,
                          histnorm="probability density", nbinsx=60)
    if vline is not None:
        fig.add_vline(x=np.log10(vline) if log else vline, line_dash="dash", line_color="#0F172A",
                      annotation_text="decision", annotation_position="top")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title=("log10 " if log else "") + xcol.replace("_", " "))
    return plotly_layout(fig, 320, title)


LOWER_IS_BETTER = {"FPR", "FNR", "Cost / image"}


def delta_style(row) -> list[str]:
    """Green when Phase 2 is better, red when worse (direction-aware)."""
    d = row["Δ (P2 − P1)"]
    if d is None or pd.isna(d) or abs(d) < 1e-9:
        return [""] * len(row)
    good = (d < 0) if row["Metric"] in LOWER_IS_BETTER else (d > 0)
    css = "background-color:#DCFCE7;color:#166534;font-weight:600" if good else \
        "background-color:#FEE2E2;color:#991B1B;font-weight:600"
    return ["" if c != "Δ (P2 − P1)" else css for c in row.index]
