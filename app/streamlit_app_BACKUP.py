"""SemiVision AI – Semiconductor Defect Intelligence Platform (Streamlit dashboard).

Launch:  streamlit run app/streamlit_app.py
All numbers shown are read from the result files produced by `python run_all.py` (nothing hard-coded).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dashboard_utils as U  # noqa: E402
from src.inference import ScreeningEngine  # noqa: E402

st.set_page_config(page_title="SemiVision AI", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
st.markdown(U.CSS, unsafe_allow_html=True)

CFG, PATHS = U.ctx()
PAGES = ["🏠  Home / Project Overview", "🔍  Single SEM Screening", "🗂️  Batch Screening", "📊  Research Metrics",
         "🌪️  Distribution Drift", "🧪  Ablation Study", "⚙️  Model & Policy Information", "📘  About / Methodology"]


@st.cache_resource(show_spinner="Loading frozen ConvAE and policies ...")
def get_engine():
    ok, missing = ScreeningEngine.available(CFG)
    return (ScreeningEngine(CFG), []) if ok else (None, missing)


@st.cache_data(show_spinner=False, ttl=60)
def get_results():
    return U.load_results(PATHS)


engine, missing = get_engine()
RES = get_results()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🔬 SemiVision AI")
    st.caption("Semiconductor Defect Intelligence Platform")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**System status**")
    st.markdown(f"Model & policies: {'🟢 ready' if engine else '🔴 not found'}")
    st.markdown(f"Experiment results: {'🟢 available' if RES['m2'] else '🟠 run pipeline'}")
    st.markdown(f"Dataset: {'🟢 found' if PATHS.csv.exists() else '⚪ not present'}")
    st.markdown("---")
    st.caption("Seed 42 · CPU inference · frozen policies")


def need_results():
    st.warning("No experiment results found yet. Run `python run_all.py` from the project root, then refresh.")
    st.stop()


def need_engine():
    st.error("Frozen model/policies not found: " + ", ".join(missing) +
             ".\n\nRun `python run_all.py` (or at least steps 01–06) first.")
    st.stop()


def hero(title, subtitle, tags=()):
    t = "".join(f'<span class="tag">{x}</span>' for x in tags)
    st.markdown(f'<div class="hero">{t}<h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpis(items, accent_first=False):
    cols = st.columns(len(items))
    for i, (c, (lab, val, sub)) in enumerate(zip(cols, items)):
        c.markdown(U.kpi(lab, val, sub, accent=accent_first and i == 0), unsafe_allow_html=True)


# ======================================================================= HOME
if page == PAGES[0]:
    hero(U.APP_NAME.split("–")[0].strip() + " · Research Dashboard", U.RESEARCH_TITLE,
         ["Carinthia-S SEM", "Reference-only ConvAE", "Conformal risk", "Cost-aware", "Drift-robust"])
    v, m1, m2 = RES["verification"], RES["m1"], RES["m2"]
    if v:
        kpis([("SEM images", U.fmt(v["images_found"]), f"{list(v['image_sizes'])[0] if v['image_sizes'] else ''} grayscale"),
              ("Reference (empty mask)", U.fmt(v["empty_masks_reference"]), "normal ground truth"),
              ("Anomaly (non-empty mask)", U.fmt(v["non_empty_masks_anomaly"]), "defect ground truth"),
              ("Verification", v["status"], f"leakage check: {RES['leakage']['status'] if RES['leakage'] else '–'}")])
    if m2:
        st.markdown('<div class="section-title">Headline results · held-out TEST set</div>', unsafe_allow_html=True)
        l1, l2 = m1.get("localization") or {}, m2.get("localization") or {}
        kpis([("ROC-AUC (RCC)", U.fmt(m2["roc_auc"]), f"Phase 1: {U.fmt(m1['roc_auc'])}"),
              ("Balanced accuracy", U.fmt(m2["balanced_accuracy"]), f"Phase 1: {U.fmt(m1['balanced_accuracy'])}"),
              ("Cost / image", U.fmt(m2["cost_per_image"]), f"10·FN + 1·FP · Phase 1: {U.fmt(m1['cost_per_image'])}"),
              ("Defect Dice", U.fmt(l2.get("dice")), f"Phase 1: {U.fmt(l1.get('dice'))}"),
              ("Pixel AUROC", U.fmt(l2.get("pixel_auroc")), f"Phase 1: {U.fmt(l1.get('pixel_auroc'))}")], True)
    else:
        st.info("Experiment results will appear here after running `python run_all.py`.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Research objective</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">Screen SEM images of semiconductor wafer surfaces for defects using a model '
                    'trained <b>only on reference (defect-free) images</b>, and turn its reconstruction evidence into '
                    '<b>calibrated, cost-aware REFERENCE / ANOMALY decisions</b> with pixel-level defect localisation '
                    'that remain reliable under imaging drift.</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Phase 1 · 2-D ConvAE baseline</div>', unsafe_allow_html=True)
        st.markdown('<div class="flow p1"><span>SEM image</span><b>→</b><span>Reference-only ConvAE</span><b>→</b>'
                    '<span>Reconstruction</span><b>→</b><span>Global MSE</span><b>→</b><span>Calibration threshold</span>'
                    '<b>→</b><span>Decision</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Phase 2 · RCC-ConvAE (proposed)</div>', unsafe_allow_html=True)
        st.markdown('<div class="flow"><span>Frozen ConvAE</span><b>→</b><span>Error map</span><b>→</b>'
                    '<span>Global + local top-k</span><b>→</b><span>Robust z-fusion</span><b>→</b>'
                    '<span>Conformal p-value</span><b>→</b><span>Cost-aware α*</span><b>→</b><span>Decision + mask</span>'
                    '<b>→</b><span>Drift recalibration</span></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title">Dataset & split</div>', unsafe_allow_html=True)
        if v and v.get("split"):
            rc = pd.DataFrame(v["split"]["role_counts"]).T.reset_index().rename(columns={"index": "role"})
            import plotly.express as px
            fig = px.bar(rc.melt(id_vars="role", value_vars=["reference", "anomaly"]), y="role", x="value",
                         color="variable", orientation="h", color_discrete_map={"reference": U.TEAL, "anomaly": U.RED},
                         text="value")
            st.plotly_chart(U.plotly_layout(fig, 300), use_container_width=True)
            cd = pd.DataFrame([{"class": k, **vv} for k, vv in v["class_vs_ground_truth"].items()])
            st.dataframe(cd, hide_index=True, use_container_width=True)
        else:
            st.info("Run `python experiments/01_verify_dataset.py` to populate dataset statistics.")
        st.markdown('<div class="section-title">Model status</div>', unsafe_allow_html=True)
        if engine:
            i = engine.info()
            st.markdown(f'<span class="pill ok">Frozen ConvAE · {i["parameters"]:,} params</span> '
                        f'<span class="pill ok">top-k {i["topk"]:g}</span> <span class="pill ok">λ {i["lambda"]:g}</span> '
                        f'<span class="pill ok">α* {i["alpha"]:g}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill bad">Model not trained yet</span>', unsafe_allow_html=True)

# ============================================================ SINGLE SCREENING
elif page == PAGES[1]:
    hero("Single SEM Screening", "Upload one SEM image (or pick a held-out TEST image) and screen it with the frozen policy.")
    if not engine:
        need_engine()
    c0, c1, c2 = st.columns([1.3, 1, 1])
    src = c0.radio("Image source", ["Upload", "Held-out TEST image"], horizontal=True)
    phase = c1.radio("Screening model", ["Phase 2 · RCC-ConvAE", "Phase 1 · ConvAE"], horizontal=False)
    phase_n = 2 if phase.startswith("Phase 2") else 1
    img, gt, name = None, None, None
    if src == "Upload":
        up = c2.file_uploader("SEM image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
        if up:
            img, name = Image.open(up), up.name
    else:
        try:
            from src import data as D
            split = D.load_split(PATHS.split_manifest, PATHS.data_dir, PATHS.csv)
            test = split[split.role == "TEST"].sort_values(["is_anomaly", "filename"])
            opts = [f"{r.filename}  ·  {'ANOMALY' if r.is_anomaly else 'REFERENCE'} (class {r.label})"
                    for r in test.itertuples()]
            sel = c2.selectbox("TEST image", opts)
            row = test.iloc[opts.index(sel)]
            img, name = Image.open(row.image_path), row.filename
            if row.is_anomaly:
                from src.preprocessing import load_mask
                gt = load_mask(row.mask_path, CFG["data"]["image_size"], CFG["data"]["mask_threshold"])
            else:
                gt = np.zeros((CFG["data"]["image_size"],) * 2, bool)
        except Exception as e:
            st.info(f"Dataset not available on this machine ({e}). Use Upload instead.")
    if img is not None:
        r = engine.screen(img, phase_n)
        cls = "anom" if r["decision"] == "ANOMALY" else "ref"
        d1, d2 = st.columns([1, 2.4])
        with d1:
            st.markdown(f'<div class="decision {cls}">{r["decision"]}<small>{r["rule"]}</small></div>',
                        unsafe_allow_html=True)
            st.write("")
            items = [("Global score (MSE)", f"{r['global_score']:.6f}"), ("Local top-k score", f"{r['local_score']:.6f}")]
            if phase_n == 2:
                items.append(("Fused score", f"{r['fused_score']:.3f}"))
            items += [("Conformal p-value", f"{r['p_value']:.4f}"),
                      ("Threshold" if phase_n == 1 else "α*", f"{r['threshold']:.4g}"),
                      ("Pred. defect area", f"{100 * r['pred_mask'].mean():.2f}%")]
            for j in range(0, len(items), 2):
                cc = st.columns(2)
                for c, (lab, val) in zip(cc, items[j:j + 2]):
                    c.markdown(U.kpi(lab, val, small=True), unsafe_allow_html=True)
                st.write("")
        with d2:
            cols = st.columns(3)
            cols[0].image(U.gray(r["image"]), caption=f"Uploaded · {name}", use_container_width=True)
            cols[1].image(U.gray(r["recon"]), caption="Reconstruction", use_container_width=True)
            cols[2].image(U.colorize(r["heatmap"], vmax=r["heat_vmax"]), caption="Error heatmap (fixed scale)", use_container_width=True)
            cols = st.columns(3)
            cols[0].image(U.overlay(r["image"], r["heatmap"], gt, vmax=r["heat_vmax"]), caption="Heatmap overlay (cyan = GT edge)",
                          use_container_width=True)
            cols[1].image(U.mask_img(r["pred_mask"]), caption="Predicted defect mask", use_container_width=True)
            if gt is not None:
                from src.localization import dice_iou
                dsc, iou = dice_iou(r["pred_mask"], gt) if gt.any() else (None, None)
                cols[2].image(U.mask_img(gt), caption="Ground-truth mask" +
                              (f" · Dice {dsc:.3f} · IoU {iou:.3f}" if dsc is not None else " (empty = reference)"),
                              use_container_width=True)
            else:
                cols[2].info("Ground-truth mask is shown when a held-out TEST image is selected.")
        st.caption("The decision is a calibrated hypothesis test (p-value vs α*), not a confidence percentage.")

# ============================================================= BATCH SCREENING
elif page == PAGES[2]:
    hero("Batch Screening", "Screen many SEM images at once with the frozen policy and export the decisions.")
    if not engine:
        need_engine()
    c1, c2 = st.columns([2, 1])
    files = c1.file_uploader("SEM images", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
                             accept_multiple_files=True)
    phase = c2.radio("Screening model", ["Phase 2 · RCC-ConvAE", "Phase 1 · ConvAE"])
    if files:
        prog = st.progress(0.0, text="Screening ...")
        items = []
        for i, f in enumerate(files):
            items.append((f.name, Image.open(f)))
            prog.progress((i + 1) / len(files), text=f"Loaded {i + 1}/{len(files)}")
        df = engine.screen_batch(items, 2 if phase.startswith("Phase 2") else 1)
        prog.empty()
        n = len(df)
        na = int((df.decision == "ANOMALY").sum())
        kpis([("Total screened", U.fmt(n), ""), ("Reference", U.fmt(n - na), ""), ("Anomaly", U.fmt(na), ""),
              ("Anomaly rate", f"{100 * na / max(n, 1):.1f}%", "")], True)
        st.write("")
        st.dataframe(df.style.map(lambda v: "color:#991B1B;font-weight:700" if v == "ANOMALY" else
                                  ("color:#065F46;font-weight:700" if v == "REFERENCE" else ""), subset=["decision"]),
                     hide_index=True, use_container_width=True)
        st.download_button("⬇️  Download results (CSV)", df.to_csv(index=False).encode(), "semivision_batch_results.csv",
                           "text/csv", type="primary")

# ============================================================ RESEARCH METRICS
elif page == PAGES[3]:
    hero("Research Metrics", "Phase 1 ConvAE vs Phase 2 RCC-ConvAE on the held-out TEST set (loaded from result files).")
    m1, m2 = RES["m1"], RES["m2"]
    if not (m1 and m2):
        need_results()
    l1, l2 = m1.get("localization") or {}, m2.get("localization") or {}
    rows = [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("F1", "f1"), ("Balanced accuracy", "balanced_accuracy"),
            ("FPR", "fpr"), ("FNR", "fnr"), ("Cost / image", "cost_per_image")]
    table = [{"Metric": n, "Phase 1 ConvAE": m1.get(k), "Phase 2 RCC-ConvAE": m2.get(k)} for n, k in rows]
    table += [{"Metric": n, "Phase 1 ConvAE": l1.get(k), "Phase 2 RCC-ConvAE": l2.get(k)}
              for n, k in [("Dice", "dice"), ("IoU", "iou"), ("Pixel AUROC", "pixel_auroc"), ("Pixel AP", "pixel_ap")]]
    t = pd.DataFrame(table)
    t["Δ (P2 − P1)"] = t["Phase 2 RCC-ConvAE"].astype(float) - t["Phase 1 ConvAE"].astype(float)
    kpis([("ROC-AUC", U.fmt(m2["roc_auc"]), f"P1 {U.fmt(m1['roc_auc'])}"),
          ("F1", U.fmt(m2["f1"]), f"P1 {U.fmt(m1['f1'])}"),
          ("FNR", U.fmt(m2["fnr"]), f"P1 {U.fmt(m1['fnr'])}"),
          ("Dice", U.fmt(l2.get("dice")), f"P1 {U.fmt(l1.get('dice'))}"),
          ("IoU", U.fmt(l2.get("iou")), f"P1 {U.fmt(l1.get('iou'))}"),
          ("Pixel AUROC", U.fmt(l2.get("pixel_auroc")), f"P1 {U.fmt(l1.get('pixel_auroc'))}")], True)
    st.write("")
    a, b = st.columns([1.15, 1])
    a.dataframe(t.style.format({c: "{:.4f}" for c in t.columns[1:]}, na_rep="–")
                .apply(U.delta_style, axis=1),
                hide_index=True, use_container_width=True, height=420)
    import plotly.graph_objects as go
    fig = go.Figure()
    bars = [("Dice", l1.get("dice"), l2.get("dice")), ("IoU", l1.get("iou"), l2.get("iou")),
            ("Pixel AUROC", l1.get("pixel_auroc"), l2.get("pixel_auroc"))]
    fig.add_bar(x=[x[0] for x in bars], y=[x[1] for x in bars], name="Phase 1", marker_color="#94A3B8",
                text=[U.fmt(x[1], 3) for x in bars])
    fig.add_bar(x=[x[0] for x in bars], y=[x[2] for x in bars], name="RCC-ConvAE", marker_color=U.BLUE,
                text=[U.fmt(x[2], 3) for x in bars])
    b.plotly_chart(U.plotly_layout(fig, 420, "Pixel-level defect localisation"), use_container_width=True)
    p1, p2 = RES["p1_pred"], RES["p2_pred"]
    c1, c2 = st.columns(2)
    c1.plotly_chart(U.roc_fig(p1, p2), use_container_width=True)
    c2.plotly_chart(U.pr_fig(p1, p2), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(U.cm_fig(m1, "Confusion matrix · Phase 1"), use_container_width=True)
    c2.plotly_chart(U.cm_fig(m2, "Confusion matrix · RCC-ConvAE"), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(U.score_hist(p1, "Phase-1 global score distribution", log=True, vline=m1.get("decision_threshold")),
                    use_container_width=True)
    c2.plotly_chart(U.score_hist(p2, "Conformal p-values · RCC-ConvAE", xcol="p_value", vline=m2.get("selected_alpha")),
                    use_container_width=True)
    if RES["alpha_table"] is not None:
        import plotly.graph_objects as go
        at = RES["alpha_table"]
        fig = go.Figure()
        fig.add_scatter(x=at.alpha, y=at.total_cost, mode="lines+markers", name="Total cost", line=dict(color=U.BLUE))
        fig.add_bar(x=at.alpha, y=at.fn_cost, name="FN cost (×10)", marker_color=U.RED, opacity=0.45, width=0.006)
        fig.add_bar(x=at.alpha, y=at.fp_cost, name="FP cost (×1)", marker_color=U.AMBER, opacity=0.45, width=0.006)
        fig.add_vline(x=m2["selected_alpha"], line_dash="dash", annotation_text=f"α* = {m2['selected_alpha']:g}")
        st.plotly_chart(U.plotly_layout(fig, 320, "Cost-aware α selection · POLICY-VALIDATION set"),
                        use_container_width=True)
    fig_path = PATHS.figures / "localization_examples.png"
    if fig_path.exists():
        st.markdown('<div class="section-title">Localisation examples (TEST)</div>', unsafe_allow_html=True)
        st.image(str(fig_path), use_container_width=True)
    with st.expander("Full metric JSON"):
        st.json({"phase1": m1, "phase2": m2})

# ============================================================ DRIFT
elif page == PAGES[4]:
    hero("Distribution Drift", "Imaging perturbations (brightness, contrast, gamma, noise, blur): frozen vs recalibrated policies.")
    dr, ds = RES["drift"], RES["drift_summary"]
    if dr is None:
        need_results()
    import plotly.express as px
    metric = st.selectbox("Metric", ["balanced_accuracy", "roc_auc", "pr_auc", "f1", "fpr", "fnr", "cost_per_image"])
    fig = px.bar(dr, x="condition", y=metric, color="policy", barmode="group",
                 color_discrete_map=U.POLICY_COLORS, category_orders={"policy": list(U.POLICY_COLORS)})
    st.plotly_chart(U.plotly_layout(fig, 420, f"{metric.replace('_', ' ').title()} by imaging condition"),
                    use_container_width=True)
    if ds is not None:
        st.markdown('<div class="section-title">Aggregate over shifted conditions</div>', unsafe_allow_html=True)
        cols = st.columns(len(ds))
        for c, r in zip(cols, ds.itertuples()):
            c.markdown(U.kpi(r.policy, U.fmt(r.mean_balanced_accuracy), f"mean BA · worst {U.fmt(r.worst_balanced_accuracy)}"
                             f" ({r.worst_condition})"), unsafe_allow_html=True)
        st.write("")
        st.dataframe(ds.round(4), hide_index=True, use_container_width=True)
    heat = dr.pivot(index="policy", columns="condition", values=metric).reindex(list(U.POLICY_COLORS))
    heat = heat[list(dict.fromkeys(dr.condition))]
    fig = px.imshow(heat, text_auto=".3f", color_continuous_scale="Blues" if metric not in ("fpr", "fnr", "cost_per_image")
                    else "Reds", aspect="auto")
    st.plotly_chart(U.plotly_layout(fig, 300, f"{metric} heatmap"), use_container_width=True)
    st.caption("Recalibration re-estimates the Phase-1 threshold / Phase-2 conformal calibration scores from shifted "
               "CALIBRATION reference images only; no TEST labels are used.")

# ============================================================ ABLATION
elif page == PAGES[5]:
    hero("Ablation Study", "Contribution of local evidence, conformal calibration and cost-aware α on the TEST set.")
    ab = RES["ablation"]
    if ab is None:
        need_results()
    import plotly.graph_objects as go
    c1, c2, c3 = st.columns(3)
    for c, (col, t) in zip([c1, c2, c3], [("balanced_accuracy", "Balanced accuracy"), ("cost_per_image", "Cost / image"),
                                          ("dice", "Dice")]):
        fig = go.Figure(go.Bar(x=ab.variant, y=ab[col], text=[U.fmt(v, 3) for v in ab[col]],
                               marker_color=[U.BLUE if v == "D" else "#94A3B8" for v in ab.variant]))
        c.plotly_chart(U.plotly_layout(fig, 280, t), use_container_width=True)
    st.dataframe(ab.round(4), hide_index=True, use_container_width=True)
    st.caption("A = Phase 1 · B = global+local · C = +conformal (fixed α) · D = +cost-aware α* (RCC-ConvAE) · E = local only.")

# ============================================================ MODEL INFO
elif page == PAGES[6]:
    hero("Model & Policy Information", "Frozen checkpoint, calibrated thresholds and selection provenance.")
    if not engine:
        need_engine()
    i = engine.info()
    kpis([("Parameters", f"{i['parameters']:,}", i["input"]), ("Seed", str(i["seed"]), f"best epoch {i['best_epoch']}"),
          ("Phase-1 threshold", f"{i['phase1_threshold']:.3g}", f"calibration n = {i['phase1_calibration_size']}"),
          ("top-k · λ · α*", f"{i['topk']:g} · {i['lambda']:g} · {i['alpha']:g}", f"calibration n = {i['calibration_size']}"),
          ("Cost FN : FP", f"{i['cost_fn']:g} : {i['cost_fp']:g}", "missed defect is 10× worse")], True)
    st.write("")
    c1, c2 = st.columns(2)
    c1.markdown('<div class="section-title">Version hashes</div>', unsafe_allow_html=True)
    c1.dataframe(pd.DataFrame([{"artifact": "convae_best.pt", "sha256[:16]": i["checkpoint_sha256"]},
                               {"artifact": "rcc_policy.json", "sha256[:16]": i["policy_sha256"]},
                               {"artifact": "split_manifest.csv",
                                "sha256[:16]": (RES["manifest"] or {}).get("split_manifest_sha256")}]),
                 hide_index=True, use_container_width=True)
    c1.markdown('<div class="section-title">Localisation rules</div>', unsafe_allow_html=True)
    c1.json({"phase1": i["loc_phase1"], "phase2": i["loc_phase2"]})
    c2.markdown('<div class="section-title">Selection provenance (no TEST data)</div>', unsafe_allow_html=True)
    c2.json(engine.p2.get("selection", {}))
    if RES["tl_table"] is not None:
        import plotly.express as px
        tl = RES["tl_table"].pivot(index="topk", columns="lambda", values="val_roc_auc")
        fig = px.imshow(tl, text_auto=".4f", aspect="auto", color_continuous_scale="Blues",
                        labels=dict(x="λ", y="top-k", color="val AUC"))
        c2.plotly_chart(U.plotly_layout(fig, 300, "Validation ROC-AUC over (top-k, λ)"), use_container_width=True)
    if RES["history"] is not None:
        import plotly.graph_objects as go
        h = RES["history"]
        fig = go.Figure()
        fig.add_scatter(x=h.epoch, y=h.train_loss, name="train (reference)", line=dict(color=U.SLATE))
        fig.add_scatter(x=h.epoch, y=h.val_loss, name="validation (reference)", line=dict(color=U.BLUE))
        fig.update_yaxes(type="log", title="MSE")
        st.plotly_chart(U.plotly_layout(fig, 300, "Phase-1 training curve"), use_container_width=True)
    if RES["manifest"]:
        with st.expander("Experiment manifest"):
            st.json(RES["manifest"])

# ============================================================ ABOUT
else:
    hero("About / Methodology", U.RESEARCH_TITLE)
    st.markdown(r"""
**Ground truth.** Carinthia-S masks define the label: an *empty* mask is REFERENCE, a *non-empty* mask is ANOMALY.
The metadata class number is never used as the normal/defect label.

**Phase 1 — 2-D ConvAE.** A lightweight convolutional autoencoder is trained on reference images only.
For an image $x$ with reconstruction $\hat{x}$: $E_i=(x_i-\hat{x}_i)^2$, $S_{global}=\frac{1}{N}\sum_i E_i$.
The decision threshold is the 95th percentile of reference CALIBRATION scores.

**Phase 2 — RCC-ConvAE** (on the *same frozen* ConvAE):
1. Smoothed error map $\tilde{E}=G_\sigma * E$; local evidence $S_{local}$ = mean of the top-$k$ fraction of $\tilde{E}$.
2. Robust normalisation $Z=(S-\mathrm{med})/(1.4826\,\mathrm{MAD})$ fitted on VALIDATION references.
3. Fusion $S_{fused}=Z_{global}+\lambda Z_{local}$; $k,\lambda$ chosen on VALIDATION.
4. Conformal p-value $p=\frac{1+\#\{i:\alpha_i\ge\alpha_{new}\}}{n+1}$ against CALIBRATION references.
5. Cost-aware $\alpha^*=\arg\min_\alpha\,10\,FN+1\,FP$ on POLICY-VALIDATION; flag ANOMALY if $p\le\alpha^*$.
6. Pixel masks: threshold + morphology chosen on VALIDATION anomalies.
7. Drift: brightness, contrast, gamma, noise, blur — frozen vs. reference-only recalibration.

**Leakage control.** TEST images are used once, after every component is frozen (see `results/dataset/leakage_report.json`).

**Base paper.** Gorman *et al.*, IEEE TSM 36(1), 2023 — 1-D CAE with localised reconstruction error, reported AUC = 1.00
on batch-process data. It is methodological inspiration, not a directly comparable benchmark.

**Scope.** RCC-ConvAE is a risk-calibrated, cost-aware *decision framework* around a frozen ConvAE backbone, not a new CNN.
""")
