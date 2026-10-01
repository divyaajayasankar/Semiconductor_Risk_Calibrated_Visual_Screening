# Risk-Calibrated and Cost-Aware Anomaly Detection Visual Screening for Semiconductor Manufacturing Under Distribution Drift

**Application:** SemiVision AI – Semiconductor Defect Intelligence Platform
**Dataset:** Carinthia-S SEM images (4,591 images + expert masks) · **Seed:** 42 · **Runs on:** Windows / Linux, CPU

---

## 1. Problem statement
Scanning-electron-microscope (SEM) review of wafer surfaces produces far more images than engineers can inspect.
Defect types are diverse, rare and evolve, so fully supervised classifiers are brittle. An automated screen must
(i) learn from defect-free **reference** images, (ii) output decisions whose false-alarm risk is **quantified**,
(iii) reflect that a **missed defect costs far more than a false alarm**, (iv) **localise** the defect, and
(v) stay reliable when **imaging conditions drift** (brightness, contrast, focus, noise).

## 2. Research objective
Build and evaluate a two-phase, leakage-free pipeline:
**Phase 1** — a reference-only 2-D convolutional autoencoder (ConvAE) with a global reconstruction-error threshold;
**Phase 2 — RCC-ConvAE** — a risk-calibrated, cost-aware decision framework on the *same frozen* ConvAE that adds
local evidence, robust score fusion, conformal p-values, cost-optimal operating-point selection, pixel-level
localisation and drift recalibration.

## 3. Research gap
Reconstruction-based detectors report ranking metrics (AUC) but leave the operating threshold ad hoc, give no
finite-sample false-alarm guarantee, ignore asymmetric manufacturing cost, rarely evaluate localisation against
expert masks on real SEM data, and are seldom stress-tested under imaging drift.

## 4. Novelty (what is new here)
1. Conformal risk calibration using **reference-only** calibration images for SEM visual screening.
2. **Cost-aware α\*** selection (10·FN + 1·FP) on a dedicated policy-validation split.
3. **Global + local top-k** reconstruction-evidence fusion with robust (median/MAD) normalisation.
4. Pixel-level localisation evaluated against expert masks (Dice, IoU, pixel AUROC/AP).
5. Frozen vs **label-free recalibration** under nine imaging-drift conditions.
6. A strict five-role split with an automated leakage audit.

RCC-ConvAE is **not** a new CNN; it is a decision framework built around a frozen ConvAE backbone.

## 5. Base paper
M. Gorman, X. Ding, L. Maguire, D. Coyle, “Anomaly Detection in Batch Manufacturing Processes Using Localized
Reconstruction Errors From 1-D Convolutional AutoEncoders,” *IEEE Trans. Semicond. Manuf.*, 36(1):147–150, 2023,
doi:10.1109/TSM.2022.3216032. The paper reports **AUC = 1.00** (not “100 % accuracy”) on 1-D batch-process data.
It is methodological inspiration (localised reconstruction error); results are not directly comparable.

## 6. Dataset and ground truth
Carinthia-S (Kofler & Hasić, Zenodo 2025, doi:10.5281/zenodo.16895427, CC BY 4.0): 480×480 grayscale SEM images of
a single production layer with binary expert masks and a six-value metadata class.
**Ground truth comes from the mask only:** empty mask → REFERENCE; non-empty mask → ANOMALY.
The class number is never used as the normal label. See `DATASET_SETUP.md`.

## 7. Why reference-only (unsupervised) learning
Only ≈5 % of images are defect-free and defect appearance is open-ended. Training on references only means any
unfamiliar structure — including unseen defect types — produces high reconstruction error.

## 8. Method
**Phase 1 (2-D ConvAE).** 4 strided Conv–ReLU encoder / 4 ConvTranspose decoder (256×256 → 16×16×64 → 256×256),
MSE loss, Adam, early stopping on reference-validation loss.
Error map `E_i = (x_i − x̂_i)²`; score `S_global = mean(E)`; threshold = 95th percentile of calibration-reference scores.

**Phase 2 (RCC-ConvAE).**
| Step | Formula | Fitted on |
|---|---|---|
| Local evidence | `S_local = mean(top-k fraction of G_σ * E)`, k ∈ {0.001, 0.005, 0.01, 0.02, 0.05} | — |
| Robust normalisation | `Z = (S − median) / (1.4826·MAD)` | VALIDATION references |
| Fusion | `S_fused = Z_global + λ·Z_local` | k, λ on VALIDATION |
| Conformal p-value | `p = (1 + #{α_i ≥ α_new}) / (n + 1)` | CALIBRATION references |
| Cost-aware decision | `α* = argmin 10·FN + 1·FP`; ANOMALY iff `p ≤ α*` | POLICY_VALIDATION |
| Localisation | threshold quantile + opening radius maximising Dice | VALIDATION anomalies |

## 9. Split and leakage prevention
| Role | Reference share | Anomaly share | Purpose |
|---|---|---|---|
| TRAIN | 40 % | 0 % | ConvAE weights |
| VALIDATION | 10 % | 10 % | early stopping (refs), top-k, λ, normalisation, localisation |
| CALIBRATION | 20 % | 0 % | Phase-1 threshold, conformal scores |
| POLICY_VALIDATION | 10 % | 20 % | cost-aware α* |
| TEST | 20 % | 70 % | final evaluation only, once, after freezing |

Anomalies are stratified by metadata class. `results/dataset/leakage_report.json` verifies all TEST
intersections are empty; `tests/test_no_test_leakage.py` proves that tampering with TEST scores changes no policy.

## 10. Metrics
Image level: ROC-AUC, PR-AUC, accuracy, balanced accuracy, precision, recall, specificity, F1, FPR, FNR,
TN/FP/FN/TP, cost/image. Pixel level: Dice, IoU (anomalous TEST images), pixel AUROC and AP (histogram estimator).

## 11. Ablation (A–E) and distribution drift
A Phase 1 · B global+local · C +conformal (α = 0.05) · D +cost-aware α\* (= RCC-ConvAE) · E local only.
Drift: clean, brightness ±0.10, contrast 0.70, gamma 1.50, noise 0.02/0.05, Gaussian blur 1.0/2.0 — each for
Phase 1/2 frozen and recalibrated (calibration references re-scored under the shift; no labels, no TEST).

## 12. Dashboard
`streamlit run app\streamlit_app.py` — Home, Single SEM Screening, Batch Screening (CSV export), Research Metrics,
Distribution Drift, Ablation, Model & Policy Information, About. All values are read from result files.

## 13. Commands (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\verify_environment.py
python experiments\01_verify_dataset.py
python run_all.py
pytest -q
streamlit run app\streamlit_app.py
python scripts\export_paper_tables.py
python scripts\package_project.py
```
Details and expected outputs: `RUN_INSTRUCTIONS.md`.

## 14. Repository map
```
src/            config, seed, data, preprocessing, models_convae, training, scoring, conformal,
                cost_policy, localization, drift, metrics, evaluation, inference, plots, utils
experiments/    01_verify_dataset · 02_train_phase1 · 03_evaluate_phase1 · 04_phase2_rcc ·
                05_distribution_drift · 06_ablation · 07_generate_reports
app/            streamlit_app.py, dashboard_utils.py
models/         convae_best.pt, phase1_policy.json, phase2/{rcc_policy,normalization,localization_threshold}.json,
                phase2/calibration_scores.npy
results/        dataset/ phase1/ phase2/ ablation/ drift/ figures/ experiment_manifest.json
reports/        final_metrics.csv, localization_summary.csv, ablation_table.csv, drift_summary.csv, experiment_summary.md
paper/          IEEE conference paper (LaTeX) — tables regenerate from your results
tests/          pytest suite (uses a tiny synthetic fixture; never reported as results)
```

## 15. Limitations
- Only 224 reference images exist; calibration uses ≈45, so the smallest attainable p-value is ≈1/46 ≈ 0.022 and
  α below that cannot flag anything.
- Test set is anomaly-heavy (≈98.5 % anomalies), so accuracy/F1/PR-AUC are dominated by the anomaly class;
  balanced accuracy, FPR and cost/image are the more informative image-level metrics.
- Drift is simulated by photometric/blur perturbations, not by real tool or recipe changes.
- One production layer; no wafer/lot identifiers are available, so the split is image-level rather than lot-level.
- Conformal guarantees assume exchangeability between calibration and new reference images.

## 16. Final interpretation
Report what the run produces. If Phase 1 and Phase 2 have similar image-level scores, that is expected — both
use the same backbone. Phase 2’s contribution is then in calibrated risk, cost-optimal operating point,
localisation quality and recoverability under drift, which the ablation and drift tables quantify.
