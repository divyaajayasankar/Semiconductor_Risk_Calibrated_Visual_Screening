# Run Instructions (Windows PowerShell)

All commands are run from the project root `Semiconductor_Risk_Calibrated_Visual_Screening\`.
Linux/macOS: use `/` instead of `\` and `source .venv/bin/activate`.

## 0. Environment (once)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\verify_environment.py
```
**Expected:** every package printed with a version, `ENVIRONMENT: PASS`.
If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then re-run the activate line.

## 1. Place the dataset
Follow `DATASET_SETUP.md` → `data\raw\carinthia_s\data\{images,masks,carinthia-s.csv}`.

## 2. Verify the dataset and create the leakage-free split
```powershell
python experiments\01_verify_dataset.py
```
**Expected:** `Images=4591 Masks=4591`, `Sizes={'480x480': 4591}`, `Empty masks (REFERENCE)=224`,
`Non-empty masks (ANOMALY)=4367`, a role table (TRAIN 90 refs / TEST ≈3100), and
`Dataset verification: PASS | Leakage check: PASS`.
Files: `results\dataset\verification.json`, `image_metadata.csv`, `split_manifest.csv`, `leakage_report.json`.

## 3. Run the full experiment
```powershell
python run_all.py
```
| Step | What happens | Expected console line |
|---|---|---|
| 01 | dataset verification + split | `Dataset verification: PASS` |
| 02 | Phase-1 ConvAE training (skipped if a valid checkpoint exists) | `epoch ... val=... *` then `Best checkpoint: epoch N` |
| 03 | Phase-1 threshold on CALIBRATION, TEST evaluation | `PHASE 1 (TEST): ROC-AUC=...` |
| 04 | freeze Phase 1 (SHA-256 lock) | `Phase-1 frozen: checkpoint sha=...` |
| 05 | RCC policy fit (VALIDATION / CALIBRATION / POLICY_VALIDATION only) | `RCC policy frozen: top-k=... lambda=... alpha*=...` |
| 06 | Phase-2 TEST evaluation + localisation (both phases) | `PHASE 2 (TEST): ...` and `LOCALISATION P1: Dice=... P2: Dice=...` |
| 07 | ablation A–E | ablation table |
| 08 | 9 drift conditions × 4 policies | per-condition tables + `Drift summary` |
| 09 | figures, reports, manifest, paper tables | `Reports written to ...` |
| 10 | packaging readiness | `Packaging readiness: PASS` |

**Runtime (CPU laptop, indicative):** training 5–15 min; scoring ≈2 min; drift 8–15 min; total ≈20–40 min.

Resume / partial runs:
```powershell
python run_all.py --from 04        # re-uses the frozen checkpoint, never retrains
python run_all.py --only 08        # drift only
python run_all.py --retrain        # force Phase-1 retraining
```
Faster drift on slow machines: set `drift: max_test_images: 1000` in `experiment_config.yaml`.

Individual scripts (same effect as the runner):
```powershell
python experiments\02_train_phase1.py
python experiments\03_evaluate_phase1.py
python experiments\04_phase2_rcc.py
python experiments\06_ablation.py
python experiments\05_distribution_drift.py
python experiments\07_generate_reports.py
```

## 4. Outputs to check
- `reports\final_metrics.csv` — Phase 1 vs Phase 2 (all image + pixel metrics)
- `reports\experiment_summary.md` — human-readable summary
- `reports\ablation_table.csv`, `reports\drift_summary.csv`, `reports\localization_summary.csv`
- `results\figures\*.png` — ROC, PR, confusion matrices, score/p-value distributions, α-cost curve,
  training curve, localisation examples, ablation, drift
- `results\experiment_manifest.json` — versions, hashes, hyper-parameters

## 5. Tests
```powershell
pytest -q
```
**Expected:** `20 passed`. Tests use a tiny synthetic fixture in a temp folder; they never touch your results.

## 6. Dashboard
```powershell
streamlit run app\streamlit_app.py
```
**Expected:** browser opens at `http://localhost:8501` with the SemiVision AI dashboard; sidebar shows
`Model & policies: 🟢 ready` and `Experiment results: 🟢 available`. Stop with `Ctrl+C`.

## 7. Update the IEEE paper with your numbers
```powershell
python scripts\export_paper_tables.py
```
Rewrites `paper\tables\*.tex` (tables + `macros.tex` used in the text). Compile in Overleaf (upload the
`paper\` folder) or locally: `pdflatex main; bibtex main; pdflatex main; pdflatex main`.
Re-read the Results/Discussion wording after regenerating in case a direction of change differs.

## 8. Build the final ZIP
```powershell
python scripts\package_project.py              # includes the dataset if present
python scripts\package_project.py --no-dataset # smaller ZIP without images/masks
```
**Expected:** `Readiness check: PASS` then
`ZIP created: dist\Risk_Calibrated_Semiconductor_Visual_Screening.zip`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `Dataset CSV not found` | check the folder layout in `DATASET_SETUP.md` |
| `images_missing > 0` | CSV paths don’t match files; keep `images\` and `masks\` beside the CSV |
| `Phase-1 policy is not frozen` | run `python run_all.py --from 03` |
| `Checkpoint changed after Phase-1 freeze` | re-run from 03 so Phase 1 is re-frozen on the new checkpoint |
| Dashboard says model not found | run `python run_all.py` (at least steps 01–06) |
| Out of memory | lower `training.eval_batch_size` or `data.image_size` in `experiment_config.yaml` |
