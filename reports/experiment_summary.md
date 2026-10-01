# Experiment Summary

**Risk-Calibrated and Cost-Aware Anomaly Detection Visual Screening for Semiconductor Manufacturing Under Distribution Drift**

Generated: 2026-10-01 14:42:03  
Seed: 42 | Device: cpu | Python 3.11.9

## Dataset (Carinthia-S)

- Images: 4591 | Masks: 4591 | CSV rows: 4591
- Reference (empty mask): 226 | Anomaly (non-empty): 4365
- Image sizes: {'480x480': 4591} | modes: {'L': 4591}
- Split roles: TRAIN=90 (90 ref / 0 anom), VALIDATION=460 (23 ref / 437 anom), CALIBRATION=45 (45 ref / 0 anom), POLICY_VALIDATION=897 (23 ref / 874 anom), TEST=3099 (45 ref / 3054 anom)

## Frozen policies

- Phase 1 threshold: 9.72117e-05 (0.950-quantile of CALIBRATION reference global scores (finite-sample corrected))
- Phase 2: top-k = 0.001, lambda = 1.0, alpha* = 0.03, calibration n = 45, costs FN:FP = 10.0:1.0

## Image-level results (TEST)

| Metric | Phase 1 ConvAE | Phase 2 RCC-ConvAE |
|---|---|---|
| roc_auc | 1.0000 | 1.0000 |
| pr_auc | 1.0000 | 1.0000 |
| accuracy | 0.9990 | 0.9990 |
| balanced_accuracy | 0.9667 | 0.9667 |
| precision | 0.9990 | 0.9990 |
| recall | 1.0000 | 1.0000 |
| specificity | 0.9333 | 0.9333 |
| f1 | 0.9995 | 0.9995 |
| fpr | 0.0667 | 0.0667 |
| fnr | 0.0000 | 0.0000 |
| tn | 42 | 42 |
| fp | 3 | 3 |
| fn | 0 | 0 |
| tp | 3054 | 3054 |
| cost_per_image | 0.0010 | 0.0010 |

## Pixel-level localisation (TEST)

| Metric | Phase 1 | Phase 2 |
|---|---|---|
| dice | 0.4073 | 0.5957 |
| iou | 0.2643 | 0.4426 |
| pixel_auroc | 0.9205 | 0.9769 |
| pixel_ap | 0.4473 | 0.5491 |

## Ablation (TEST)

| variant   | description                                                 |   roc_auc |   pr_auc |     f1 |   balanced_accuracy |   recall |    fpr |    fnr |   fp |   fn |   cost_per_image |   dice |    iou |   pixel_auroc |
|:----------|:------------------------------------------------------------|----------:|---------:|-------:|--------------------:|---------:|-------:|-------:|-----:|-----:|-----------------:|-------:|-------:|--------------:|
| A         | Phase 1: global reconstruction error                        |         1 |        1 | 0.9995 |              0.9667 |   1      | 0.0667 | 0      |    3 |    0 |           0.001  | 0.4073 | 0.2643 |        0.9205 |
| B         | Global + local evidence (fused)                             |         1 |        1 | 0.9995 |              0.9667 |   1      | 0.0667 | 0      |    3 |    0 |           0.001  | 0.5957 | 0.4426 |        0.9769 |
| C         | Global + local + conformal calibration                      |         1 |        1 | 0.9995 |              0.9667 |   1      | 0.0667 | 0      |    3 |    0 |           0.001  | 0.5957 | 0.4426 |        0.9769 |
| D         | Global + local + conformal + cost-aware alpha* (RCC-ConvAE) |         1 |        1 | 0.9995 |              0.9667 |   1      | 0.0667 | 0      |    3 |    0 |           0.001  | 0.5957 | 0.4426 |        0.9769 |
| E         | Local evidence only (+ conformal + cost-aware alpha)        |         1 |        1 | 0.9997 |              0.9887 |   0.9997 | 0.0222 | 0.0003 |    1 |    1 |           0.0035 | 0.5957 | 0.4426 |        0.9769 |

## Distribution drift summary

| policy              |   clean_roc_auc |   clean_balanced_accuracy |   mean_roc_auc |   mean_balanced_accuracy |   mean_fpr |   mean_fnr |   mean_cost_per_image |   worst_balanced_accuracy | worst_condition   |   mean_delta_roc_auc_vs_clean |   mean_delta_balanced_accuracy_vs_clean |
|:--------------------|----------------:|--------------------------:|---------------:|-------------------------:|-----------:|-----------:|----------------------:|--------------------------:|:------------------|------------------------------:|----------------------------------------:|
| Phase1-frozen       |               1 |                    0.9667 |         0.9225 |                   0.7198 |     0.5333 |     0.0271 |                0.2744 |                    0.5    | brightness+0.10   |                       -0.0775 |                                 -0.2469 |
| Phase1-recalibrated |               1 |                    0.9667 |         0.9225 |                   0.9259 |     0.0361 |     0.1121 |                1.1057 |                    0.5387 | brightness+0.10   |                       -0.0775 |                                 -0.0408 |
| Phase2-frozen       |               1 |                    0.9667 |         0.9475 |                   0.7135 |     0.5    |     0.073  |                0.7268 |                    0.5    | brightness+0.10   |                       -0.0525 |                                 -0.2532 |
| Phase2-recalibrated |               1 |                    0.9667 |         0.9475 |                   0.9013 |     0.0222 |     0.1752 |                1.7267 |                    0.5704 | brightness+0.10   |                       -0.0525 |                                 -0.0654 |

## Interpretation notes

- The base paper (Gorman et al., IEEE TSM 2023) reports AUC = 1.00 on 1-D batch-process data; it is methodological inspiration, not a directly comparable benchmark.
- RCC-ConvAE is a risk-calibrated, cost-aware decision framework on the same frozen ConvAE backbone (not a new CNN). Image-level parity with Phase 1 is reported as-is.
- TEST data was used only after every policy component had been frozen.