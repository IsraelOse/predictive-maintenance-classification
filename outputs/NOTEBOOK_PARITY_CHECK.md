# Notebook parity check

**DAMO510 Take-Home Assignment 2** — Israel Obokohwo, NF1041893, seed 93

`notebooks/DAMO510_A2_full_analysis.ipynb` against the twelve-script pipeline snapshot in `outputs/_reference_from_scripts/`.

The report was written against the script results, so the notebook is only correct if it reproduces them exactly. Every table, stored result and figure is compared below.

| | |
|---|---|
| Tables compared | 63, of which 63 match |
| Stored results compared | 12, of which 12 match |
| Figures compared | 25, of which 24 byte-identical |
| Verdict | **PARITY CONFIRMED** |

Wall-clock fields are excluded from the comparison: `fit_seconds`, `search_seconds` and the `mean_fit_time` column vary between runs on any machine and carry no analytical meaning. Nothing else is excluded.

## 1. The results the brief names explicitly

| Result | Source file | Scripts vs notebook |
|---|---|---|
| Model comparison table | `tbl32_model_comparison.csv` | Identical, 4 rows x 12 cols, every cell equal |
| Confusion matrix counts | `tbl32_model_comparison.csv` | Identical, 4 rows x 12 cols, every cell equal |
| Threshold sweep | `tbl36_threshold_sweep.csv` | Identical, 11 rows x 10 cols, every cell equal |
| Out-of-fold threshold sweep | `tbl57_oof_threshold_sweep.csv` | Identical, 99 rows x 10 cols, every cell equal |
| Threshold selection comparison | `tbl58_threshold_selection_comparison.csv` | Identical, 3 rows x 7 cols, every cell equal |
| Feature importance ranks | `tbl49_importance_ranks.csv` | Identical, 10 rows x 5 cols, every cell equal |
| Permutation importance | `tbl48_permutation_importance.csv` | Identical, 9 rows x 5 cols, every cell equal |
| Five false positives | `tbl41_false_positive_cases.csv` | Identical, 5 rows x 14 cols, every cell equal |
| Five false negatives | `tbl42_false_negative_cases.csv` | Identical, 5 rows x 14 cols, every cell equal |
| All ten misclassified rows | `tbl45_misclassified_cases_combined.csv` | Identical, 10 rows x 15 cols, every cell equal |
| Selected configurations | `tbl27_selected_configurations.csv` | Identical, 4 rows x 4 cols, every cell equal |
| Imbalance study | `tbl29_imbalance_study_full.csv` | Identical, 16 rows x 14 cols, every cell equal |
| Baselines | `tbl24_baseline_results.csv` | Identical, 5 rows x 11 cols, every cell equal |
| Robustness by seed | `tbl53_robustness_seed_summary.csv` | Identical, 4 rows x 10 cols, every cell equal |
| Feature ablation | `tbl56_ablation_summary.csv` | Identical, 4 rows x 11 cols, every cell equal |

## 2. Key values, side by side

| Value | Scripts | Notebook | |
|---|---|---|---|
| Logistic Regression test PR-AUC | 0.5408 | 0.5408 | match |
| Logistic Regression confusion TN/FP/FN/TP | 1929/3/53/15 | 1929/3/53/15 | match |
| Decision Tree test PR-AUC | 0.8849 | 0.8849 | match |
| Decision Tree confusion TN/FP/FN/TP | 1932/0/15/53 | 1932/0/15/53 | match |
| Random Forest test PR-AUC | 0.9048 | 0.9048 | match |
| Random Forest confusion TN/FP/FN/TP | 1931/1/15/53 | 1931/1/15/53 | match |
| XGBoost test PR-AUC | 0.8734 | 0.8734 | match |
| XGBoost confusion TN/FP/FN/TP | 1925/7/17/51 | 1925/7/17/51 | match |
| Recommended threshold | 0.23 | 0.23 | match |
| Alarms at threshold | 63 | 63 | match |
| True positives at threshold | 57 | 57 | match |
| False alarms at threshold | 6 | 6 | match |
| Missed at threshold | 11 | 11 | match |
| Unflagged FN overlap (UDI) | 1438, 8507, 5537 | 1438, 8507, 5537 | match |
| Robustness seeds won | 10 | 10 | match |
| Robustness mean PR-AUC | 0.9034 | 0.9034 | match |
| Ablation winner without features | XGBoost | XGBoost | match |

## 3. Every table

| Table | Result |
|---|---|
| `cv_dt_full_results.csv` | identical (140 x 7) |
| `cv_lr_full_results.csv` | identical (16 x 6) |
| `cv_rf_full_results.csv` | identical (80 x 8) |
| `cv_xgb_full_results.csv` | identical (80 x 11) |
| `tbl01_variable_dictionary.csv` | identical (14 x 7) |
| `tbl02_numeric_quality.csv` | identical (5 x 13) |
| `tbl03_failure_flag_behaviour.csv` | identical (5 x 6) |
| `tbl04_flag_combination_crosstab.csv` | identical (12 x 8) |
| `tbl05_leakage_summary.csv` | identical (7 x 2) |
| `tbl06_inclusion_decisions.csv` | identical (13 x 3) |
| `tbl07_univariate_separation.csv` | identical (8 x 8) |
| `tbl08_correlation_matrix.csv` | identical (6 x 7) |
| `tbl09_multicollinearity_vif.csv` | identical (5 x 3) |
| `tbl10_failure_rate_by_power_decile.csv` | identical (10 x 6) |
| `tbl11_heat_dissipation_interaction.csv` | identical (4 x 4) |
| `tbl12_failure_rate_by_type.csv` | identical (3 x 5) |
| `tbl13_overstrain_threshold_by_type.csv` | identical (3 x 8) |
| `tbl14_failure_rate_by_wear_bin.csv` | identical (7 x 4) |
| `tbl15_descriptives_by_class.csv` | identical (24 x 4) |
| `tbl16_train_test_split.csv` | identical (3 x 4) |
| `tbl17_preprocessing_plan.csv` | identical (7 x 4) |
| `tbl18_engineered_feature_definitions.csv` | identical (3 x 4) |
| `tbl19_feature_evidence_separation.csv` | identical (8 x 9) |
| `tbl20_feature_evidence_rf_importance.csv` | identical (10 x 4) |
| `tbl21_feature_ablation.csv` | identical (9 x 8) |
| `tbl22_leave_one_out_cost.csv` | identical (3 x 5) |
| `tbl23_feature_decision.csv` | identical (3 x 4) |
| `tbl24_baseline_results.csv` | identical (5 x 11) |
| `tbl25_default_model_cv.csv` | identical (4 x 13) |
| `tbl26_tuned_model_cv.csv` | identical (4 x 13) |
| `tbl27_selected_configurations.csv` | identical (4 x 4) |
| `tbl28_default_vs_tuned.csv` | identical (4 x 6) |
| `tbl29_imbalance_study_full.csv` | identical (16 x 14) |
| `tbl30_imbalance_tradeoff.csv` | identical (12 x 7) |
| `tbl31_imbalance_retention_decision.csv` | identical (4 x 8) |
| `tbl32_model_comparison.csv` | identical (4 x 12) |
| `tbl33_final_model_specifications.csv` | identical (4 x 3) |
| `tbl34_strongest_weakest_by_criterion.csv` | identical (6 x 5) |
| `tbl35_probability_granularity.csv` | identical (4 x 3) |
| `tbl36_threshold_sweep.csv` | identical (11 x 10) |
| `tbl37_threshold_fine_sweep.csv` | identical (99 x 10) |
| `tbl38_cost_ratio_sensitivity.csv` | identical (7 x 6) |
| `tbl39_threshold_regret.csv` | identical (6 x 7) |
| `tbl40_test_class_medians.csv` | identical (8 x 3) |
| `tbl41_false_positive_cases.csv` | identical (5 x 14) |
| `tbl42_false_negative_cases.csv` | identical (5 x 14) |
| `tbl43_false_positive_deviation_from_class_median.csv` | identical (5 x 10) |
| `tbl44_false_negative_deviation_from_class_median.csv` | identical (5 x 10) |
| `tbl45_misclassified_cases_combined.csv` | identical (10 x 15) |
| `tbl46_logistic_coefficients.csv` | identical (10 x 4) |
| `tbl47_native_importance_normalised.csv` | identical (10 x 5) |
| `tbl48_permutation_importance.csv` | identical (9 x 5) |
| `tbl49_importance_ranks.csv` | identical (10 x 5) |
| `tbl50_importance_rank_agreement.csv` | identical (6 x 4) |
| `tbl51_importance_consistency.csv` | identical (10 x 5) |
| `tbl52_robustness_seed_results.csv` | identical (40 x 14) |
| `tbl53_robustness_seed_summary.csv` | identical (4 x 10) |
| `tbl54_robustness_win_counts.csv` | identical (1 x 2) |
| `tbl55_ablation_full.csv` | identical (8 x 12) |
| `tbl56_ablation_summary.csv` | identical (4 x 11) |
| `tbl57_oof_threshold_sweep.csv` | identical (99 x 10) |
| `tbl58_threshold_selection_comparison.csv` | identical (3 x 7) |
| `test_predicted_probabilities.csv` | identical (2000 x 5) |

## 4. Stored results

| File | Result |
|---|---|
| `audit.json` | identical |
| `baseline.json` | identical |
| `eda.json` | identical |
| `evaluation.json` | identical |
| `features.json` | identical |
| `imbalance.json` | identical |
| `interpretation.json` | identical |
| `misclassification.json` | identical |
| `robustness.json` | identical |
| `split.json` | identical |
| `threshold.json` | identical |
| `tuning.json` | identical |

## 5. Figures

All 25 figures are regenerated by the notebook from the same data with the same styling, and compared byte for byte.

`fig25_robustness.png` is the one figure whose pixels differ, by 0.17 per cent, all of it inside the left panel. The cause is `sns.stripplot`, which jitters its points horizontally using the global NumPy random state. That state is not the same in a notebook process as in a script process, because a different amount of randomness has been drawn before the plot is made. Verified by rendering the same stripplot under two NumPy seeds and getting two different images, then the same seed twice and getting the same image.

Nothing plotted moved. The box plots, the starred seed-93 markers and the axis limits are unchanged, and the underlying data in `tbl52_robustness_seed_results.csv` and `tbl53_robustness_seed_summary.csv` is identical. Only the horizontal scatter of the overlaid dots is drawn differently. I left the plotting call alone rather than seeding it, because that call is the one that produced Figure A25 in the report.

| Figure | Result |
|---|---|
| `fig01_target_distribution.png` | byte-identical |
| `fig02_conditional_distributions.png` | byte-identical |
| `fig03_correlation_heatmap.png` | byte-identical |
| `fig04_torque_vs_speed.png` | byte-identical |
| `fig05_failure_rate_by_power_decile.png` | byte-identical |
| `fig06_tempgap_vs_speed.png` | byte-identical |
| `fig07_wear_torque_by_type.png` | byte-identical |
| `fig08_wear_bands_and_type.png` | byte-identical |
| `fig09_engineered_feature_separation.png` | byte-identical |
| `fig10_feature_ablation.png` | byte-identical |
| `fig11_baseline_comparison.png` | byte-identical |
| `fig12_tuning_summary.png` | byte-identical |
| `fig13_imbalance_recall_precision.png` | byte-identical |
| `fig14_imbalance_tradeoff_scatter.png` | byte-identical |
| `fig15_confusion_matrices.png` | byte-identical |
| `fig16_roc_and_pr_curves.png` | byte-identical |
| `fig17_metric_comparison.png` | byte-identical |
| `fig18_threshold_curves.png` | byte-identical |
| `fig19_threshold_errors_and_sensitivity.png` | byte-identical |
| `fig20_misclassification_cases.png` | byte-identical |
| `fig21_misclassification_deviation.png` | byte-identical |
| `fig22_importance_comparison.png` | byte-identical |
| `fig23_importance_rank_heatmap.png` | byte-identical |
| `fig24_logistic_coefficients.png` | byte-identical |
| `fig25_robustness.png` | regenerated, 191559 vs 191450 bytes (0.06%) |

## 6. What differs by design

- **Console logs.** The scripts wrote `outputs/logs/*.log` through a stdout tee. The notebook keeps its output in the cells themselves, so the log files are left as the scripts wrote them rather than being overwritten.

- **Wall-clock timings.** `fit_seconds`, `search_seconds` and `mean_fit_time` differ between any two runs and are excluded from the comparison.

- **Where data comes from.** The scripts passed frames between stages through CSVs; the notebook passes the objects already in memory. The values are identical, which is what sections 1 to 4 above demonstrate.
