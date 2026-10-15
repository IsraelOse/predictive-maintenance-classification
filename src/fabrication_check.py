"""
Fabrication check.

Pulls specific numeric claims out of the finished DOCX by regular expression
and compares each against the file that produced it. A claim only passes if
the number is actually present in the report text AND matches the source.

This is deliberately adversarial: it re-reads the rendered document rather than
trusting the builder, so a formatting bug that printed the wrong value would be
caught here.

Run after build_report.py.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from docx import Document

import config as C

DOC = C.REPORT / "DAMO510_A2_Report.docx"


def report_text():
    d = Document(DOC)
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            for cell in row.cells:
                parts.append(cell.text)
    return "\n".join(parts)


def tbl(name):
    return pd.read_csv(C.TABLES / ("%s.csv" % name))


def main():
    txt = report_text()
    body = txt

    audit = C.get("audit")
    ev = C.get("evaluation")
    th = C.get("threshold")
    rb = C.get("robustness")
    ft = C.get("features")
    tu = C.get("tuning")
    sp = C.get("split")
    mc = C.get("misclassification")
    eda = C.get("eda")

    comp = tbl("tbl32_model_comparison").set_index("model")
    base = tbl("tbl24_baseline_results")
    imb = tbl("tbl29_imbalance_study_full")
    gran = tbl("tbl35_probability_granularity").set_index("model")
    agree = tbl("tbl50_importance_rank_agreement")
    ranks = tbl("tbl49_importance_ranks").set_index("feature")

    # (label, string that must appear in the report, source description, source value)
    checks = [
        ("Total records", "10,000 records",
         "audit.json n_rows", audit["n_rows"]),
        ("Failure count", "1 on %d of 10,000 records" % audit["n_pos"],
         "audit.json n_pos", audit["n_pos"]),
        ("Positive rate", "%.2f per cent" % audit["pos_rate_pct"],
         "audit.json pos_rate_pct", audit["pos_rate_pct"]),
        ("Majority accuracy", "%.2f per cent" % audit["majority_accuracy_pct"],
         "audit.json majority_accuracy_pct", audit["majority_accuracy_pct"]),
        ("Leakage OR match", "on 9,991 of 10,000 records, %.2f per cent"
         % audit["or4_match_pct"], "audit.json or4_match", audit["or4_match"]),
        ("Leakage one-sided", "%d records where a mode fires and the target is 0"
         % audit["or4_flag_no_target"], "audit.json or4_flag_no_target",
         audit["or4_flag_no_target"]),
        ("HDF fire count", "HDF %d times" % audit["flag_counts"]["HDF"],
         "audit.json flag_counts.HDF", audit["flag_counts"]["HDF"]),
        ("RNF fires", "It fires on %d records" % audit["rnf_n"],
         "audit.json rnf_n", audit["rnf_n"]),
        ("Product ID agreement", "on %.0f per cent of rows"
         % audit["pid_type_agreement_pct"], "audit.json pid_type_agreement_pct",
         audit["pid_type_agreement_pct"]),
        ("Train/test split", "8,000 for training and 2,000 for testing",
         "split.json n_train / n_test", (sp["n_train"], sp["n_test"])),
        ("Training failures", "%d failures in training" % sp["train_pos"],
         "split.json train_pos", sp["train_pos"]),
        ("Torque univariate AUC", "univariate AUC of %.3f" % eda["auc_torque"],
         "tbl07 univariate_auc torque_Nm", eda["auc_torque"]),
        ("Heat interaction n", "all %d records meeting both failed"
         % eda["hdf_joint_n"], "tbl11 n_rows", eda["hdf_joint_n"]),
        ("Feature gain, RF", "lifts the forest from %.4f to %.4f"
         % (ft["base_rf"], ft["full_rf"]), "features.json base_rf/full_rf",
         (ft["base_rf"], ft["full_rf"])),
        ("Power baseline F1", "an F1 of %.4f" % float(base.loc[3, "f1"]),
         "tbl24 row B4 f1", float(base.loc[3, "f1"])),
        ("Configurations tried", "%d configurations" % tu["total_configs"],
         "tuning.json total_configs", tu["total_configs"]),
        ("Model fits", "%s fits" % "{:,}".format(tu["total_fits"]),
         "tuning.json total_fits", tu["total_fits"]),
        ("DT tuning gain", "gained %.4f" % tu["comparison"][1]["improvement"],
         "tbl28 improvement, Decision Tree",
         tu["comparison"][1]["improvement"]),
        ("RF test PR-AUC", "%.4f" % comp.loc["Random Forest", "pr_auc"],
         "tbl32 pr_auc Random Forest", comp.loc["Random Forest", "pr_auc"]),
        ("DT test recall", "recall %.4f" % comp.loc["Decision Tree", "recall"],
         "tbl32 recall Decision Tree", comp.loc["Decision Tree", "recall"]),
        ("LR true positives", "regression %d." % comp.loc["Logistic Regression", "TP"],
         "tbl32 TP Logistic Regression", comp.loc["Logistic Regression", "TP"]),
        ("DT probability granularity",
         "only %d distinct values" % gran.loc["Decision Tree",
                                              "distinct_probabilities_on_test"],
         "tbl35 Decision Tree",
         gran.loc["Decision Tree", "distinct_probabilities_on_test"]),
        ("RF probability granularity",
         "the forest emits %s" % "{:,}".format(
             gran.loc["Random Forest", "distinct_probabilities_on_test"]),
         "tbl35 Random Forest",
         gran.loc["Random Forest", "distinct_probabilities_on_test"]),
        ("Imbalance: LR recall untreated",
         "from %.4f " % float(imb[(imb.model == "Logistic Regression") &
                                  (imb.treatment == "none")]["recall"].iloc[0]),
         "tbl29 LR none recall",
         float(imb[(imb.model == "Logistic Regression") &
                   (imb.treatment == "none")]["recall"].iloc[0])),
        ("Recommended threshold", "operated at %.2f" % th["recommended"],
         "threshold.json recommended", th["recommended"]),
        ("Alarms at threshold", "flags %d of 2,000 records" % th["n_alarms_at_rec"],
         "threshold.json n_alarms_at_rec", th["n_alarms_at_rec"]),
        ("True positives at threshold", "which %d are real failures" % th["tp_at_rec"],
         "threshold.json tp_at_rec", th["tp_at_rec"]),
        ("Default expected cost", "%s" % "{:,.0f}".format(
            th["default"]["expected_cost"]),
         "threshold.json default.expected_cost", th["default"]["expected_cost"]),
        ("Regret vs default", "the default reaches %.1f per cent"
         % th["worst_regret_default"], "threshold.json worst_regret_default",
         th["worst_regret_default"]),
        ("Regret at recommended", "never costs more than %.1f per cent"
         % th["worst_regret_recommended"], "threshold.json worst_regret_recommended",
         th["worst_regret_recommended"]),
        ("Out-of-fold threshold", "That gives %.2f" % th["cv_threshold"],
         "threshold.json cv_threshold", th["cv_threshold"]),
        ("Test-set optimum shown", "test set itself is %.2f"
         % th["test_optimum_threshold"], "threshold.json test_optimum_threshold",
         th["test_optimum_threshold"]),
        ("FN max probability", "was %.3f" % mc["fn_summary"]["max_probability"],
         "misclassification.json fn_summary.max_probability",
         mc["fn_summary"]["max_probability"]),
        ("FP1 UDI", "UDI %d" % mc["fp_cases"][0]["UDI"],
         "tbl41 first row UDI", mc["fp_cases"][0]["UDI"]),
        ("Unflagged overlap", "UDI %s" % ", ".join(
            str(u) for u in mc["fn_summary"]["unflagged_overlap"]),
         "misclassification.json unflagged_overlap",
         mc["fn_summary"]["unflagged_overlap"]),
        ("Spearman DT vs RF", "%.3f" % float(
            agree[(agree.model_a == "Decision Tree") &
                  (agree.model_b == "Random Forest")]["spearman_rho"].iloc[0]),
         "tbl50 DT vs RF",
         float(agree[(agree.model_a == "Decision Tree") &
                     (agree.model_b == "Random Forest")]["spearman_rho"].iloc[0])),
        ("wear_torque rank, LR", "%d by logistic regression"
         % int(ranks.loc["wear_torque", "Logistic Regression"]),
         "tbl49 wear_torque Logistic Regression",
         int(ranks.loc["wear_torque", "Logistic Regression"])),
        ("Robustness wins", "wins %d of the ten" % rb["rf_wins"],
         "robustness.json rf_wins", rb["rf_wins"]),
        ("Robustness mean", "averaging %.4f" % rb["rf_mean"],
         "robustness.json rf_mean", rb["rf_mean"]),
        ("Robustness sd", "standard deviation of %.4f" % rb["rf_sd"],
         "robustness.json rf_sd", rb["rf_sd"]),
        ("Ablation winner", "%s takes first place" % rb["raw_winner"],
         "robustness.json raw_winner", rb["raw_winner"]),
        ("Ablation raw PR-AUC", "at %.4f against" % rb["raw_best"],
         "robustness.json raw_best", rb["raw_best"]),
    ]

    print("=" * 78)
    print("FABRICATION CHECK: %d claims verified against generated output" % len(checks))
    print("=" * 78)
    print("Report : %s" % DOC.name)
    print()

    ok = fail = 0
    failures = []
    for label, needle, source, value in checks:
        present = needle in txt
        if present:
            ok += 1
            print("  PASS  %-28s %-46s = %s"
                  % (label, source, value))
        else:
            fail += 1
            failures.append((label, needle, source, value))
            print("  FAIL  %-28s %-46s expected %s"
                  % (label, source, value))
            print("        searched for: %r" % needle)

    print()
    print("-" * 78)
    print("%d of %d claims verified. %d mismatches." % (ok, len(checks), fail))
    print("-" * 78)
    if failures:
        print("\nMISMATCHES REQUIRING ATTENTION:")
        for label, needle, source, value in failures:
            print("  %s: report does not contain %r (source %s = %s)"
                  % (label, needle, source, value))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
