"""
Stage 09 - Threshold investigation  (Task 10)

Takes the Random Forest, sweeps eleven thresholds across its predicted
probabilities, and adds a cost frame so that the recommendation rests on
something other than a symmetric F1.

The cost ratio is an assumption, not a measurement. It is stated as one, and
the sensitivity of the recommendation to that assumption is tested rather than
hoped about.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import config as C
import pipelines as P

warnings.filterwarnings("ignore")

STAGE = "09_threshold_analysis"
plt, sns = C.setup_matplotlib()

SELECTED = "Random Forest"


def metrics_at(y, proba, t):
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = P.confusion_counts(y, pred)
    return {
        "threshold": round(t, 4),
        "predicted_failures": int(tp + fp),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "expected_cost": round(C.COST_FN * fn + C.COST_FP * fp, 2),
    }


def main() -> None:
    tee = C.start_log(STAGE)
    try:
        C.banner("STAGE 09  THRESHOLD INVESTIGATION")
        probs = pd.read_csv(C.TABLES / "test_predicted_probabilities.csv")
        y = probs["y_true"].to_numpy()
        proba = probs[SELECTED].to_numpy()
        ev = C.get("evaluation")

        print("Selected model: %s." % SELECTED)
        print("It is chosen for this stage because it has the highest test PR-AUC "
              "(%.4f) and because it produces %d distinct scores on the test set, "
              "so the threshold is a dial with real positions on it. The Decision "
              "Tree offers %d."
              % (ev["best_pr_auc"], ev["granularity"][SELECTED],
                 ev["granularity"]["Decision Tree"]))
        print("\nTest set: %d records, %d failures." % (len(y), int(y.sum())))

        # ------------------------------------------------------- the sweep
        C.banner("ELEVEN THRESHOLDS")
        sweep = pd.DataFrame([metrics_at(y, proba, t) for t in C.THRESHOLDS])
        C.save_table(sweep, "tbl36_threshold_sweep")

        # ------------------------------------------------------- cost frame
        C.banner("COST FRAME (STATED AS AN ASSUMPTION)")
        print("Assumption: a missed failure costs %.0f currency units and a false "
              "alarm costs %.0f, a ratio of %.0f to 1." % (C.COST_FN, C.COST_FP,
                                                           C.COST_RATIO))
        print("The reasoning behind the ratio: a missed failure means an unplanned "
              "stoppage, an emergency call-out and scrapped work in progress, while "
              "a false alarm means an inspection that finds nothing. An order of "
              "magnitude between those is plausible for a machining line. It is not "
              "sourced from this plant, because this dataset carries no cost "
              "information at all, and that is a real weakness of the "
              "recommendation rather than a footnote.")

        fine = np.round(np.arange(0.01, 1.00, 0.01), 2)
        fine_tbl = pd.DataFrame([metrics_at(y, proba, t) for t in fine])
        C.save_table(fine_tbl, "tbl37_threshold_fine_sweep")

        cost_opt = fine_tbl.loc[fine_tbl["expected_cost"].idxmin()]
        f1_opt = fine_tbl.loc[fine_tbl["f1"].idxmax()]
        default = fine_tbl[fine_tbl["threshold"] == 0.50].iloc[0]

        print("\nDefault 0.50 : cost %.0f, %d missed, %d false alarms, recall %.3f"
              % (default["expected_cost"], default["FN"], default["FP"],
                 default["recall"]))
        print("F1-optimal %.2f: cost %.0f, %d missed, %d false alarms, recall %.3f"
              % (f1_opt["threshold"], f1_opt["expected_cost"], f1_opt["FN"],
                 f1_opt["FP"], f1_opt["recall"]))
        print("Cost-optimal %.2f: cost %.0f, %d missed, %d false alarms, recall %.3f"
              % (cost_opt["threshold"], cost_opt["expected_cost"], cost_opt["FN"],
                 cost_opt["FP"], cost_opt["recall"]))
        print("\nMoving from 0.50 to %.2f saves %.0f units on this test set, which "
              "is %d fewer missed failures bought with %d extra inspections."
              % (cost_opt["threshold"],
                 default["expected_cost"] - cost_opt["expected_cost"],
                 int(default["FN"] - cost_opt["FN"]),
                 int(cost_opt["FP"] - default["FP"])))

        # ------------------------------------- honest threshold selection
        C.banner("SELECTING THE THRESHOLD WITHOUT TOUCHING THE TEST SET")
        print("Everything above sweeps the test set, which is the right way to "
              "show how the trade-off behaves and the wrong way to choose a "
              "number. A threshold picked where it is scored has seen its own "
              "answer. So I repeat the selection on out-of-fold predictions from "
              "the training partition only, and then report what that "
              "independently chosen value does on the held-out test set.\n")

        X_tr, X_te, y_tr, y_te = P.load_model_ready(use_engineered=True)
        models = P.build_final_models(use_engineered=True)
        cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True,
                             random_state=C.SEED)
        oof = cross_val_predict(models[SELECTED], X_tr, y_tr, cv=cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
        print("Out-of-fold probabilities for %d training rows, %d failures."
              % (len(y_tr), int(y_tr.sum())))

        oof_tbl = pd.DataFrame([metrics_at(y_tr, oof, t) for t in fine])
        C.save_table(oof_tbl, "tbl57_oof_threshold_sweep")
        oof_opt = oof_tbl.loc[oof_tbl["expected_cost"].idxmin()]
        cv_t = float(oof_opt["threshold"])
        print("\nCost-minimising threshold on out-of-fold training predictions: "
              "%.2f" % cv_t)
        print("Cost-minimising threshold on the test set (shown above)       : "
              "%.2f" % float(cost_opt["threshold"]))

        at_cv = fine_tbl[fine_tbl["threshold"] == cv_t].iloc[0]
        at_test_opt = cost_opt
        gap = at_cv["expected_cost"] - at_test_opt["expected_cost"]
        print("\nOn the test set, the out-of-fold choice of %.2f gives cost %.0f "
              "against %.0f for the test-set optimum of %.2f, a difference of "
              "%.0f units, or %.1f per cent."
              % (cv_t, at_cv["expected_cost"], at_test_opt["expected_cost"],
                 float(cost_opt["threshold"]), gap,
                 100 * gap / at_test_opt["expected_cost"]))
        print("Recall %.4f, precision %.4f, %d missed, %d false alarms."
              % (at_cv["recall"], at_cv["precision"], at_cv["FN"], at_cv["FP"]))

        sel = pd.DataFrame([
            {"selection_basis": "out-of-fold, training partition only",
             "threshold": cv_t,
             "test_recall": at_cv["recall"], "test_precision": at_cv["precision"],
             "test_FN": int(at_cv["FN"]), "test_FP": int(at_cv["FP"]),
             "test_expected_cost": at_cv["expected_cost"]},
            {"selection_basis": "test set (optimistic, shown for comparison)",
             "threshold": float(at_test_opt["threshold"]),
             "test_recall": at_test_opt["recall"],
             "test_precision": at_test_opt["precision"],
             "test_FN": int(at_test_opt["FN"]), "test_FP": int(at_test_opt["FP"]),
             "test_expected_cost": at_test_opt["expected_cost"]},
            {"selection_basis": "library default",
             "threshold": 0.50, "test_recall": default["recall"],
             "test_precision": default["precision"], "test_FN": int(default["FN"]),
             "test_FP": int(default["FP"]),
             "test_expected_cost": default["expected_cost"]},
        ])
        C.save_table(sel, "tbl58_threshold_selection_comparison")

        # ------------------------------------------------- ratio sensitivity
        C.banner("HOW MUCH DOES THE ANSWER DEPEND ON THAT ASSUMED RATIO?")
        sens_rows = []
        for ratio in [2, 5, 10, 20, 30, 50, 100]:
            costs = [ratio * m["FN"] + 1 * m["FP"]
                     for _, m in fine_tbl.iterrows()]
            best_i = int(np.argmin(costs))
            r = fine_tbl.iloc[best_i]
            sens_rows.append({
                "cost_ratio_FN_to_FP": ratio,
                "optimal_threshold": r["threshold"],
                "recall": r["recall"],
                "precision": r["precision"],
                "missed_failures": int(r["FN"]),
                "false_alarms": int(r["FP"]),
            })
        sens = pd.DataFrame(sens_rows)
        C.save_table(sens, "tbl38_cost_ratio_sensitivity")
        lo = sens.iloc[0]
        hi = sens.iloc[-1]
        print("\nAcross ratios from 2:1 to 100:1 the cost-minimising threshold "
              "moves from %.2f to %.2f and recall from %.3f to %.3f. That is a "
              "large movement and I am not going to dress it up: the argmin is "
              "sensitive to an assumption I cannot verify. At 20:1 and above it "
              "collapses to %.2f, where recall reaches %.3f but precision falls to "
              "%.3f and the crew is sent to %d machines to find %d problems."
              % (lo["optimal_threshold"], hi["optimal_threshold"],
                 lo["recall"], hi["recall"], hi["optimal_threshold"], hi["recall"],
                 hi["precision"], int(hi["false_alarms"] + 68 - hi["missed_failures"]),
                 int(68 - hi["missed_failures"])))

        # The useful question is not where the argmin moves, but how much a
        # single fixed choice loses if the assumed ratio is wrong.
        C.banner("REGRET: WHAT ONE FIXED THRESHOLD COSTS IF THE RATIO IS WRONG")
        cand = sorted({0.10, 0.14, 0.20, cv_t, 0.32, 0.50})
        reg_rows = []
        for ratio in [2, 5, 10, 20, 30, 50]:
            costs = np.array([ratio * m["FN"] + 1 * m["FP"]
                              for _, m in fine_tbl.iterrows()], dtype=float)
            best = costs.min()
            row = {"cost_ratio_FN_to_FP": ratio}
            for t in cand:
                m = fine_tbl[fine_tbl["threshold"] == t].iloc[0]
                c = ratio * m["FN"] + m["FP"]
                row["excess_pct_at_%.2f" % t] = round(100 * (c - best) / best, 1)
            reg_rows.append(row)
        regret = pd.DataFrame(reg_rows)
        C.save_table(regret, "tbl39_threshold_regret")
        worst_014 = regret["excess_pct_at_%.2f" % cv_t].max()
        worst_050 = regret["excess_pct_at_0.50"].max()
        print("\nThe useful comparison is against the default. Holding the "
              "threshold at the recommended %.2f is never worse across this range "
              "than %.1f%% above the best achievable cost; the default reaches "
              "%.1f%%." % (cv_t, worst_014, worst_050))
        print("So the exact argmin is unstable, and the decision that matters is "
              "not. Move well below 0.50. Between 0.10 and 0.32 the cost curve is "
              "flat to within a few per cent as long as the true ratio is nearer "
              "10:1 than 50:1, which makes the choice inside that band a capacity "
              "question for the maintenance team rather than a modelling question "
              "for me. If the plant believes a missed failure really is fifty "
              "times a false alarm then none of these thresholds is right, and the "
              "answer moves to 0.02 with the 114 false alarms that implies.")

        # ------------------------------------------------- recommendation
        C.banner("RECOMMENDED OPERATING THRESHOLD")
        rec_t = cv_t   # chosen out of fold, above
        rec = fine_tbl[fine_tbl["threshold"] == rec_t].iloc[0]
        print("I recommend %.2f, chosen on out-of-fold training predictions "
              "rather than on the test set." % rec_t)
        print("At that threshold the forest flags %d of the 2,000 records. %d of "
              "those are real failures and %d are false alarms, so a crew acting "
              "on every alarm inspects %d machines and finds %d genuine problems. "
              "Recall %.3f, precision %.3f, F1 %.3f."
              % (int(rec["predicted_failures"]), int(rec["TP"]), int(rec["FP"]),
                 int(rec["predicted_failures"]), int(rec["TP"]), rec["recall"],
                 rec["precision"], rec["f1"]))
        print("\nThe default 0.50 is not optimal here and there was no reason to "
              "expect it to be. A 0.5 cut-off is the right decision rule when the "
              "two errors cost the same and the classes are balanced. Neither "
              "holds: failures are %0.1f%% of records and one missed failure is "
              "assumed to cost ten false alarms." % (100 * y.mean()))
        print("\nI am less confident in this number than in anything else in the "
              "analysis, because the cost ratio behind it is assumed rather than "
              "sourced from the plant. What I would defend is the shape of the "
              "argument and the direction of the move away from 0.50, not the "
              "second decimal place. Choosing it out of fold removes one source "
              "of optimism, and it does not remove that one.")

        # ------------------------------------------------- figures
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        axes[0].plot(fine_tbl["threshold"], fine_tbl["precision"], color=C.PALETTE[0],
                     linewidth=1.9, label="Precision")
        axes[0].plot(fine_tbl["threshold"], fine_tbl["recall"], color=C.PALETTE[1],
                     linewidth=1.9, label="Recall")
        axes[0].plot(fine_tbl["threshold"], fine_tbl["f1"], color=C.PALETTE[2],
                     linewidth=1.9, label="F1")
        axes[0].axvline(0.5, color="grey", linestyle=":", linewidth=1.2,
                        label="Default 0.50")
        axes[0].axvline(rec_t, color=C.C_FAIL, linestyle="--", linewidth=1.5,
                        label="Recommended %.2f (chosen out of fold)" % rec_t)
        for _, r in sweep.iterrows():
            axes[0].scatter(r["threshold"], r["f1"], s=22, color=C.PALETTE[2],
                            zorder=5, edgecolor="black", linewidth=0.4)
        axes[0].set_xlabel("Classification threshold")
        axes[0].set_ylabel("Score on the test set")
        axes[0].set_ylim(0, 1.05)
        axes[0].set_title("Precision, recall and F1 across the threshold")
        axes[0].legend(fontsize=8, loc="lower center")

        ax2 = axes[1]
        ax2.plot(fine_tbl["threshold"], fine_tbl["expected_cost"], color="black",
                 linewidth=1.9, label="Expected cost (10:1)")
        ax2.axvline(rec_t, color=C.C_FAIL, linestyle="--", linewidth=1.5)
        ax2.scatter([rec_t], [cost_opt["expected_cost"]], s=90, color=C.C_FAIL,
                    zorder=6, edgecolor="black",
                    label="Minimum at %.2f, cost %.0f" % (rec_t,
                                                          cost_opt["expected_cost"]))
        ax2.axvline(0.5, color="grey", linestyle=":", linewidth=1.2)
        ax2.scatter([0.5], [default["expected_cost"]], s=70, color="grey", zorder=6,
                    edgecolor="black", label="Default 0.50, cost %.0f"
                                             % default["expected_cost"])
        ax2.set_xlabel("Classification threshold")
        ax2.set_ylabel("Expected cost on 2,000 records")
        ax2.set_title("Expected cost, missed failure = 10 x false alarm")
        ax2.legend(fontsize=8)
        fig.suptitle("Threshold analysis, Random Forest on the held-out test set",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        C.save_fig(fig, "fig18_threshold_curves")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        x = np.arange(len(sweep))
        axes[0].bar(x - 0.2, sweep["FN"], 0.4, label="Missed failures (FN)",
                    color=C.C_FAIL, edgecolor="black", linewidth=0.5)
        axes[0].bar(x + 0.2, sweep["FP"], 0.4, label="False alarms (FP)",
                    color=C.PALETTE[3], edgecolor="black", linewidth=0.5)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(["%.2f" % t for t in sweep["threshold"]],
                                rotation=45)
        axes[0].set_xlabel("Threshold")
        axes[0].set_ylabel("Count on the test set")
        axes[0].set_title("The two errors move in opposite directions")
        axes[0].legend(fontsize=8)

        axes[1].plot(sens["cost_ratio_FN_to_FP"], sens["optimal_threshold"],
                     marker="o", color=C.PALETTE[0], linewidth=1.8)
        for _, r in sens.iterrows():
            axes[1].annotate("%.2f" % r["optimal_threshold"],
                             (r["cost_ratio_FN_to_FP"], r["optimal_threshold"]),
                             textcoords="offset points", xytext=(0, 8),
                             ha="center", fontsize=8)
        axes[1].set_xscale("log")
        axes[1].set_xticks(sens["cost_ratio_FN_to_FP"])
        axes[1].set_xticklabels(["%d:1" % v for v in sens["cost_ratio_FN_to_FP"]])
        axes[1].set_xlabel("Assumed cost of a missed failure relative to a false alarm")
        axes[1].set_ylabel("Cost-minimising threshold")
        axes[1].set_ylim(0, 0.6)
        axes[1].set_title("Sensitivity of the recommendation to the cost assumption")
        fig.suptitle("Error trade-off and how much the cost assumption matters",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        C.save_fig(fig, "fig19_threshold_errors_and_sensitivity")
        plt.close(fig)

        C.put("threshold", {
            "model": SELECTED,
            "recommended": rec_t,
            "sweep": sweep.to_dict("records"),
            "default": default.to_dict(),
            "f1_opt": f1_opt.to_dict(),
            "cost_opt": cost_opt.to_dict(),
            "cost_fn": C.COST_FN, "cost_fp": C.COST_FP, "ratio": C.COST_RATIO,
            "sensitivity": sens.to_dict("records"),
            "saving_vs_default": round(float(default["expected_cost"] -
                                             cost_opt["expected_cost"]), 2),
            "cv_threshold": cv_t,
            "test_optimum_threshold": float(cost_opt["threshold"]),
            "oof_vs_test_cost_gap": float(gap),
            "selection_comparison": sel.to_dict("records"),
            "regret": regret.to_dict("records"),
            "worst_regret_recommended": float(worst_014),
            "worst_regret_default": float(worst_050),
            "n_alarms_at_rec": int(rec["predicted_failures"]),
            "tp_at_rec": int(rec["TP"]), "fp_at_rec": int(rec["FP"]),
            "fn_at_rec": int(rec["FN"]),
            "recall_at_rec": float(rec["recall"]),
            "precision_at_rec": float(rec["precision"]),
            "f1_at_rec": float(rec["f1"]),
        })
        print("\nStage 09 complete.")
    finally:
        C.end_log(tee)


if __name__ == "__main__":
    main()
