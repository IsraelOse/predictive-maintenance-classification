"""
Stage 08 - Common evaluation framework  (Task 9)

Fits each model in its final form (tuned parameters from stage 06, retained
imbalance treatment from stage 07) on the full 8,000-row training partition and
scores it once on the 2,000 held-out records. The test set has been untouched
until this point.

Produces the single consolidated comparison table, confusion matrices, ROC
curves and precision-recall curves, and stores the predicted probabilities that
stages 09 to 11 work from.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             precision_recall_curve, roc_auc_score, roc_curve)

import config as C
import pipelines as P

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

STAGE = "08_evaluation"
plt, sns = C.setup_matplotlib()


def main() -> None:
    tee = C.start_log(STAGE)
    try:
        C.banner("STAGE 08  MODEL EVALUATION ON THE HELD-OUT TEST SET")
        X_tr, X_te, y_tr, y_te = P.load_model_ready(use_engineered=True)
        imb = C.get("imbalance")
        tuning = C.get("tuning")["chosen"]
        print("Test set: %d records, %d failures (%.2f%%). Untouched until now."
              % (len(X_te), y_te.sum(), 100 * y_te.mean()))

        models = P.build_final_models(use_engineered=True)
        rows, probas, fitted = [], {}, {}

        for name in P.MODEL_ORDER:
            pipe = models[name]
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            r = P.score_row(name, y_te, pred, proba)
            r["imbalance_treatment"] = imb["retained"][name]
            r["parameters"] = ", ".join("%s=%s" % (k, v) for k, v in
                                        sorted(tuning[name]["params"].items()))
            rows.append(r)
            probas[name] = proba
            fitted[name] = pipe
            print("%-20s acc %.4f  prec %.4f  rec %.4f  F1 %.4f  ROC %.4f  PR %.4f"
                  % (name, r["accuracy"], r["precision"], r["recall"], r["f1"],
                     r["roc_auc"], r["pr_auc"]))

        comp = pd.DataFrame(rows)
        main_cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc",
                     "pr_auc", "TN", "FP", "FN", "TP", "imbalance_treatment"]
        C.save_table(comp[main_cols], "tbl32_model_comparison")
        C.save_table(comp[["model", "parameters", "imbalance_treatment"]],
                     "tbl33_final_model_specifications")

        # ------------------------------------------------------------------
        C.banner("WHICH MODEL IS BEST DEPENDS ON WHAT THE PLANT FEARS")
        by = {}
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            top = comp.loc[comp[metric].idxmax()]
            bot = comp.loc[comp[metric].idxmin()]
            by[metric] = {"best": str(top["model"]), "best_value": float(top[metric]),
                          "worst": str(bot["model"]), "worst_value": float(bot[metric])}
            print("%-10s best: %-20s %.4f     weakest: %-20s %.4f"
                  % (metric, top["model"], top[metric], bot["model"], bot[metric]))

        winners = pd.DataFrame([
            {"criterion": k, "strongest_model": v["best"], "value": v["best_value"],
             "weakest_model": v["worst"], "weakest_value": v["worst_value"]}
            for k, v in by.items()])
        C.save_table(winners, "tbl34_strongest_weakest_by_criterion")

        best_rec = comp.loc[comp["recall"].idxmax()]
        best_prec = comp.loc[comp["precision"].idxmax()]
        best_pr = comp.loc[comp["pr_auc"].idxmax()]

        # How many operating points does each model actually offer?
        gran = pd.DataFrame({
            "model": P.MODEL_ORDER,
            "distinct_probabilities_on_test": [int(pd.Series(probas[m]).nunique())
                                               for m in P.MODEL_ORDER],
        })
        gran["available_operating_points"] = gran["distinct_probabilities_on_test"]
        C.save_table(gran, "tbl35_probability_granularity")

        dt = comp[comp["model"] == "Decision Tree"].iloc[0]
        rf = comp[comp["model"] == "Random Forest"].iloc[0]
        lr = comp[comp["model"] == "Logistic Regression"].iloc[0]
        dt_gran = int(gran.loc[gran["model"] == "Decision Tree",
                               "distinct_probabilities_on_test"].iloc[0])
        rf_gran = int(gran.loc[gran["model"] == "Random Forest",
                               "distinct_probabilities_on_test"].iloc[0])

        print("\nAt the 0.5 cut-off the Decision Tree tops every point metric: "
              "accuracy %.4f, precision %.4f, recall %.4f, F1 %.4f. The Random "
              "Forest catches the same %d failures and differs by a single false "
              "alarm. On that evidence alone the tree looks like the answer, and "
              "it is the simpler model, so this is not a close call to wave away."
              % (dt["accuracy"], dt["precision"], dt["recall"], dt["f1"],
                 int(dt["TP"])))
        print("\nThe ranking metrics disagree. Random Forest leads on ROC-AUC "
              "(%.4f against %.4f) and on PR-AUC (%.4f against %.4f). The reason "
              "is visible in the predicted probabilities: the tuned tree has a "
              "depth-6 structure with a minimum of ten samples per leaf, so on "
              "the test set it emits only %d distinct probability values, one per "
              "reachable leaf. The forest emits %d."
              % (rf["roc_auc"], dt["roc_auc"], rf["pr_auc"], dt["pr_auc"],
                 dt_gran, rf_gran))
        print("That is the whole argument. A model with %d possible scores has %d "
              "possible operating points, and stage 09 exists precisely to move "
              "the operating point. The tree's advantage at 0.5 is a piece of "
              "luck about where its leaves happen to fall, not a property that "
              "survives being asked for more recall. The forest's ranking is what "
              "can be re-cut." % (dt_gran, dt_gran))
        print("\nSo the answer to 'which is best' splits cleanly. If the plant "
              "will run the model at a fixed default cut-off and never revisit "
              "it, the Decision Tree is the strongest and the cheapest to explain. "
              "If the plant wants to choose its own balance between %d missed "
              "failures and the false alarms that reducing them would cost, only "
              "the forest and XGBoost offer a dial fine enough to turn."
              % int(dt["FN"]))
        print("\nLogistic regression is the weakest on every criterion: PR-AUC "
              "%.4f against %.4f for the forest, and it finds %d of %d failures. "
              "That is the exploratory analysis coming true. Failures sit at both "
              "ends of the torque and speed range and behind a temperature-by-"
              "speed interaction, and a single linear boundary cannot enclose two "
              "opposite corners of a plane at once."
              % (lr["pr_auc"], rf["pr_auc"], int(lr["TP"]), int(y_te.sum())))
        print("\nAccuracy separates almost nothing: the four models span %.4f to "
              "%.4f, and the do-nothing baseline already sits at %.4f. Reporting "
              "it is a formality; deciding on it would be a mistake."
              % (comp["accuracy"].min(), comp["accuracy"].max(), 1 - y_te.mean()))

        # ------------------------------------------------------------------
        # Confusion matrices
        fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.2))
        for ax, name in zip(axes.ravel(), P.MODEL_ORDER):
            r = comp[comp["model"] == name].iloc[0]
            cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        square=True, linewidths=1.2, linecolor="white", ax=ax,
                        annot_kws={"fontsize": 13, "fontweight": "bold"},
                        xticklabels=["predicted\nno failure", "predicted\nfailure"],
                        yticklabels=["actual\nno failure", "actual\nfailure"])
            ax.set_title("%s\nrecall %.3f, precision %.3f, F1 %.3f"
                         % (name, r["recall"], r["precision"], r["f1"]), fontsize=10)
        fig.suptitle("Confusion matrices at the default 0.5 threshold, "
                     "test set of 2,000 records (68 failures)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        C.save_fig(fig, "fig15_confusion_matrices")
        plt.close(fig)

        # ROC curves
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
        for name, colour in zip(P.MODEL_ORDER, C.PALETTE[:4]):
            fpr, tpr, _ = roc_curve(y_te, probas[name])
            axes[0].plot(fpr, tpr, color=colour, linewidth=1.8,
                         label="%s (AUC %.3f)" % (name, roc_auc_score(y_te, probas[name])))
        axes[0].plot([0, 1], [0, 1], "k:", linewidth=1, label="Chance")
        axes[0].set_xlabel("False positive rate")
        axes[0].set_ylabel("True positive rate (recall)")
        axes[0].set_title("ROC curves")
        axes[0].legend(loc="lower right", fontsize=8.5)
        axes[0].set_xlim(-0.01, 1.01)
        axes[0].set_ylim(-0.01, 1.01)

        for name, colour in zip(P.MODEL_ORDER, C.PALETTE[:4]):
            prec, rec, _ = precision_recall_curve(y_te, probas[name])
            axes[1].plot(rec, prec, color=colour, linewidth=1.8,
                         label="%s (PR-AUC %.3f)"
                               % (name, average_precision_score(y_te, probas[name])))
        axes[1].axhline(y_te.mean(), color="black", linestyle=":", linewidth=1,
                        label="Base rate %.3f" % y_te.mean())
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision-recall curves")
        axes[1].legend(loc="lower left", fontsize=8.5)
        axes[1].set_xlim(-0.01, 1.01)
        axes[1].set_ylim(-0.01, 1.05)
        fig.suptitle("The two curves tell different stories at a 3.4% base rate",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        C.save_fig(fig, "fig16_roc_and_pr_curves")
        plt.close(fig)

        # metric comparison bars
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        x = np.arange(len(metrics))
        w = 0.2
        for i, (name, colour) in enumerate(zip(P.MODEL_ORDER, C.PALETTE[:4])):
            r = comp[comp["model"] == name].iloc[0]
            vals = [r[m] for m in metrics]
            ax.bar(x + (i - 1.5) * w, vals, w, label=name, color=colour,
                   edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC",
                            "PR-AUC"])
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Test-set score")
        ax.axhline(1 - y_te.mean(), color="black", linestyle=":", linewidth=1,
                   label="Majority-class accuracy")
        ax.set_title("Consolidated model comparison at the 0.5 threshold")
        ax.legend(fontsize=8, ncol=3, loc="lower left")
        fig.tight_layout()
        C.save_fig(fig, "fig17_metric_comparison")
        plt.close(fig)

        # ------------------------------------------------------------------
        # persist probabilities for stages 09 to 11
        # Rounded to 12 decimals on the way out. A Random Forest fitted with
        # n_jobs=-1 averages its trees in a thread-dependent order, so a handful
        # of probabilities differ in the final bit (around 1e-16) between runs
        # on the same seed. That is far below any threshold this study uses, and
        # rounding here makes the saved artifact byte-stable so that
        # src/reproducibility_check.py can assert exact reproduction.
        pr = pd.DataFrame(probas).round(12)
        pr["y_true"] = y_te
        pr.to_csv(C.TABLES / "test_predicted_probabilities.csv", index=False)
        print("\nPredicted probabilities written to "
              "outputs/tables/test_predicted_probabilities.csv")

        C.put("evaluation", {
            "table": comp[main_cols].to_dict("records"),
            "by_metric": by,
            "n_test": int(len(X_te)), "n_test_pos": int(y_te.sum()),
            "majority_acc": round(float(1 - y_te.mean()), 4),
            "acc_min": float(comp["accuracy"].min()),
            "acc_max": float(comp["accuracy"].max()),
            "best_pr_auc_model": str(best_pr["model"]),
            "best_pr_auc": float(best_pr["pr_auc"]),
            "best_recall_model": str(best_rec["model"]),
            "best_recall": float(best_rec["recall"]),
            "best_precision_model": str(best_prec["model"]),
            "granularity": {m: int(pd.Series(probas[m]).nunique())
                            for m in P.MODEL_ORDER},
        })
        print("\nStage 08 complete.")
    finally:
        C.end_log(tee)


if __name__ == "__main__":
    main()
