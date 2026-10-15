"""
Stage 05 - Baselines  (Task 6)

Five reference points, from the trivial to the plausible:
  a majority-class predictor, which sets the accuracy trap;
  a stratified random guesser, which shows what chance alone buys;
  two rules a maintenance planner could already apply without any model;
  a one-variable logistic regression on torque.

Anything the four required models produce has to beat these to have earned the
extra complexity.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import config as C
import pipelines as P

STAGE = "05_baseline"
plt, sns = C.setup_matplotlib()


def main() -> None:
    tee = C.start_log(STAGE)
    try:
        C.banner("STAGE 05  BASELINE PERFORMANCE")
        X_tr, X_te, y_tr, y_te = P.load_model_ready(use_engineered=True)
        num, cat = P.feature_columns(True)
        print("Train %d rows (%d failures), test %d rows (%d failures)"
              % (len(X_tr), y_tr.sum(), len(X_te), y_te.sum()))

        rows = []

        # ---------------------------------------------------------- 1. majority
        dm = DummyClassifier(strategy="most_frequent")
        dm.fit(X_tr, y_tr)
        rows.append(P.score_row("B1 Majority class (always predict no failure)",
                                y_te, dm.predict(X_te), dm.predict_proba(X_te)[:, 1]))

        # ---------------------------------------------------------- 2. stratified
        ds = DummyClassifier(strategy="stratified", random_state=C.SEED)
        ds.fit(X_tr, y_tr)
        rows.append(P.score_row("B2 Stratified random guess (seed 93)",
                                y_te, ds.predict(X_te), ds.predict_proba(X_te)[:, 1]))

        # ---------------------------------------------------------- 3. wear rule
        wear_rule = (X_te["tool_wear_min"] >= 200).astype(int).to_numpy()
        rows.append(P.score_row("B3 Operator rule: tool wear >= 200 min",
                                y_te, wear_rule,
                                X_te["tool_wear_min"].to_numpy() / 260.0))

        # ---------------------------------------------------------- 4. power rule
        pw = X_te["power_W"].to_numpy()
        power_rule = ((pw < 3500) | (pw > 9000)).astype(int)
        # a monotone score for the curve metrics: distance outside the safe band
        power_score = np.maximum(3500 - pw, pw - 9000)
        power_score = (power_score - power_score.min()) / np.ptp(power_score)
        rows.append(P.score_row("B4 Operator rule: power outside 3,500 to 9,000 W",
                                y_te, power_rule, power_score))

        # ---------------------------------------------------------- 5. one feature
        one = Pipeline([
            ("pre", P.build_preprocessor_cols(["torque_Nm"], [], scale=True)),
            ("clf", LogisticRegression(max_iter=2000, random_state=C.SEED)),
        ])
        one.fit(X_tr[["torque_Nm"]], y_tr)
        rows.append(P.score_row("B5 Logistic regression on torque alone",
                                y_te, one.predict(X_te[["torque_Nm"]]),
                                one.predict_proba(X_te[["torque_Nm"]])[:, 1]))

        base = pd.DataFrame(rows)
        C.save_table(base, "tbl24_baseline_results")

        # ------------------------------------------------------------------
        C.banner("WHAT THE BASELINES SHOW")
        maj = base.iloc[0]
        strat = base.iloc[1]
        wear = base.iloc[2]
        power = base.iloc[3]
        onef = base.iloc[4]

        print("The majority-class predictor scores %.2f%% accuracy while catching "
              "none of the %d failures in the test set. Its recall is %.2f and its "
              "precision is undefined, reported as %.2f because it never issues a "
              "positive prediction. That is the whole problem with accuracy at a "
              "3.4%% base rate: a model can be right almost all the time and useless "
              "all of the time."
              % (100 * maj["accuracy"], int(y_te.sum()), maj["recall"], maj["precision"]))
        print("\nStratified guessing lands at %.2f%% accuracy and %.3f recall, which "
              "is roughly the base rate reappearing on both sides of the confusion "
              "matrix. It flags %d records to catch %d failures."
              % (100 * strat["accuracy"], strat["recall"],
                 int(strat["FP"] + strat["TP"]), int(strat["TP"])))
        print("\nThe tool-wear rule is the interesting one. It catches %d of %d "
              "failures at %.1f%% precision, so an engineer following it inspects "
              "%d machines to find %d real problems. Recall %.3f, F1 %.3f."
              % (int(wear["TP"]), int(y_te.sum()), 100 * wear["precision"],
                 int(wear["FP"] + wear["TP"]), int(wear["TP"]),
                 wear["recall"], wear["f1"]))
        print("\nThe power-band rule catches %d failures at %.1f%% precision, "
              "F1 %.3f. It is the strongest thing available without fitting "
              "anything, and it is the number the machine-learning models actually "
              "have to beat."
              % (int(power["TP"]), 100 * power["precision"], power["f1"]))
        print("\nLogistic regression on torque alone reaches ROC-AUC %.3f but "
              "predicts failure for %d records at the default 0.5 cut-off, so its "
              "recall is %.3f. The ranking information is there; the default "
              "threshold cannot reach it."
              % (onef["roc_auc"], int(onef["FP"] + onef["TP"]), onef["recall"]))

        print("\nTaken together these say the problem is hard in a specific way. "
              "Ranking failures above healthy records is achievable. Converting "
              "that ranking into a decision at a fixed 0.5 cut-off is where the "
              "base rate does its damage, which is why stage 09 treats the "
              "threshold as a parameter rather than a constant.")

        # ------------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
        labels = [s.split(" ", 1)[0] for s in base["model"]]
        xpos = np.arange(len(base))
        w = 0.27
        axes[0].bar(xpos - w, base["accuracy"], w, label="Accuracy",
                    color=C.PALETTE[0], edgecolor="black", linewidth=0.5)
        axes[0].bar(xpos, base["recall"], w, label="Recall",
                    color=C.PALETTE[1], edgecolor="black", linewidth=0.5)
        axes[0].bar(xpos + w, base["precision"], w, label="Precision",
                    color=C.PALETTE[2], edgecolor="black", linewidth=0.5)
        axes[0].axhline(1 - y_te.mean(), color="black", linestyle=":",
                        label="Base-rate accuracy %.3f" % (1 - y_te.mean()))
        axes[0].set_xticks(xpos)
        axes[0].set_xticklabels(labels)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("Score on the held-out test set")
        axes[0].set_title("High accuracy and zero recall can coexist")
        axes[0].legend(fontsize=8, loc="upper right")

        axes[1].barh(xpos, base["TP"], color=C.C_FAIL, edgecolor="black",
                     linewidth=0.5, label="Failures caught (TP)")
        axes[1].barh(xpos, base["FN"], left=base["TP"], color="#cccccc",
                     edgecolor="black", linewidth=0.5, label="Failures missed (FN)")
        for i, r in base.iterrows():
            axes[1].text(int(y_te.sum()) + 1.5, i, "%d false alarms" % int(r["FP"]),
                         va="center", fontsize=8.5)
        axes[1].set_yticks(xpos)
        axes[1].set_yticklabels(labels)
        axes[1].set_xlim(0, int(y_te.sum()) * 1.55)
        axes[1].set_xlabel("Test-set failures (n = %d)" % int(y_te.sum()))
        axes[1].set_title("What each baseline actually catches")
        axes[1].legend(fontsize=8, loc="lower right")
        fig.suptitle("Baseline reference points, held-out test set of 2,000 records",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        C.save_fig(fig, "fig11_baseline_comparison")
        plt.close(fig)

        C.put("baseline", {
            "n_test": int(len(X_te)), "n_test_pos": int(y_te.sum()),
            "majority_acc": float(maj["accuracy"]),
            "majority_recall": float(maj["recall"]),
            "strat_acc": float(strat["accuracy"]),
            "strat_recall": float(strat["recall"]),
            "strat_tp": int(strat["TP"]), "strat_fp": int(strat["FP"]),
            "wear_tp": int(wear["TP"]), "wear_fp": int(wear["FP"]),
            "wear_recall": float(wear["recall"]), "wear_precision": float(wear["precision"]),
            "wear_f1": float(wear["f1"]),
            "power_tp": int(power["TP"]), "power_fp": int(power["FP"]),
            "power_recall": float(power["recall"]),
            "power_precision": float(power["precision"]), "power_f1": float(power["f1"]),
            "onef_auc": float(onef["roc_auc"]), "onef_recall": float(onef["recall"]),
            "best_baseline_f1": float(base["f1"].max()),
            "best_baseline_name": str(base.loc[base["f1"].idxmax(), "model"]),
            "best_baseline_pr_auc": float(base["pr_auc"].max()),
            "table": base.to_dict("records"),
        })
        print("\nStage 05 complete.")
    finally:
        C.end_log(tee)


if __name__ == "__main__":
    main()
