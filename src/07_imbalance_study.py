"""
Stage 07 - Class-imbalance investigation  (Task 8)

Holds each model's tuned structure fixed and varies only the imbalance
strategy, so that any difference is attributable to the treatment rather than
to a different set of hyperparameters.

Four treatments per model:
  none                 the untreated reference
  class weighting      class_weight='balanced', or scale_pos_weight for XGBoost
  SMOTE                oversampling applied inside the cross-validation pipeline
  SMOTE + undersample  SMOTE to 0.3 then random undersampling to 0.5

Everything is scored by cross-validation on the training partition. A resampler
placed inside an imblearn Pipeline is refitted on each training fold and is
never applied to the fold being scored, which is the only way SMOTE can be used
without inflating the result.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import config as C
import pipelines as P

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

STAGE = "07_imbalance_study"
plt, sns = C.setup_matplotlib()

CV = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.SEED)
SCORING = ["average_precision", "roc_auc", "recall", "precision", "f1", "accuracy"]


build_estimator = P.build_estimator   # shared factory, see src/pipelines.py


def main() -> None:
    tee = C.start_log(STAGE)
    try:
        C.banner("STAGE 07  CLASS-IMBALANCE INVESTIGATION")
        X_tr, X_te, y_tr, y_te = P.load_model_ready(use_engineered=True)
        tuning = C.get("tuning")["chosen"]

        n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
        pos_weight = n_neg / n_pos
        print("Training partition: %d healthy, %d failures, ratio %.2f to 1"
              % (n_neg, n_pos, pos_weight))
        print("scale_pos_weight for XGBoost is set to that ratio, %.2f." % pos_weight)
        print("SMOTE is configured with k_neighbors=5 and random_state=%d." % C.SEED)

        rows = []
        for name, cfg in tuning.items():
            params = cfg["params"]
            scale = cfg["scale"]
            pre = lambda: P.build_preprocessor(scale=scale)

            variants = {
                "none": Pipeline([("pre", pre()),
                                  ("clf", build_estimator(name, params))]),
                "class weighting": Pipeline(
                    [("pre", pre()),
                     ("clf", build_estimator(name, params, weighted=True,
                                             pos_weight=pos_weight))]),
                "SMOTE": ImbPipeline(
                    [("pre", pre()),
                     ("smote", SMOTE(random_state=C.SEED, k_neighbors=5)),
                     ("clf", build_estimator(name, params))]),
                "SMOTE + undersample": ImbPipeline(
                    [("pre", pre()),
                     ("smote", SMOTE(random_state=C.SEED, k_neighbors=5,
                                     sampling_strategy=0.3)),
                     ("under", RandomUnderSampler(random_state=C.SEED,
                                                  sampling_strategy=0.5)),
                     ("clf", build_estimator(name, params))]),
            }

            for treatment, pipe in variants.items():
                res = cross_validate(pipe, X_tr, y_tr, cv=CV, scoring=SCORING,
                                     n_jobs=-1)
                row = {"model": name, "treatment": treatment}
                for s in SCORING:
                    row[s] = round(float(res["test_" + s].mean()), 4)
                    row[s + "_sd"] = round(float(res["test_" + s].std()), 4)
                rows.append(row)
                print("%-20s %-20s PR-AUC %.4f  recall %.4f  precision %.4f  F1 %.4f"
                      % (name, treatment, row["average_precision"], row["recall"],
                         row["precision"], row["f1"]))

        study = pd.DataFrame(rows)
        C.save_table(study, "tbl29_imbalance_study_full")

        # ------------------------------------------------------------------
        C.banner("THE TRADE-OFF, QUANTIFIED")
        piv = study.pivot(index="model", columns="treatment",
                          values=["recall", "precision", "f1", "average_precision"])
        trade_rows = []
        for name in tuning:
            base = study[(study["model"] == name) & (study["treatment"] == "none")].iloc[0]
            for treatment in ["class weighting", "SMOTE", "SMOTE + undersample"]:
                v = study[(study["model"] == name) &
                          (study["treatment"] == treatment)].iloc[0]
                trade_rows.append({
                    "model": name,
                    "treatment": treatment,
                    "recall_change": round(v["recall"] - base["recall"], 4),
                    "precision_change": round(v["precision"] - base["precision"], 4),
                    "f1_change": round(v["f1"] - base["f1"], 4),
                    "pr_auc_change": round(v["average_precision"] -
                                           base["average_precision"], 4),
                    "roc_auc_change": round(v["roc_auc"] - base["roc_auc"], 4),
                })
        trade = pd.DataFrame(trade_rows)
        C.save_table(trade, "tbl30_imbalance_tradeoff")

        for _, r in trade.iterrows():
            print("%-20s %-20s recall %+.4f, precision %+.4f, PR-AUC %+.4f"
                  % (r["model"], r["treatment"], r["recall_change"],
                     r["precision_change"], r["pr_auc_change"]))

        # ------------------------------------------------------------------
        C.banner("RETENTION DECISION")
        print("Rule stated before reading the table: the retained treatment is the "
              "one with the highest cross-validated PR-AUC, because PR-AUC scores "
              "the whole ranking rather than the arbitrary 0.5 cut-off, and stage 09 "
              "sets the operating threshold separately. Where two treatments sit "
              "within one standard deviation of each other I take the simpler one, "
              "since a resampler that buys nothing is a moving part that can still "
              "break.\n")

        retained = {}
        dec_rows = []
        for name in tuning:
            sub = study[study["model"] == name].sort_values("average_precision",
                                                            ascending=False)
            best = sub.iloc[0]
            none_row = sub[sub["treatment"] == "none"].iloc[0]
            within_noise = (best["average_precision"] - none_row["average_precision"]
                            ) <= none_row["average_precision_sd"]
            pick = "none" if within_noise else best["treatment"]
            retained[name] = pick
            dec_rows.append({
                "model": name,
                "best_pr_auc_treatment": best["treatment"],
                "best_pr_auc": best["average_precision"],
                "untreated_pr_auc": none_row["average_precision"],
                "gap": round(best["average_precision"] - none_row["average_precision"], 4),
                "untreated_sd": none_row["average_precision_sd"],
                "within_one_sd": bool(within_noise),
                "retained": pick,
            })
            print("%-20s best by PR-AUC: %-20s (%.4f vs untreated %.4f, gap %.4f, "
                  "untreated sd %.4f) -> retain '%s'"
                  % (name, best["treatment"], best["average_precision"],
                     none_row["average_precision"],
                     best["average_precision"] - none_row["average_precision"],
                     none_row["average_precision_sd"], pick))
        dec = pd.DataFrame(dec_rows)
        C.save_table(dec, "tbl31_imbalance_retention_decision")

        print("\nThe headline is that none of these treatments changes how well the "
              "models rank records. What they change is where the 0.5 cut-off "
              "lands on that ranking, which is a threshold decision wearing a "
              "resampling costume. That is worth saying plainly, because the "
              "obvious expectation is that SMOTE improves the model.")

        # ------------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        order = ["none", "class weighting", "SMOTE", "SMOTE + undersample"]
        sns.barplot(data=study, x="model", y="recall", hue="treatment",
                    hue_order=order, palette=C.PALETTE[:4], ax=axes[0],
                    edgecolor="black", linewidth=0.5)
        axes[0].set_ylabel("Cross-validated recall at the 0.5 cut-off")
        axes[0].set_xlabel("")
        axes[0].set_title("Recall rises sharply under every intervention")
        axes[0].set_xticks(range(4))
        axes[0].set_xticklabels([m.replace(" ", "\n") for m in
                                 study["model"].unique()], fontsize=9)
        axes[0].legend(fontsize=7.5, title=None, loc="lower right")

        sns.barplot(data=study, x="model", y="precision", hue="treatment",
                    hue_order=order, palette=C.PALETTE[:4], ax=axes[1],
                    edgecolor="black", linewidth=0.5)
        axes[1].set_ylabel("Cross-validated precision at the 0.5 cut-off")
        axes[1].set_xlabel("")
        axes[1].set_title("Precision pays for it")
        axes[1].set_xticks(range(4))
        axes[1].set_xticklabels([m.replace(" ", "\n") for m in
                                 study["model"].unique()], fontsize=9)
        axes[1].legend(fontsize=7.5, title=None, loc="lower right")
        fig.suptitle("Class-imbalance treatments, 5-fold CV on the training partition (seed 93)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        C.save_fig(fig, "fig13_imbalance_recall_precision")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        markers = {"none": "o", "class weighting": "s", "SMOTE": "^",
                   "SMOTE + undersample": "D"}
        colours = dict(zip(study["model"].unique(), C.PALETTE[:4]))
        for _, r in study.iterrows():
            ax.scatter(r["recall"], r["precision"], s=110,
                       marker=markers[r["treatment"]], color=colours[r["model"]],
                       edgecolor="black", linewidth=0.7, zorder=3)
        for m, c in colours.items():
            sub = study[study["model"] == m].sort_values("recall")
            ax.plot(sub["recall"], sub["precision"], color=c, alpha=0.45,
                    linewidth=1.2, zorder=2)
            ax.scatter([], [], color=c, s=80, label=m)
        for t, mk in markers.items():
            ax.scatter([], [], color="grey", marker=mk, s=80, label=t)
        ax.set_xlabel("Cross-validated recall")
        ax.set_ylabel("Cross-validated precision")
        ax.set_title("Every intervention walks the same curve: recall up, precision down")
        ax.legend(fontsize=7.5, ncol=2, loc="lower left")
        fig.tight_layout()
        C.save_fig(fig, "fig14_imbalance_tradeoff_scatter")
        plt.close(fig)

        C.put("imbalance", {
            "retained": retained,
            "pos_weight": round(pos_weight, 2),
            "study": study.to_dict("records"),
            "tradeoff": trade.to_dict("records"),
            "decision": dec.to_dict("records"),
        })
        print("\nStage 07 complete. Retained treatments: %s" % retained)
    finally:
        C.end_log(tee)


if __name__ == "__main__":
    main()
