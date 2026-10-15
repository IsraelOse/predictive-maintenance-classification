"""
Stage 06 - The four required models and their parameter search  (Task 7)

For each of Logistic Regression, Decision Tree, Random Forest and XGBoost:
  an initial version at library defaults, scored by cross-validation;
  a systematic search over a stated space;
  the selected configuration and why it won.

The selection metric is fixed before anything runs: cross-validated PR-AUC
(average precision), 5-fold StratifiedKFold with shuffle=True and
random_state=93. Full cv_results_ tables are written to outputs/tables/ so the
search is evidenced rather than asserted.

Class imbalance is deliberately left untreated here. Stage 07 varies only the
imbalance strategy on top of these tuned structures, so that the two questions
stay separable.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import config as C
import pipelines as P

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

STAGE = "06_models_tuning"
plt, sns = C.setup_matplotlib()

CV = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.SEED)
SCORING = ["average_precision", "roc_auc", "recall", "precision", "f1", "accuracy"]


def make_pipe(estimator, scale: bool) -> Pipeline:
    return Pipeline([("pre", P.build_preprocessor(scale=scale)), ("clf", estimator)])


def cv_report(name, estimator, scale, X, y):
    """Cross-validate one configuration and return a tidy one-row summary."""
    res = cross_validate(make_pipe(estimator, scale), X, y, cv=CV,
                         scoring=SCORING, n_jobs=-1, return_train_score=False)
    row = {"model": name}
    for s in SCORING:
        row["cv_" + s] = round(float(res["test_" + s].mean()), 4)
        row["sd_" + s] = round(float(res["test_" + s].std()), 4)
    row["fit_seconds"] = round(float(res["fit_time"].sum()), 2)
    return row


def run_search(label, search, X, y, tag):
    t0 = time.time()
    search.fit(X, y)
    elapsed = time.time() - t0
    cvr = pd.DataFrame(search.cv_results_)
    keep = [c for c in cvr.columns
            if c.startswith("param_") or c in
            ("mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time")]
    tidy = (cvr[keep].sort_values("rank_test_score").reset_index(drop=True))
    tidy = tidy.round(5)
    C.save_table(tidy, "cv_%s_full_results" % tag)
    print("\n%s: %d configurations x %d folds = %d fits in %.1f s"
          % (label, len(cvr), C.CV_FOLDS, len(cvr) * C.CV_FOLDS, elapsed))
    print("Best cross-validated %s: %.4f" % (C.SELECTION_METRIC_LABEL,
                                             search.best_score_))
    print("Best parameters: %s" % search.best_params_)
    print("\nTop five configurations:")
    print(tidy.head(5).to_string(index=False))
    print("\nWorst configuration in the space scored %.4f, so the search spans "
          "a %.4f range." % (cvr["mean_test_score"].min(),
                             search.best_score_ - cvr["mean_test_score"].min()))
    return search, tidy, elapsed, len(cvr)


def main() -> None:
    tee = C.start_log(STAGE)
    try:
        C.banner("STAGE 06  MODEL DEVELOPMENT AND PARAMETER SEARCH")
        X_tr, X_te, y_tr, y_te = P.load_model_ready(use_engineered=True)
        print("Training on %d rows, %d failures (%.2f%%)"
              % (len(X_tr), y_tr.sum(), 100 * y_tr.mean()))
        print("\nSELECTION METRIC, DECLARED BEFORE ANY SEARCH RUNS")
        print("  %s, maximised over 5-fold StratifiedKFold "
              "(shuffle=True, random_state=%d)." % (C.SELECTION_METRIC_LABEL, C.SEED))
        print("  Accuracy is rejected as a selection metric because the majority-"
              "class predictor already scores 96.6% on it while catching zero "
              "failures, so it cannot distinguish a useful model from a useless one.")
        print("  ROC-AUC is reported but not used to select, because at a 3.4% "
              "positive rate the false-positive rate denominator is dominated by "
              "the 9,661 healthy records and large changes in the number of false "
              "alarms barely move the curve. PR-AUC uses precision instead, so it "
              "responds to exactly the quantity a maintenance planner pays for.")

        # ================================================== initial versions
        C.banner("STEP 1  INITIAL VERSIONS AT LIBRARY DEFAULTS")
        defaults = [
            ("Logistic Regression", LogisticRegression(max_iter=5000,
                                                       random_state=C.SEED), True),
            ("Decision Tree", DecisionTreeClassifier(random_state=C.SEED), False),
            ("Random Forest", RandomForestClassifier(random_state=C.SEED,
                                                     n_jobs=-1), False),
            ("XGBoost", XGBClassifier(random_state=C.SEED, eval_metric="logloss",
                                      n_jobs=-1, tree_method="hist"), False),
        ]
        default_rows = [cv_report(n, e, s, X_tr, y_tr) for n, e, s in defaults]
        dflt = pd.DataFrame(default_rows)
        C.save_table(dflt, "tbl25_default_model_cv")

        # ================================================== searches
        C.banner("STEP 2  SYSTEMATIC PARAMETER EXPLORATION")
        print("Grid search is used where the space is small enough to enumerate "
              "honestly. Randomised search with 80 draws is used for the two "
              "ensembles, because their spaces run to several hundred combinations "
              "and a random sample of a large space beats an exhaustive sweep of an "
              "artificially small one for the same compute.")

        searches = {}

        # ---- Logistic Regression ----------------------------------------
        C.banner("2a  LOGISTIC REGRESSION")
        lr_grid = {
            "clf__C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100],
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["liblinear"],
        }
        print("Space: C over eight orders of magnitude x {l1, l2} penalty. "
              "The penalty choice is the substantive one: L1 can zero out the "
              "collinear temperature pair, L2 shrinks it instead.")
        s, t, el, n = run_search("Logistic Regression",
                                 GridSearchCV(make_pipe(LogisticRegression(
                                     max_iter=5000, random_state=C.SEED), True),
                                     lr_grid, scoring=C.SELECTION_METRIC, cv=CV,
                                     n_jobs=-1, return_train_score=False),
                                 X_tr, y_tr, "lr")
        searches["Logistic Regression"] = (s, el, n, True)

        # ---- Decision Tree ----------------------------------------------
        C.banner("2b  DECISION TREE")
        dt_grid = {
            "clf__max_depth": [3, 4, 5, 6, 8, 10, None],
            "clf__min_samples_leaf": [1, 5, 10, 20, 50],
            "clf__criterion": ["gini", "entropy"],
            "clf__min_samples_split": [2, 10],
        }
        print("Space: depth, leaf size, split size and impurity criterion. With 271 "
              "training failures, a leaf minimum of 1 will happily isolate single "
              "failures, so leaf size is the parameter that actually controls "
              "overfitting here.")
        s, t, el, n = run_search("Decision Tree",
                                 GridSearchCV(make_pipe(DecisionTreeClassifier(
                                     random_state=C.SEED), False),
                                     dt_grid, scoring=C.SELECTION_METRIC, cv=CV,
                                     n_jobs=-1, return_train_score=False),
                                 X_tr, y_tr, "dt")
        searches["Decision Tree"] = (s, el, n, False)

        # ---- Random Forest ----------------------------------------------
        C.banner("2c  RANDOM FOREST")
        rf_space = {
            "clf__n_estimators": [200, 300, 400, 600, 800],
            "clf__max_depth": [None, 6, 8, 12, 16, 20],
            "clf__min_samples_leaf": [1, 2, 3, 5, 10],
            "clf__min_samples_split": [2, 5, 10],
            "clf__max_features": ["sqrt", "log2", 0.4, 0.6],
        }
        print("Space: 1,800 combinations, sampled 80 times with random_state=%d."
              % C.SEED)
        s, t, el, n = run_search("Random Forest",
                                 RandomizedSearchCV(make_pipe(RandomForestClassifier(
                                     random_state=C.SEED, n_jobs=1), False),
                                     rf_space, n_iter=80, scoring=C.SELECTION_METRIC,
                                     cv=CV, n_jobs=-1, random_state=C.SEED,
                                     return_train_score=False),
                                 X_tr, y_tr, "rf")
        searches["Random Forest"] = (s, el, n, False)

        # ---- XGBoost ------------------------------------------------------
        C.banner("2d  XGBOOST")
        xgb_space = {
            "clf__n_estimators": [200, 300, 400, 600, 800],
            "clf__max_depth": [2, 3, 4, 5, 6, 8],
            "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
            "clf__subsample": [0.6, 0.8, 1.0],
            "clf__colsample_bytree": [0.6, 0.8, 1.0],
            "clf__min_child_weight": [1, 3, 5, 10],
            "clf__gamma": [0, 0.1, 0.5],
            "clf__reg_lambda": [0.5, 1, 5],
        }
        print("Space: 116,640 combinations, sampled 80 times with random_state=%d. "
              "Exhaustive search is not on the table at that size, which is exactly "
              "why the randomised strategy is the right one." % C.SEED)
        s, t, el, n = run_search("XGBoost",
                                 RandomizedSearchCV(make_pipe(XGBClassifier(
                                     random_state=C.SEED, eval_metric="logloss",
                                     n_jobs=1, tree_method="hist"), False),
                                     xgb_space, n_iter=80, scoring=C.SELECTION_METRIC,
                                     cv=CV, n_jobs=-1, random_state=C.SEED,
                                     return_train_score=False),
                                 X_tr, y_tr, "xgb")
        searches["XGBoost"] = (s, el, n, False)

        # ================================================== selected configs
        C.banner("STEP 3  SELECTED CONFIGURATIONS")
        tuned_rows, chosen = [], {}
        for name, (search, el, n_cfg, scale) in searches.items():
            best = search.best_estimator_.named_steps["clf"]
            chosen[name] = {
                "params": {k.replace("clf__", ""): v
                           for k, v in search.best_params_.items()},
                "scale": scale,
                "cv_best": round(float(search.best_score_), 4),
                "n_configs": int(n_cfg),
                "search_seconds": round(el, 1),
            }
            tuned_rows.append(cv_report(name + " (tuned)", best, scale, X_tr, y_tr))
        tuned = pd.DataFrame(tuned_rows)
        C.save_table(tuned, "tbl26_tuned_model_cv")

        cfg_tbl = pd.DataFrame([
            {"model": k,
             "n_configurations_tried": v["n_configs"],
             "search_seconds": v["search_seconds"],
             "selected_parameters": ", ".join("%s=%s" % (a, b)
                                              for a, b in sorted(v["params"].items())),
             "cv_pr_auc": v["cv_best"]}
            for k, v in chosen.items()
        ])
        C.save_table(cfg_tbl, "tbl27_selected_configurations")

        # default versus tuned, side by side
        comp = pd.DataFrame({
            "model": dflt["model"],
            "default_cv_pr_auc": dflt["cv_average_precision"],
            "tuned_cv_pr_auc": tuned["cv_average_precision"].to_numpy(),
        })
        comp["improvement"] = (comp["tuned_cv_pr_auc"] -
                               comp["default_cv_pr_auc"]).round(4)
        comp["default_cv_recall"] = dflt["cv_recall"]
        comp["tuned_cv_recall"] = tuned["cv_recall"].to_numpy()
        C.save_table(comp, "tbl28_default_vs_tuned")

        C.banner("WHAT THE SEARCH BOUGHT")
        for _, r in comp.iterrows():
            print("%-22s default %.4f -> tuned %.4f  (%+.4f)"
                  % (r["model"], r["default_cv_pr_auc"], r["tuned_cv_pr_auc"],
                     r["improvement"]))

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
        x = np.arange(len(comp))
        w = 0.36
        axes[0].bar(x - w / 2, comp["default_cv_pr_auc"], w, label="Library defaults",
                    color=C.PALETTE[0], edgecolor="black", linewidth=0.5)
        axes[0].bar(x + w / 2, comp["tuned_cv_pr_auc"], w, label="After the search",
                    color=C.PALETTE[1], edgecolor="black", linewidth=0.5)
        for i, r in comp.iterrows():
            axes[0].text(i - w / 2, r["default_cv_pr_auc"] + 0.008,
                         "%.3f" % r["default_cv_pr_auc"], ha="center", fontsize=8)
            axes[0].text(i + w / 2, r["tuned_cv_pr_auc"] + 0.008,
                         "%.3f" % r["tuned_cv_pr_auc"], ha="center", fontsize=8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([m.replace(" ", "\n") for m in comp["model"]],
                                fontsize=9)
        axes[0].set_ylabel("Cross-validated PR-AUC")
        axes[0].set_ylim(0, 1.02)
        axes[0].set_title("Parameter search against library defaults")
        axes[0].legend(fontsize=8)

        # spread of the search for each model
        spread = []
        for tag, name in [("lr", "Logistic Regression"), ("dt", "Decision Tree"),
                          ("rf", "Random Forest"), ("xgb", "XGBoost")]:
            t = pd.read_csv(C.TABLES / ("cv_%s_full_results.csv" % tag))
            spread.append(pd.DataFrame({"model": name,
                                        "score": t["mean_test_score"]}))
        spread = pd.concat(spread)
        sns.boxplot(data=spread, x="model", y="score", ax=axes[1],
                    hue="model", palette=C.PALETTE[:4], legend=False,
                    width=0.55, fliersize=2.5)
        axes[1].set_xticks(range(4))
        axes[1].set_xticklabels([m.replace(" ", "\n") for m in comp["model"]],
                                fontsize=9)
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Cross-validated PR-AUC of each configuration tried")
        axes[1].set_title("How much the parameters actually mattered")
        fig.suptitle("Parameter exploration, 5-fold CV on the training partition (seed 93)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        C.save_fig(fig, "fig12_tuning_summary")
        plt.close(fig)

        C.put("tuning", {
            "chosen": chosen,
            "defaults": dflt.to_dict("records"),
            "tuned": tuned.to_dict("records"),
            "comparison": comp.to_dict("records"),
            "total_configs": int(sum(v["n_configs"] for v in chosen.values())),
            "total_fits": int(sum(v["n_configs"] for v in chosen.values()) * C.CV_FOLDS),
            "search_spread": {name: {"min": round(float(g["score"].min()), 4),
                                     "max": round(float(g["score"].max()), 4)}
                              for name, g in spread.groupby("model")},
        })
        print("\nStage 06 complete. %d configurations evaluated in total, "
              "%d model fits." % (sum(v["n_configs"] for v in chosen.values()),
                                  sum(v["n_configs"] for v in chosen.values()) * C.CV_FOLDS))
    finally:
        C.end_log(tee)


if __name__ == "__main__":
    main()
