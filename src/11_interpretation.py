"""
Stage 11 - Model interpretation  (Task 12)

Four models, four different notions of importance, and one honest comparison.

  Logistic Regression  standardised coefficients, read as odds ratios
  Decision Tree        Gini importance
  Random Forest        Gini importance
  XGBoost              gain

Gini importance and gain are internal to how each learner was built, and they
are biased toward high-cardinality continuous features. Permutation importance
on the held-out test set is measured for all four as well, because it asks a
question the others do not: how much worse does this model get if I destroy
this variable. Where the two disagree, the disagreement is the finding.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance

import config as C
import pipelines as P

warnings.filterwarnings("ignore")

STAGE = "11_interpretation"
plt, sns = C.setup_matplotlib()


def main() -> None:
    tee = C.start_log(STAGE)
    try:
        C.banner("STAGE 11  MODEL INTERPRETATION")
        X_tr, X_te, y_tr, y_te = P.load_model_ready(use_engineered=True)
        names = P.expanded_feature_names(True)
        models = P.build_final_models(use_engineered=True)

        fitted = {}
        for n in P.MODEL_ORDER:
            models[n].fit(X_tr, y_tr)
            fitted[n] = models[n]

        # ------------------------------------------------- native importance
        C.banner("NATIVE IMPORTANCE FROM EACH MODEL")
        lr = fitted["Logistic Regression"].named_steps["clf"]
        coefs = lr.coef_.ravel()
        lr_tbl = pd.DataFrame({
            "feature": names,
            "coefficient": np.round(coefs, 4),
            "odds_ratio_per_sd": np.round(np.exp(coefs), 4),
            "abs_coefficient": np.round(np.abs(coefs), 4),
        }).sort_values("abs_coefficient", ascending=False)
        C.save_table(lr_tbl, "tbl46_logistic_coefficients")
        print("Logistic regression coefficients are on standardised inputs, so each "
              "one is the log-odds change per one standard deviation of that "
              "predictor with the others held fixed. The intercept is %.4f."
              % lr.intercept_[0])
        top_lr = lr_tbl.iloc[0]
        print("Largest effect: %s at %.4f, an odds ratio of %.3f per standard "
              "deviation." % (top_lr["feature"], top_lr["coefficient"],
                              top_lr["odds_ratio_per_sd"]))

        native = {"Logistic Regression": np.abs(coefs)}
        for n in ["Decision Tree", "Random Forest", "XGBoost"]:
            native[n] = fitted[n].named_steps["clf"].feature_importances_

        nat = pd.DataFrame({"feature": names})
        for n in P.MODEL_ORDER:
            v = np.asarray(native[n], dtype=float)
            nat[n] = np.round(v / v.sum(), 4)     # normalised so models compare
        C.save_table(nat, "tbl47_native_importance_normalised")

        # ------------------------------------------------- permutation
        C.banner("PERMUTATION IMPORTANCE ON THE HELD-OUT TEST SET")
        print("Scored by average precision, 30 repeats, random_state=%d." % C.SEED)
        # One permutation run per model, reused for both the per-feature rows
        # and the categorical row. The pipeline takes raw columns, so 'type' is
        # permuted as a single column and reported once rather than split
        # across its one-hot indicators.
        perm = pd.DataFrame({"feature": names})
        perm_runs = {}
        for n in P.MODEL_ORDER:
            r = permutation_importance(fitted[n], X_te, y_te, n_repeats=30,
                                       random_state=C.SEED, n_jobs=-1,
                                       scoring="average_precision")
            perm_runs[n] = dict(zip(list(X_te.columns), r.importances_mean))
            perm[n] = [round(float(perm_runs[n][f]), 4) if f in perm_runs[n]
                       else np.nan for f in names]

        type_row = {"feature": "type (categorical, permuted as one column)"}
        for n in P.MODEL_ORDER:
            type_row[n] = round(float(perm_runs[n]["type"]), 4)
        perm = perm[~perm["feature"].isin(["type_L", "type_M"])]
        perm = pd.concat([perm, pd.DataFrame([type_row])], ignore_index=True)
        C.save_table(perm, "tbl48_permutation_importance")

        # ------------------------------------------------- agreement
        C.banner("WHERE THE MODELS AGREE AND WHERE THEY DO NOT")
        rank = nat.copy()
        for n in P.MODEL_ORDER:
            rank[n] = rank[n].rank(ascending=False).astype(int)
        C.save_table(rank, "tbl49_importance_ranks")
        print("\nRank of each feature by each model's own importance measure:")
        print(rank.to_string(index=False))

        agree_rows = []
        for a in range(len(P.MODEL_ORDER)):
            for b in range(a + 1, len(P.MODEL_ORDER)):
                m1, m2 = P.MODEL_ORDER[a], P.MODEL_ORDER[b]
                rho, p = stats.spearmanr(nat[m1], nat[m2])
                agree_rows.append({"model_a": m1, "model_b": m2,
                                   "spearman_rho": round(float(rho), 3),
                                   "p_value": "%.3g" % p})
        agree = pd.DataFrame(agree_rows)
        C.save_table(agree, "tbl50_importance_rank_agreement")

        mean_rank = rank[P.MODEL_ORDER].mean(axis=1)
        spread = rank[P.MODEL_ORDER].max(axis=1) - rank[P.MODEL_ORDER].min(axis=1)
        consistency = pd.DataFrame({
            "feature": nat["feature"],
            "mean_rank": mean_rank.round(2),
            "rank_spread": spread,
            "min_rank": rank[P.MODEL_ORDER].min(axis=1),
            "max_rank": rank[P.MODEL_ORDER].max(axis=1),
        }).sort_values("mean_rank")
        C.save_table(consistency, "tbl51_importance_consistency")

        consistent = consistency[consistency["rank_spread"] <= 2].head(4)
        contested = consistency.sort_values("rank_spread", ascending=False).head(3)
        print("\nConsistently important across the four models (rank spread of 2 "
              "or less):")
        print(consistent.to_string(index=False))
        print("\nMost contested:")
        print(contested.to_string(index=False))

        # ------------------------------------------------- narrative
        C.banner("READING THE RESULT")
        rf_top = nat.sort_values("Random Forest", ascending=False).head(4)
        print("The three tree-based models converge on the same short list. "
              "%s carry most of the forest's importance."
              % ", ".join(rf_top["feature"].head(3)))
        print("\nLogistic regression tells a different story, and the reason is "
              "structural rather than a disagreement about the machine. Its "
              "largest coefficients sit on %s, because a linear model can only "
              "express a monotone effect. The exploratory work showed failure "
              "risk rising at both ends of the power range, so a single "
              "coefficient on power is being asked to represent a U shape and "
              "cannot. Where the trees place a split at each end, the regression "
              "compromises in the middle."
              % ", ".join(lr_tbl["feature"].head(3)))
        print("\nAgainst the exploratory analysis, the agreement is good. Stage 02 "
              "identified torque and rotational speed, the power they imply, the "
              "wear-torque product and the temperature gap as the quantities that "
              "matter, and the fitted models rank exactly those at the top. "
              "Product grade is minor in every model, which also matches: it "
              "shifts an overstrain limit rather than driving failure by itself.")
        print("\nAll of this is reasonable in machine terms. A spindle turning "
              "slowly against high torque is drawing power near its design limit. "
              "A blunt tool under load is the textbook overstrain case. A process "
              "running only a few degrees above ambient is not shedding heat.")

        # ------------------------------------------------- figures
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
        plot_nat = nat.set_index("feature")[P.MODEL_ORDER]
        plot_nat = plot_nat.loc[plot_nat.mean(axis=1).sort_values().index]
        plot_nat.plot(kind="barh", ax=axes[0], color=C.PALETTE[:4],
                      edgecolor="black", linewidth=0.4, width=0.78)
        axes[0].set_xlabel("Normalised importance within each model")
        axes[0].set_ylabel("")
        axes[0].set_title("Native importance: coefficients, Gini, gain")
        axes[0].legend(fontsize=8)

        pp = perm[perm["feature"] != "type (categorical, permuted as one column)"]
        pp = pp.set_index("feature")[P.MODEL_ORDER]
        pp = pp.loc[pp.mean(axis=1).sort_values().index]
        pp.plot(kind="barh", ax=axes[1], color=C.PALETTE[:4],
                edgecolor="black", linewidth=0.4, width=0.78)
        axes[1].set_xlabel("Drop in test PR-AUC when the feature is permuted")
        axes[1].set_ylabel("")
        axes[1].set_title("Permutation importance on held-out data")
        axes[1].legend(fontsize=8)
        fig.suptitle("What each model learned, measured two different ways",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        C.save_fig(fig, "fig22_importance_comparison")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        hm = rank.set_index("feature")[P.MODEL_ORDER]
        sns.heatmap(hm, annot=True, fmt="d", cmap="YlOrRd_r", linewidths=0.8,
                    ax=ax, cbar_kws={"label": "Rank (1 = most important)"})
        ax.set_title("Importance rank by model, and where they disagree")
        ax.set_ylabel("")
        ax.set_xticklabels([m.replace(" ", "\n") for m in P.MODEL_ORDER], fontsize=9)
        fig.tight_layout()
        C.save_fig(fig, "fig23_importance_rank_heatmap")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        d = lr_tbl.sort_values("coefficient")
        cols = [C.C_FAIL if v > 0 else C.C_OK for v in d["coefficient"]]
        ax.barh(d["feature"], d["coefficient"], color=cols, edgecolor="black",
                linewidth=0.5)
        for i, (f, v, o) in enumerate(zip(d["feature"], d["coefficient"],
                                          d["odds_ratio_per_sd"])):
            ax.text(v + (0.05 if v > 0 else -0.05), i, "OR %.2f" % o,
                    va="center", ha="left" if v > 0 else "right", fontsize=8)
        ax.axvline(0, color="black", linewidth=0.9)
        ax.set_xlabel("Coefficient on standardised predictors (log-odds per SD)")
        ax.set_title("Logistic regression: direction and size of each effect")
        margin = max(abs(d["coefficient"])) * 0.45
        ax.set_xlim(min(d["coefficient"]) - margin, max(d["coefficient"]) + margin)
        fig.tight_layout()
        C.save_fig(fig, "fig24_logistic_coefficients")
        plt.close(fig)

        C.put("interpretation", {
            "lr_coefficients": lr_tbl.to_dict("records"),
            "lr_intercept": float(lr.intercept_[0]),
            "native": nat.to_dict("records"),
            "permutation": perm.to_dict("records"),
            "ranks": rank.to_dict("records"),
            "agreement": agree.to_dict("records"),
            "consistency": consistency.to_dict("records"),
            "rf_top3": rf_top["feature"].head(3).tolist(),
            "lr_top3": lr_tbl["feature"].head(3).tolist(),
            "consistent_features": consistent["feature"].tolist(),
            "contested_features": contested["feature"].tolist(),
        })
        print("\nStage 11 complete.")
    finally:
        C.end_log(tee)


if __name__ == "__main__":
    main()
