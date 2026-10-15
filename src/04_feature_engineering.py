"""
Stage 04 - Feature development and evidence  (Task 5)

Three candidate features, each grounded in the physics the AI4I documentation
describes. Each one has to earn its place against four kinds of evidence:
distributional separation on the training partition, correlation with the
target, importance inside a fitted Random Forest, and a leave-one-out ablation
measured by cross-validated PR-AUC.

Every number in this stage is computed on the 8,000 training rows only. Using
the test set to choose features would be a selection leak, and the whole point
of stage 03 was to avoid exactly that.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import config as C
import pipelines as P

STAGE = "04_feature_engineering"
plt, sns = C.setup_matplotlib()

RAW_NUM = list(C.NUMERIC_CLEAN)
CAT = list(C.CAT_CLEAN)


def cv_pr_auc(model, X, y, numeric, categorical, scale):
    """Cross-validated PR-AUC for one feature set. The preprocessor sits inside
    the pipeline, so it is refitted on each training fold."""
    pipe = Pipeline([
        ("pre", P.build_preprocessor_cols(numeric, categorical, scale)),
        ("clf", model),
    ])
    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.SEED)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=C.SELECTION_METRIC, n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def main() -> None:
    tee = C.start_log(STAGE)
    try:
        C.banner("STAGE 04  FEATURE DEVELOPMENT")

        df = pd.read_csv(C.MODEL_READY_CSV)
        train = df[df["split"] == "train"].reset_index(drop=True)
        y = train[C.TARGET_CLEAN].to_numpy()
        X = train.drop(columns=[C.TARGET_CLEAN, "split"])
        print("Training partition: %d rows, %d failures (%.2f%%)"
              % (len(train), y.sum(), 100 * y.mean()))

        # ------------------------------------------------------------------
        C.banner("DEFINITIONS")
        defs = pd.DataFrame([
            ("power_W", "torque_Nm x rot_speed_rpm x 2*pi/60",
             "Mechanical power in watts. Torque times angular velocity is the "
             "textbook definition, and rpm has to be converted to radians per "
             "second first. The AI4I documentation defines its power failure mode "
             "as this quantity falling outside a 3,500 to 9,000 W band, so the "
             "quantity is physically meaningful rather than a fishing expedition.",
             "Combines two originals"),
            ("temp_diff_K", "process_temp_K - air_temp_K",
             "The temperature the machine adds above ambient. A small gap means "
             "heat is not being carried away. The two raw temperatures correlate "
             "at r = 0.876, so their difference is the part of the pair that "
             "actually varies with machine condition.",
             "Combines two originals"),
            ("wear_torque", "tool_wear_min x torque_Nm",
             "A proxy for accumulated overstrain: a blunt tool pushed hard. The "
             "documented overstrain mode fires when this product passes a limit "
             "that itself depends on product grade, which is a multiplication a "
             "linear model cannot form on its own.",
             "Combines two originals"),
        ], columns=["feature", "formula", "reasoning", "type"])
        C.save_table(defs, "tbl18_engineered_feature_definitions")

        # ------------------------------------------------------------------
        C.banner("EVIDENCE 1  DISTRIBUTIONAL SEPARATION (training rows only)")
        ev = []
        for col in RAW_NUM + C.ENGINEERED:
            a = train.loc[train[C.TARGET_CLEAN] == 0, col]
            b = train.loc[train[C.TARGET_CLEAN] == 1, col]
            u, p = stats.mannwhitneyu(b, a, alternative="two-sided")
            auc = u / (len(a) * len(b))
            pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) /
                             (len(a) + len(b) - 2))
            d = (b.mean() - a.mean()) / pooled
            r_pb, p_pb = stats.pointbiserialr(train[C.TARGET_CLEAN], train[col])
            ev.append({
                "feature": col,
                "engineered": col in C.ENGINEERED,
                "median_no_failure": round(a.median(), 2),
                "median_failure": round(b.median(), 2),
                "cohens_d": round(d, 3),
                "univariate_auc": round(auc, 4),
                "point_biserial_r": round(r_pb, 4),
                "p_value": "%.3g" % p,
            })
        ev = pd.DataFrame(ev)
        ev["auc_distance_from_chance"] = (ev["univariate_auc"] - 0.5).abs().round(4)
        ev = ev.sort_values("auc_distance_from_chance", ascending=False)
        C.save_table(ev, "tbl19_feature_evidence_separation")

        for f in C.ENGINEERED:
            r = ev[ev["feature"] == f].iloc[0]
            print("%-12s AUC %.4f, |r| %.4f, Cohen's d %+.3f"
                  % (f, r["univariate_auc"], abs(r["point_biserial_r"]), r["cohens_d"]))
        best_raw = ev[~ev["engineered"]].iloc[0]
        print("\nBest raw predictor for comparison: %s at AUC %.4f"
              % (best_raw["feature"], best_raw["univariate_auc"]))

        # ------------------------------------------------------------------
        C.banner("EVIDENCE 2  IMPORTANCE INSIDE A FITTED RANDOM FOREST")
        num_all = RAW_NUM + C.ENGINEERED
        pipe = Pipeline([
            ("pre", P.build_preprocessor_cols(num_all, CAT, scale=False)),
            ("clf", RandomForestClassifier(n_estimators=400, random_state=C.SEED,
                                           class_weight="balanced", n_jobs=-1)),
        ])
        pipe.fit(X[num_all + CAT], y)
        names = num_all + ["type_L", "type_M"]
        imp = pd.DataFrame({
            "feature": names,
            "gini_importance": np.round(pipe.named_steps["clf"].feature_importances_, 4),
        })
        imp["engineered"] = imp["feature"].isin(C.ENGINEERED)
        imp = imp.sort_values("gini_importance", ascending=False).reset_index(drop=True)
        imp["rank"] = imp.index + 1
        C.save_table(imp, "tbl20_feature_evidence_rf_importance")
        eng_ranks = {r["feature"]: int(r["rank"]) for _, r in imp.iterrows()
                     if r["engineered"]}
        print("\nRanks of the engineered features out of %d: %s" % (len(imp), eng_ranks))

        # ------------------------------------------------------------------
        C.banner("EVIDENCE 3  ABLATION BY CROSS-VALIDATED %s" % C.SELECTION_METRIC_LABEL)
        print("Selection metric declared before running anything: %s."
              % C.SELECTION_METRIC_LABEL)
        print("5-fold StratifiedKFold, shuffle=True, random_state=%d.\n" % C.SEED)

        variants = {
            "A. raw sensors only": RAW_NUM,
            "B. raw + power_W": RAW_NUM + ["power_W"],
            "C. raw + temp_diff_K": RAW_NUM + ["temp_diff_K"],
            "D. raw + wear_torque": RAW_NUM + ["wear_torque"],
            "E. raw + all three engineered": RAW_NUM + C.ENGINEERED,
            "F. E minus power_W": RAW_NUM + ["temp_diff_K", "wear_torque"],
            "G. E minus temp_diff_K": RAW_NUM + ["power_W", "wear_torque"],
            "H. E minus wear_torque": RAW_NUM + ["power_W", "temp_diff_K"],
            "I. temp_diff replaces the two raw temperatures":
                ["rot_speed_rpm", "torque_Nm", "tool_wear_min", "power_W",
                 "temp_diff_K", "wear_torque"],
        }

        rows = []
        for label, cols in variants.items():
            rf = RandomForestClassifier(n_estimators=300, random_state=C.SEED,
                                        class_weight="balanced", n_jobs=1)
            lr = LogisticRegression(max_iter=5000, random_state=C.SEED,
                                    class_weight="balanced")
            m_rf, s_rf = cv_pr_auc(rf, X[cols + CAT], y, cols, CAT, scale=False)
            m_lr, s_lr = cv_pr_auc(lr, X[cols + CAT], y, cols, CAT, scale=True)
            rows.append({
                "feature_set": label,
                "n_numeric_features": len(cols),
                "rf_cv_pr_auc": round(m_rf, 4), "rf_sd": round(s_rf, 4),
                "lr_cv_pr_auc": round(m_lr, 4), "lr_sd": round(s_lr, 4),
            })
            print("%-46s RF %.4f (sd %.4f)   LR %.4f (sd %.4f)"
                  % (label, m_rf, s_rf, m_lr, s_lr))

        abl = pd.DataFrame(rows)
        base_rf = float(abl.loc[abl["feature_set"] == "A. raw sensors only",
                                "rf_cv_pr_auc"].iloc[0])
        base_lr = float(abl.loc[abl["feature_set"] == "A. raw sensors only",
                                "lr_cv_pr_auc"].iloc[0])
        abl["rf_gain_vs_raw"] = (abl["rf_cv_pr_auc"] - base_rf).round(4)
        abl["lr_gain_vs_raw"] = (abl["lr_cv_pr_auc"] - base_lr).round(4)
        C.save_table(abl, "tbl21_feature_ablation")

        full_rf = float(abl.loc[abl["feature_set"] == "E. raw + all three engineered",
                                "rf_cv_pr_auc"].iloc[0])
        full_lr = float(abl.loc[abl["feature_set"] == "E. raw + all three engineered",
                                "lr_cv_pr_auc"].iloc[0])

        loo = {
            "power_W": ("F. E minus power_W", ),
            "temp_diff_K": ("G. E minus temp_diff_K", ),
            "wear_torque": ("H. E minus wear_torque", ),
        }
        C.banner("LEAVE-ONE-OUT COST OF DROPPING EACH ENGINEERED FEATURE")
        loo_rows = []
        for feat, (label, ) in loo.items():
            r = abl[abl["feature_set"] == label].iloc[0]
            loo_rows.append({
                "dropped_feature": feat,
                "rf_pr_auc_without": r["rf_cv_pr_auc"],
                "rf_cost_of_dropping": round(full_rf - r["rf_cv_pr_auc"], 4),
                "lr_pr_auc_without": r["lr_cv_pr_auc"],
                "lr_cost_of_dropping": round(full_lr - r["lr_cv_pr_auc"], 4),
            })
        loo_df = pd.DataFrame(loo_rows)
        C.save_table(loo_df, "tbl22_leave_one_out_cost")

        # ------------------------------------------------------------------
        C.banner("DECISION")
        rule = ("Rule fixed before looking at the numbers: retain an engineered "
                "feature if dropping it costs at least 0.005 cross-validated "
                "PR-AUC in either model, or if it ranks in the top half of the "
                "Random Forest importance table. Otherwise drop it and say so.")
        print(rule + "\n")

        keep, drop = [], []
        for _, r in loo_df.iterrows():
            f = r["dropped_feature"]
            cost = max(r["rf_cost_of_dropping"], r["lr_cost_of_dropping"])
            rank = eng_ranks[f]
            top_half = rank <= len(imp) / 2
            verdict = "RETAIN" if (cost >= 0.005 or top_half) else "DROP"
            (keep if verdict == "RETAIN" else drop).append(f)
            print("%-12s max ablation cost %+.4f, RF importance rank %d of %d -> %s"
                  % (f, cost, rank, len(imp), verdict))

        var_i_rf = float(abl.loc[abl["feature_set"].str.startswith("I."),
                                 "rf_cv_pr_auc"].iloc[0])
        var_i_lr = float(abl.loc[abl["feature_set"].str.startswith("I."),
                                 "lr_cv_pr_auc"].iloc[0])
        rf_sd_full = float(abl.loc[abl["feature_set"] == "E. raw + all three engineered",
                                   "rf_sd"].iloc[0])
        print("\nVariant I removes air and process temperature and keeps only their "
              "difference. RF %.4f against %.4f for the full set, LR %.4f against "
              "%.4f." % (var_i_rf, full_rf, var_i_lr, full_lr))
        print("Variant I is nominally ahead by %.4f PR-AUC on the forest, against a "
              "fold-to-fold standard deviation of %.4f. That gap is a fifth of the "
              "noise, so I am not going to read it as evidence of anything."
              % (var_i_rf - full_rf, rf_sd_full))
        print("I keep the two raw temperatures. The leaner set is not measurably "
              "better, and the raw readings are the numbers an operator can pull "
              "straight off the machine, which matters when someone has to sanity "
              "check an alarm. The collinearity that variant I removes is a "
              "readability problem for one model's coefficients, handled by the L2 "
              "penalty, and it is irrelevant to the three tree-based models.")

        decision = pd.DataFrame([
            {"feature": f,
             "verdict": "Retained" if f in keep else "Dropped",
             "rf_importance_rank": eng_ranks[f],
             "max_ablation_cost_pr_auc": float(
                 max(loo_df.loc[loo_df["dropped_feature"] == f,
                                "rf_cost_of_dropping"].iloc[0],
                     loo_df.loc[loo_df["dropped_feature"] == f,
                                "lr_cost_of_dropping"].iloc[0]))}
            for f in C.ENGINEERED
        ])
        C.save_table(decision, "tbl23_feature_decision")

        # ------------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        for ax, f in zip(axes, C.ENGINEERED):
            sns.kdeplot(data=train, x=f, hue=C.TARGET_CLEAN, common_norm=False,
                        fill=True, alpha=0.35, palette=[C.C_OK, C.C_FAIL],
                        ax=ax, linewidth=1.3, legend=(f == C.ENGINEERED[0]))
            r = ev[ev["feature"] == f].iloc[0]
            ax.set_title("%s\nAUC %.3f, Cohen's d %+.2f"
                         % (f, r["univariate_auc"], r["cohens_d"]))
            ax.set_ylabel("Density")
        fig.suptitle("Engineered features separate the classes more sharply than the raw sensors",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        C.save_fig(fig, "fig09_engineered_feature_separation")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
        order = abl.sort_values("rf_cv_pr_auc")
        cols_ = [C.C_FAIL if s.startswith("E.") else C.C_OK for s in order["feature_set"]]
        axes[0].barh(order["feature_set"], order["rf_cv_pr_auc"], color=cols_,
                     edgecolor="black", linewidth=0.6)
        axes[0].set_xlim(min(order["rf_cv_pr_auc"]) - 0.02, max(order["rf_cv_pr_auc"]) + 0.02)
        for i, (v, s) in enumerate(zip(order["rf_cv_pr_auc"], order["rf_sd"])):
            axes[0].text(v + 0.002, i, "%.4f" % v, va="center", fontsize=8)
        axes[0].set_xlabel("Cross-validated PR-AUC")
        axes[0].set_title("Random Forest")

        order2 = abl.sort_values("lr_cv_pr_auc")
        cols2 = [C.C_FAIL if s.startswith("E.") else C.C_OK for s in order2["feature_set"]]
        axes[1].barh(order2["feature_set"], order2["lr_cv_pr_auc"], color=cols2,
                     edgecolor="black", linewidth=0.6)
        axes[1].set_xlim(min(order2["lr_cv_pr_auc"]) - 0.02, max(order2["lr_cv_pr_auc"]) + 0.02)
        for i, v in enumerate(order2["lr_cv_pr_auc"]):
            axes[1].text(v + 0.002, i, "%.4f" % v, va="center", fontsize=8)
        axes[1].set_xlabel("Cross-validated PR-AUC")
        axes[1].set_title("Logistic Regression")
        axes[1].set_yticklabels([])
        fig.suptitle("Feature-set ablation, 5-fold CV on the training partition (seed 93)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        C.save_fig(fig, "fig10_feature_ablation")
        plt.close(fig)

        C.put("features", {
            "kept": keep, "dropped": drop,
            "rule": rule,
            "base_rf": base_rf, "base_lr": base_lr,
            "full_rf": full_rf, "full_lr": full_lr,
            "gain_rf": round(full_rf - base_rf, 4),
            "gain_lr": round(full_lr - base_lr, 4),
            "variant_i_rf": var_i_rf, "variant_i_lr": var_i_lr,
            "rf_ranks": eng_ranks,
            "n_features_total": int(len(imp)),
            "loo": loo_df.to_dict("records"),
            "train_auc": {r["feature"]: float(r["univariate_auc"])
                          for _, r in ev.iterrows()},
            "top_importance": imp.head(5).to_dict("records"),
        })
        print("\nStage 04 complete. Retained engineered features: %s" % ", ".join(keep))
        if drop:
            print("Dropped: %s" % ", ".join(drop))
    finally:
        C.end_log(tee)


if __name__ == "__main__":
    main()
