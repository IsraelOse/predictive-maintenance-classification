"""
Stage 02 - Exploratory investigation  (Task 3)

Six findings, four of them involving more than one variable at a time. Every
finding ends with a note on what it changes about the modelling approach, and
every number printed here is reused verbatim in the report.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

import config as C

STAGE = "02_eda"
plt, sns = C.setup_matplotlib()


def load() -> pd.DataFrame:
    df = pd.read_csv(C.RAW_CSV)
    df = df.rename(columns=C.RENAME)
    # Derived quantities used for exploration only; they are rebuilt inside the
    # modelling pipeline later so that nothing here leaks into the folds.
    df["temp_diff_K"] = df["process_temp_K"] - df["air_temp_K"]
    df["power_W"] = df["torque_Nm"] * df["rot_speed_rpm"] * 2 * np.pi / 60.0
    df["wear_torque"] = df["tool_wear_min"] * df["torque_Nm"]
    return df


def main() -> None:
    tee = C.start_log(STAGE)
    try:
        C.banner("STAGE 02  EXPLORATORY INVESTIGATION")
        df = load()
        fail = df[df[C.TARGET_CLEAN] == 1]
        ok = df[df[C.TARGET_CLEAN] == 0]
        n_fail = len(fail)
        print("Rows %d, failures %d (%.2f%%)" % (len(df), n_fail, 100 * n_fail / len(df)))

        results = {"n_fail": n_fail}

        # ------------------------------------------------------------------
        # Figure 1: target distribution
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        vc = df[C.TARGET_CLEAN].value_counts().sort_index()
        axes[0].bar(["No failure (0)", "Failure (1)"], vc.values,
                    color=[C.C_OK, C.C_FAIL], edgecolor="black", linewidth=0.6)
        for i, v in enumerate(vc.values):
            axes[0].text(i, v + 150, "%d" % v, ha="center", fontweight="bold")
        axes[0].set_ylabel("Number of records")
        axes[0].set_title("Absolute counts")
        axes[0].set_ylim(0, vc.max() * 1.15)

        axes[1].bar(["No failure (0)", "Failure (1)"], 100 * vc.values / len(df),
                    color=[C.C_OK, C.C_FAIL], edgecolor="black", linewidth=0.6)
        for i, v in enumerate(100 * vc.values / len(df)):
            axes[1].text(i, v + 1.5, "%.2f%%" % v, ha="center", fontweight="bold")
        axes[1].set_ylabel("Share of records (%)")
        axes[1].set_title("Relative frequency")
        axes[1].set_ylim(0, 108)
        fig.suptitle("Target distribution: 339 failures in 10,000 records (3.39%)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        C.save_fig(fig, "fig01_target_distribution")
        plt.close(fig)

        # ------------------------------------------------------------------
        # FINDING 1 (univariate): which single sensors separate the classes
        # ------------------------------------------------------------------
        C.banner("FINDING 1  Single-sensor separation is weak for every raw predictor")
        rows = []
        for col in C.NUMERIC_CLEAN + ["temp_diff_K", "power_W", "wear_torque"]:
            a, b = ok[col], fail[col]
            u, p = stats.mannwhitneyu(b, a, alternative="two-sided")
            pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) /
                             (len(a) + len(b) - 2))
            d = (b.mean() - a.mean()) / pooled
            # single-variable ranking power
            auc = u / (len(a) * len(b))
            rows.append({
                "variable": col,
                "mean_no_failure": round(a.mean(), 3),
                "mean_failure": round(b.mean(), 3),
                "median_no_failure": round(a.median(), 3),
                "median_failure": round(b.median(), 3),
                "cohens_d": round(d, 3),
                "univariate_auc": round(auc, 4),
                "mannwhitney_p": ("%.3g" % p),
            })
        sep = pd.DataFrame(rows).sort_values("univariate_auc",
                                             key=lambda s: (s - 0.5).abs(),
                                             ascending=False)
        C.save_table(sep, "tbl07_univariate_separation")
        best = sep.iloc[0]
        print("\nStrongest single predictor: %s, AUC %.3f, Cohen's d %.2f"
              % (best["variable"], best["univariate_auc"], best["cohens_d"]))
        print("Air and process temperature barely move between the classes "
              "(AUC near 0.5), so no single sensor is a usable alarm on its own.")
        print("WHAT IT CHANGES: a single-threshold rule is not viable. The models "
              "have to combine sensors, which argues for keeping all five raw "
              "readings and for trying non-linear learners alongside logistic regression.")
        results["best_univariate"] = {
            "variable": str(best["variable"]),
            "auc": float(best["univariate_auc"]),
            "d": float(best["cohens_d"]),
        }
        results["auc_air"] = float(sep.loc[sep["variable"] == "air_temp_K",
                                           "univariate_auc"].iloc[0])
        results["auc_torque"] = float(sep.loc[sep["variable"] == "torque_Nm",
                                              "univariate_auc"].iloc[0])
        results["auc_wear"] = float(sep.loc[sep["variable"] == "tool_wear_min",
                                            "univariate_auc"].iloc[0])
        results["auc_power"] = float(sep.loc[sep["variable"] == "power_W",
                                             "univariate_auc"].iloc[0])
        results["auc_tempdiff"] = float(sep.loc[sep["variable"] == "temp_diff_K",
                                                "univariate_auc"].iloc[0])
        results["auc_weartorque"] = float(sep.loc[sep["variable"] == "wear_torque",
                                                  "univariate_auc"].iloc[0])

        fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
        for ax, col in zip(axes.ravel(), C.NUMERIC_CLEAN + ["temp_diff_K"]):
            sns.kdeplot(data=df, x=col, hue=C.TARGET_CLEAN, common_norm=False,
                        fill=True, alpha=0.35, palette=[C.C_OK, C.C_FAIL],
                        ax=ax, linewidth=1.3, legend=(col == C.NUMERIC_CLEAN[0]))
            ax.set_title(col)
            ax.set_ylabel("Density")
        axes.ravel()[-1].set_visible(True)
        fig.suptitle("Conditional distributions of each predictor by failure status "
                     "(densities normalised within class)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        C.save_fig(fig, "fig02_conditional_distributions")
        plt.close(fig)

        # ------------------------------------------------------------------
        # FINDING 2 (multivariate): correlation structure
        # ------------------------------------------------------------------
        C.banner("FINDING 2  Air and process temperature are near-collinear")
        corr = df[C.NUMERIC_CLEAN + [C.TARGET_CLEAN]].corr(method="pearson")
        C.save_table(corr.round(3).reset_index().rename(columns={"index": "variable"}),
                     "tbl08_correlation_matrix")
        r_temp = corr.loc["air_temp_K", "process_temp_K"]
        r_tq_rs = corr.loc["torque_Nm", "rot_speed_rpm"]
        print("\nPearson r(air, process) = %.3f" % r_temp)
        print("Pearson r(torque, rotational speed) = %.3f" % r_tq_rs)
        vifs = []
        Xn = df[C.NUMERIC_CLEAN].to_numpy(float)
        for j, col in enumerate(C.NUMERIC_CLEAN):
            other = np.delete(Xn, j, axis=1)
            other = np.column_stack([np.ones(len(other)), other])
            beta, *_ = np.linalg.lstsq(other, Xn[:, j], rcond=None)
            resid = Xn[:, j] - other @ beta
            r2 = 1 - resid.var() / Xn[:, j].var()
            vifs.append({"variable": col, "r_squared_on_others": round(r2, 4),
                         "VIF": round(1 / (1 - r2), 2)})
        vif_df = pd.DataFrame(vifs)
        C.save_table(vif_df, "tbl09_multicollinearity_vif")
        print("WHAT IT CHANGES: the two temperatures carry almost the same signal, "
              "so their logistic-regression coefficients will be unstable and hard "
              "to read. Replacing the pair with their difference is the sensible "
              "engineered response, and it is why regularisation stays switched on.")
        results["r_air_process"] = round(float(r_temp), 3)
        results["r_torque_speed"] = round(float(r_tq_rs), 3)
        results["vif_air"] = float(vif_df.loc[vif_df["variable"] == "air_temp_K", "VIF"].iloc[0])
        results["vif_process"] = float(vif_df.loc[vif_df["variable"] == "process_temp_K", "VIF"].iloc[0])

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, vmin=-1, vmax=1, square=True, linewidths=0.6,
                    cbar_kws={"shrink": 0.75, "label": "Pearson r"}, ax=ax)
        ax.set_title("Correlation among raw predictors and the target")
        fig.tight_layout()
        C.save_fig(fig, "fig03_correlation_heatmap")
        plt.close(fig)

        # ------------------------------------------------------------------
        # FINDING 3 (multivariate): torque vs rotational speed
        # ------------------------------------------------------------------
        C.banner("FINDING 3  Failures sit in two opposite corners of the torque-speed plane")
        hi_tq_lo_rs = int(((fail["torque_Nm"] >= 60) & (fail["rot_speed_rpm"] <= 1500)).sum())
        lo_tq_hi_rs = int(((fail["torque_Nm"] <= 25) & (fail["rot_speed_rpm"] >= 1800)).sum())
        print("Failures with torque >= 60 Nm and speed <= 1500 rpm : %d of %d (%.1f%%)"
              % (hi_tq_lo_rs, n_fail, 100 * hi_tq_lo_rs / n_fail))
        print("Failures with torque <= 25 Nm and speed >= 1800 rpm : %d of %d (%.1f%%)"
              % (lo_tq_hi_rs, n_fail, 100 * lo_tq_hi_rs / n_fail))
        base_hi = int(((ok["torque_Nm"] >= 60) & (ok["rot_speed_rpm"] <= 1500)).sum())
        base_lo = int(((ok["torque_Nm"] <= 25) & (ok["rot_speed_rpm"] >= 1800)).sum())
        print("The same two corners hold %d and %d healthy records respectively."
              % (base_hi, base_lo))
        print("WHAT IT CHANGES: risk is high at both ends of the torque-speed "
              "trade-off and low in the middle. That is a non-monotonic pattern in "
              "the raw coordinates, which a linear decision boundary cannot express. "
              "It is the single strongest argument for the tree ensembles, and it "
              "tells me exactly which engineered feature to build.")
        results["fail_hi_torque_lo_speed"] = hi_tq_lo_rs
        results["fail_lo_torque_hi_speed"] = lo_tq_hi_rs
        results["ok_hi_torque_lo_speed"] = base_hi
        results["ok_lo_torque_hi_speed"] = base_lo

        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        ax.scatter(ok["rot_speed_rpm"], ok["torque_Nm"], s=7, alpha=0.18,
                   color=C.C_OK, label="No failure (n=9,661)", rasterized=True)
        ax.scatter(fail["rot_speed_rpm"], fail["torque_Nm"], s=26, alpha=0.85,
                   color=C.C_FAIL, edgecolor="black", linewidth=0.3,
                   label="Failure (n=339)")
        rs_grid = np.linspace(df["rot_speed_rpm"].min(), df["rot_speed_rpm"].max(), 300)
        for pw, style in [(3500, "--"), (9000, "-.")]:
            ax.plot(rs_grid, pw / (rs_grid * 2 * np.pi / 60), style, color="black",
                    linewidth=1.2, label="%d W constant-power line" % pw)
        ax.set_xlim(df["rot_speed_rpm"].min() - 30, 2400)
        ax.set_ylim(0, 80)
        ax.set_xlabel("Rotational speed (rpm)")
        ax.set_ylabel("Torque (Nm)")
        ax.set_title("Failures concentrate outside a central power band")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        C.save_fig(fig, "fig04_torque_vs_speed")
        plt.close(fig)

        # power decile failure rate: the U shape made explicit
        df["power_decile"] = pd.qcut(df["power_W"], 10, labels=False) + 1
        pw = (df.groupby("power_decile")
                .agg(n=("power_W", "size"),
                     failures=(C.TARGET_CLEAN, "sum"),
                     power_min=("power_W", "min"),
                     power_max=("power_W", "max")))
        pw["failure_rate_pct"] = (100 * pw["failures"] / pw["n"]).round(2)
        pw = pw.round(1).reset_index()
        C.save_table(pw, "tbl10_failure_rate_by_power_decile")
        d1 = float(pw.loc[pw["power_decile"] == 1, "failure_rate_pct"].iloc[0])
        d10 = float(pw.loc[pw["power_decile"] == 10, "failure_rate_pct"].iloc[0])
        mid = float(pw.loc[pw["power_decile"].between(4, 7), "failures"].sum() /
                    pw.loc[pw["power_decile"].between(4, 7), "n"].sum() * 100)
        print("\nFailure rate, lowest power decile  : %.2f%%" % d1)
        print("Failure rate, deciles 4 to 7 (middle): %.2f%%" % mid)
        print("Failure rate, highest power decile : %.2f%%" % d10)
        results["power_d1_rate"] = d1
        results["power_d10_rate"] = d10
        results["power_mid_rate"] = round(mid, 2)

        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        bars = ax.bar(pw["power_decile"], pw["failure_rate_pct"],
                      color=[C.C_FAIL if v > 5 else C.C_OK for v in pw["failure_rate_pct"]],
                      edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, pw["failure_rate_pct"]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, "%.1f" % v,
                    ha="center", fontsize=8)
        ax.axhline(100 * n_fail / len(df), color="black", linestyle=":",
                   label="Overall rate 3.39%")
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Mechanical power decile (1 = lowest)")
        ax.set_ylabel("Failure rate (%)")
        ax.set_title("Failure risk is U-shaped in mechanical power")
        ax.legend()
        fig.tight_layout()
        C.save_fig(fig, "fig05_failure_rate_by_power_decile")
        plt.close(fig)

        # ------------------------------------------------------------------
        # FINDING 4 (multivariate): heat dissipation needs two variables at once
        # ------------------------------------------------------------------
        C.banner("FINDING 4  A small temperature gap is only dangerous at low speed")
        cond = (df["temp_diff_K"] < 8.6) & (df["rot_speed_rpm"] < 1380)
        only_gap = (df["temp_diff_K"] < 8.6) & (df["rot_speed_rpm"] >= 1380)
        only_slow = (df["temp_diff_K"] >= 8.6) & (df["rot_speed_rpm"] < 1380)
        for name, m in [("gap < 8.6 K AND speed < 1380 rpm", cond),
                        ("gap < 8.6 K only", only_gap),
                        ("speed < 1380 rpm only", only_slow),
                        ("neither", ~(cond | only_gap | only_slow))]:
            print("%-34s n=%5d  failures=%4d  rate=%6.2f%%"
                  % (name, int(m.sum()), int(df.loc[m, C.TARGET_CLEAN].sum()),
                     100 * df.loc[m, C.TARGET_CLEAN].mean()))
        joint = pd.DataFrame({
            "condition": ["gap < 8.6 K AND speed < 1380 rpm", "gap < 8.6 K only",
                          "speed < 1380 rpm only", "neither"],
            "n_rows": [int(m.sum()) for m in (cond, only_gap, only_slow,
                                              ~(cond | only_gap | only_slow))],
            "n_failures": [int(df.loc[m, C.TARGET_CLEAN].sum())
                           for m in (cond, only_gap, only_slow,
                                     ~(cond | only_gap | only_slow))],
        })
        joint["failure_rate_pct"] = (100 * joint["n_failures"] / joint["n_rows"]).round(2)
        C.save_table(joint, "tbl11_heat_dissipation_interaction")
        print("WHAT IT CHANGES: neither condition alone is alarming, and together "
              "they are. An additive model has to approximate that with main effects "
              "it does not really have, so I expect logistic regression to lose "
              "recall here unless the interaction is handed to it explicitly.")
        results["hdf_joint_rate"] = float(joint.loc[0, "failure_rate_pct"])
        results["hdf_joint_n"] = int(joint.loc[0, "n_rows"])
        results["hdf_joint_fail"] = int(joint.loc[0, "n_failures"])
        results["hdf_gap_only_rate"] = float(joint.loc[1, "failure_rate_pct"])
        results["hdf_slow_only_rate"] = float(joint.loc[2, "failure_rate_pct"])

        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.scatter(ok["rot_speed_rpm"], ok["temp_diff_K"], s=7, alpha=0.18,
                   color=C.C_OK, label="No failure", rasterized=True)
        ax.scatter(fail["rot_speed_rpm"], fail["temp_diff_K"], s=26, alpha=0.85,
                   color=C.C_FAIL, edgecolor="black", linewidth=0.3, label="Failure")
        ax.axhline(8.6, color="black", linestyle="--", linewidth=1.1)
        ax.axvline(1380, color="black", linestyle="--", linewidth=1.1)
        ax.add_patch(plt.Rectangle((df["rot_speed_rpm"].min() - 50, 0), 1380 -
                                   df["rot_speed_rpm"].min() + 50, 8.6,
                                   facecolor="red", alpha=0.08, zorder=0))
        ax.text(1200, 7.4, "joint risk region\n%.1f%% failure rate"
                % results["hdf_joint_rate"], fontsize=8.5, ha="center")
        ax.set_xlim(df["rot_speed_rpm"].min() - 50, 2400)
        ax.set_ylim(6.5, 13)
        ax.set_xlabel("Rotational speed (rpm)")
        ax.set_ylabel("Process temperature minus air temperature (K)")
        ax.set_title("Heat dissipation risk needs a small gap and a slow spindle together")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        C.save_fig(fig, "fig06_tempgap_vs_speed")
        plt.close(fig)

        # ------------------------------------------------------------------
        # FINDING 5 (multivariate): wear x torque, and it moves with product Type
        # ------------------------------------------------------------------
        C.banner("FINDING 5  Overstrain depends on wear times torque, and the limit moves with Type")
        by_type = (df.groupby("type")
                     .agg(n=("type", "size"), failures=(C.TARGET_CLEAN, "sum"),
                          mean_wear_torque=("wear_torque", "mean")))
        by_type["mean_wear_torque"] = by_type["mean_wear_torque"].round(1)
        by_type["failure_rate_pct"] = (100 * by_type["failures"] / by_type["n"]).round(2)
        by_type = by_type.reset_index()
        C.save_table(by_type, "tbl12_failure_rate_by_type")

        wt_rows = []
        for t, lim in [("L", 11000), ("M", 12000), ("H", 13000)]:
            sub = df[df["type"] == t]
            above = sub[sub["wear_torque"] > lim]
            below = sub[sub["wear_torque"] <= lim]
            wt_rows.append({
                "type": t, "documented_limit": lim,
                "n_above": len(above), "failures_above": int(above[C.TARGET_CLEAN].sum()),
                "rate_above_pct": round(100 * above[C.TARGET_CLEAN].mean(), 2) if len(above) else np.nan,
                "n_below": len(below), "failures_below": int(below[C.TARGET_CLEAN].sum()),
                "rate_below_pct": round(100 * below[C.TARGET_CLEAN].mean(), 2),
            })
        wt = pd.DataFrame(wt_rows)
        C.save_table(wt, "tbl13_overstrain_threshold_by_type")
        print("\nWHAT IT CHANGES: the dangerous value of wear times torque is not "
              "one number, it shifts by product grade. A model that sees Type and "
              "the product term can carve that out; one that sees wear and torque "
              "separately has to rediscover a multiplication it was never shown. "
              "This is why I build the product explicitly and keep Type as a predictor.")
        results["type_rates"] = {r["type"]: float(r["failure_rate_pct"])
                                 for _, r in by_type.iterrows()}
        results["type_counts"] = {r["type"]: int(r["n"]) for _, r in by_type.iterrows()}
        results["overstrain"] = wt.to_dict("records")

        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), sharey=True)
        for ax, (t, lim) in zip(axes, [("L", 11000), ("M", 12000), ("H", 13000)]):
            sub = df[df["type"] == t]
            s_ok, s_f = sub[sub[C.TARGET_CLEAN] == 0], sub[sub[C.TARGET_CLEAN] == 1]
            ax.scatter(s_ok["tool_wear_min"], s_ok["torque_Nm"], s=7, alpha=0.2,
                       color=C.C_OK, rasterized=True)
            ax.scatter(s_f["tool_wear_min"], s_f["torque_Nm"], s=28, alpha=0.9,
                       color=C.C_FAIL, edgecolor="black", linewidth=0.3)
            w = np.linspace(1, 260, 300)
            ax.plot(w, lim / w, "k--", linewidth=1.2)
            ax.set_xlim(0, 260)
            ax.set_ylim(0, 80)
            ax.set_xlabel("Tool wear (min)")
            ax.set_title("Type %s, %d rows, %.2f%% failures\ndashed line: wear x torque = %d"
                         % (t, len(sub), 100 * sub[C.TARGET_CLEAN].mean(), lim))
        axes[0].set_ylabel("Torque (Nm)")
        fig.suptitle("The overstrain boundary is a hyperbola whose position depends on product grade",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        C.save_fig(fig, "fig07_wear_torque_by_type")
        plt.close(fig)

        # ------------------------------------------------------------------
        # FINDING 6: tool wear is a staircase, not a slope
        # ------------------------------------------------------------------
        C.banner("FINDING 6  Tool wear only bites above roughly 200 minutes")
        bins = [0, 50, 100, 150, 200, 220, 240, 260]
        df["wear_bin"] = pd.cut(df["tool_wear_min"], bins=bins, right=False,
                                include_lowest=True)
        wb = (df.groupby("wear_bin", observed=True)
                .agg(n=("tool_wear_min", "size"), failures=(C.TARGET_CLEAN, "sum")))
        wb["failure_rate_pct"] = (100 * wb["failures"] / wb["n"]).round(2)
        wb = wb.reset_index()
        wb["wear_bin"] = wb["wear_bin"].astype(str)
        C.save_table(wb, "tbl14_failure_rate_by_wear_bin")
        lo = float(wb.loc[wb["wear_bin"].str.startswith("[0"), "failure_rate_pct"].iloc[0])
        hi = float(wb["failure_rate_pct"].iloc[-1])
        print("\nFailure rate below 50 minutes of wear : %.2f%%" % lo)
        print("Failure rate in the 240 to 260 band   : %.2f%%" % hi)
        print("WHAT IT CHANGES: the effect is a step, not a gradient. Logistic "
              "regression fits a single slope through that staircase and will "
              "under-react at the top end, while a tree can place a split exactly "
              "where the step is. It also tells me that scaling tool wear will not "
              "fix the shape, only the units.")
        results["wear_low_rate"] = lo
        results["wear_high_rate"] = hi

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        axes[0].bar(range(len(wb)), wb["failure_rate_pct"], color=C.C_FAIL,
                    edgecolor="black", linewidth=0.6, alpha=0.85)
        axes[0].set_xticks(range(len(wb)))
        axes[0].set_xticklabels(wb["wear_bin"], rotation=30, ha="right", fontsize=8)
        axes[0].axhline(100 * n_fail / len(df), color="black", linestyle=":",
                        label="Overall 3.39%")
        axes[0].set_xlabel("Tool wear band (minutes)")
        axes[0].set_ylabel("Failure rate (%)")
        axes[0].set_title("Failure rate by tool-wear band")
        axes[0].legend(fontsize=8)

        tp = (df.groupby("type")
                .agg(rate=(C.TARGET_CLEAN, "mean"), n=("type", "size"))
                .reindex(["L", "M", "H"]).reset_index())
        b = axes[1].bar(tp["type"], 100 * tp["rate"], color=C.PALETTE[:3],
                        edgecolor="black", linewidth=0.6)
        for bb, r, nn in zip(b, 100 * tp["rate"], tp["n"]):
            axes[1].text(bb.get_x() + bb.get_width() / 2, r + 0.06,
                         "%.2f%%\n(n=%d)" % (r, nn), ha="center", fontsize=8.5)
        axes[1].axhline(100 * n_fail / len(df), color="black", linestyle=":",
                        label="Overall 3.39%")
        axes[1].set_xlabel("Product quality grade")
        axes[1].set_ylabel("Failure rate (%)")
        axes[1].set_ylim(0, max(100 * tp["rate"]) * 1.35)
        axes[1].set_title("Failure rate by product Type")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        C.save_fig(fig, "fig08_wear_bands_and_type")
        plt.close(fig)

        # ------------------------------------------------------------------
        # descriptive table by class, used in the report
        # ------------------------------------------------------------------
        grp = df.groupby(C.TARGET_CLEAN)[C.NUMERIC_CLEAN + ["temp_diff_K", "power_W",
                                                            "wear_torque"]]
        stats_tbl = grp.agg(["mean", "std", "median"]).round(2).T.reset_index()
        stats_tbl.columns = ["variable", "statistic", "no_failure", "failure"]
        C.save_table(stats_tbl, "tbl15_descriptives_by_class")

        C.put("eda", results)
        print("\nStage 02 complete.")
    finally:
        C.end_log(tee)


if __name__ == "__main__":
    main()
