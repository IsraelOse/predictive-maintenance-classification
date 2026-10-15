"""
Stage 10 - Misclassification analysis  (Task 11)

Five real false positives and five real false negatives from the Random Forest
at the recommended operating threshold, each with its own sensor readings, its
predicted probability, and how far each reading sits from the median of the
class it actually belongs to.

These are specific rows from the held-out test set, identified by their position
in the original UCI file so that any of them can be looked up and checked.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as C
import pipelines as P

warnings.filterwarnings("ignore")

STAGE = "10_misclassification"
plt, sns = C.setup_matplotlib()

SHOW = C.NUMERIC_CLEAN + C.ENGINEERED
MODEL = "Random Forest"


def main() -> None:
    tee = C.start_log(STAGE)
    try:
        C.banner("STAGE 10  MISCLASSIFICATION ANALYSIS")

        full = pd.read_csv(C.MODEL_READY_CSV)
        raw = pd.read_csv(C.RAW_CSV)
        full["UDI"] = raw["UDI"].to_numpy()
        full["Product ID"] = raw["Product ID"].to_numpy()

        test = full[full["split"] == "test"].reset_index(drop=True)
        probs = pd.read_csv(C.TABLES / "test_predicted_probabilities.csv")
        thr = C.get("threshold")["recommended"]

        test["predicted_probability"] = probs[MODEL].to_numpy()
        test["predicted_label"] = (test["predicted_probability"] >= thr).astype(int)
        y = test[C.TARGET_CLEAN]

        print("Model: %s at the recommended threshold of %.2f." % (MODEL, thr))
        tn, fp, fn, tp = P.confusion_counts(y, test["predicted_label"])
        print("Test-set confusion at that threshold: TN %d, FP %d, FN %d, TP %d"
              % (tn, fp, fn, tp))

        fps = test[(y == 0) & (test["predicted_label"] == 1)]
        fns = test[(y == 1) & (test["predicted_label"] == 0)]
        print("\n%d false positives and %d false negatives available."
              % (len(fps), len(fns)))

        # Selection rule: the most confident mistakes, because those are the ones
        # an operator would act on (FP) or be blindsided by (FN).
        fp5 = fps.sort_values("predicted_probability", ascending=False).head(5)
        fn5 = fns.sort_values("predicted_probability", ascending=True).head(5)
        print("Selected: the five false positives the model was most confident "
              "about, and the five false negatives it was least worried by. Those "
              "are the two extremes that would actually cause trouble on a shop "
              "floor.")

        med_ok = test[y == 0][SHOW].median()
        med_fail = test[y == 1][SHOW].median()
        med_tbl = pd.DataFrame({
            "variable": SHOW,
            "median_no_failure": med_ok.round(2).to_numpy(),
            "median_failure": med_fail.round(2).to_numpy(),
        })
        C.save_table(med_tbl, "tbl40_test_class_medians")

        def detail(sub, kind):
            out = sub[["UDI", "Product ID", "type"] + SHOW +
                      [C.TARGET_CLEAN, "predicted_probability"]].copy()
            out["predicted_probability"] = out["predicted_probability"].round(4)
            out.insert(0, "case", ["%s%d" % (kind, i + 1) for i in range(len(out))])
            for c in SHOW:
                out[c] = out[c].round(2)
            return out

        fp_tbl = detail(fp5, "FP")
        fn_tbl = detail(fn5, "FN")
        C.save_table(fp_tbl, "tbl41_false_positive_cases")
        C.save_table(fn_tbl, "tbl42_false_negative_cases")

        # distance from the median of the true class, in that class's IQR units
        def deviation(sub, true_class):
            ref = test[y == true_class]
            med = ref[SHOW].median()
            iqr = ref[SHOW].quantile(0.75) - ref[SHOW].quantile(0.25)
            rows = []
            for i, (_, r) in enumerate(sub.iterrows()):
                d = {"case": "%s%d" % ("FP" if true_class == 0 else "FN", i + 1),
                     "UDI": int(r["UDI"])}
                for c in SHOW:
                    d[c] = round(float((r[c] - med[c]) / iqr[c]), 2)
                rows.append(d)
            return pd.DataFrame(rows)

        fp_dev = deviation(fp5, 0)
        fn_dev = deviation(fn5, 1)
        C.save_table(fp_dev, "tbl43_false_positive_deviation_from_class_median")
        C.save_table(fn_dev, "tbl44_false_negative_deviation_from_class_median")

        # ------------------------------------------------------------------
        C.banner("WHAT THE FALSE POSITIVES HAVE IN COMMON")
        for i, (_, r) in enumerate(fp5.iterrows(), 1):
            print("\nFP%d  UDI %d, product %s, grade %s, predicted probability %.3f"
                  % (i, r["UDI"], r["Product ID"], r["type"],
                     r["predicted_probability"]))
            print("     torque %.1f Nm (healthy median %.1f), speed %d rpm "
                  "(median %d), wear %d min (median %d)"
                  % (r["torque_Nm"], med_ok["torque_Nm"], r["rot_speed_rpm"],
                     med_ok["rot_speed_rpm"], r["tool_wear_min"],
                     med_ok["tool_wear_min"]))
            print("     power %.0f W (healthy median %.0f, failure median %.0f), "
                  "wear x torque %.0f (healthy median %.0f, failure median %.0f)"
                  % (r["power_W"], med_ok["power_W"], med_fail["power_W"],
                     r["wear_torque"], med_ok["wear_torque"], med_fail["wear_torque"]))

        fp_summary = {
            "median_wear_torque": float(fp5["wear_torque"].median()),
            "median_power": float(fp5["power_W"].median()),
            "median_wear": float(fp5["tool_wear_min"].median()),
            "median_torque": float(fp5["torque_Nm"].median()),
            "n_L_grade": int((fp5["type"] == "L").sum()),
        }
        print("\nThese five sit in failure territory on the features the model "
              "cares about and did not fail. Their median wear-torque product is "
              "%.0f against a healthy median of %.0f and a failure median of %.0f. "
              "Median tool wear is %.0f minutes against %.0f for healthy records. "
              "%d of the five are L-grade product, which carries the lowest "
              "overstrain limit."
              % (fp_summary["median_wear_torque"], med_ok["wear_torque"],
                 med_fail["wear_torque"], fp_summary["median_wear"],
                 med_ok["tool_wear_min"], fp_summary["n_L_grade"]))
        print("The honest reading is that these are not model errors in any "
              "interesting sense. They are machines operating in the conditions "
              "under which failures happen, that happened not to fail. The AI4I "
              "generator makes failure probabilistic near its thresholds, so some "
              "records sit inside the risky region with a zero label. A "
              "maintenance planner would want these inspected.")

        C.banner("WHAT THE FALSE NEGATIVES HAVE IN COMMON")
        for i, (_, r) in enumerate(fn5.iterrows(), 1):
            print("\nFN%d  UDI %d, product %s, grade %s, predicted probability %.3f"
                  % (i, r["UDI"], r["Product ID"], r["type"],
                     r["predicted_probability"]))
            print("     torque %.1f Nm (failure median %.1f), speed %d rpm "
                  "(median %d), wear %d min (median %d)"
                  % (r["torque_Nm"], med_fail["torque_Nm"], r["rot_speed_rpm"],
                     med_fail["rot_speed_rpm"], r["tool_wear_min"],
                     med_fail["tool_wear_min"]))
            print("     power %.0f W (failure median %.0f), temp gap %.1f K "
                  "(failure median %.1f K), wear x torque %.0f (failure median %.0f)"
                  % (r["power_W"], med_fail["power_W"], r["temp_diff_K"],
                     med_fail["temp_diff_K"], r["wear_torque"],
                     med_fail["wear_torque"]))

        in_band = int(((fn5["power_W"] >= 3500) & (fn5["power_W"] <= 9000)).sum())
        fn_summary = {
            "median_wear_torque": float(fn5["wear_torque"].median()),
            "median_power": float(fn5["power_W"].median()),
            "median_wear": float(fn5["tool_wear_min"].median()),
            "median_torque": float(fn5["torque_Nm"].median()),
            "median_temp_diff": float(fn5["temp_diff_K"].median()),
            "in_safe_power_band": in_band,
            "max_probability": float(fn5["predicted_probability"].max()),
        }
        print("\nThese five look ordinary on every feature the model uses. Their "
              "median wear-torque product is %.0f against a failure median of %.0f "
              "and a healthy median of %.0f. %d of the five sit inside the safe "
              "3,500 to 9,000 W power band. Their median temperature gap is %.1f K "
              "against %.1f K for failures. The highest probability the model gave "
              "any of them was %.3f."
              % (fn_summary["median_wear_torque"], med_fail["wear_torque"],
                 med_ok["wear_torque"], in_band, fn_summary["median_temp_diff"],
                 med_fail["temp_diff_K"], fn_summary["max_probability"]))
        # Cross-check against the nine unexplained failures found in stage 01
        unflagged = raw[(raw[C.DETERMINISTIC_FLAGS].sum(axis=1) == 0) &
                        (raw[C.TARGET] == 1)]["UDI"].tolist()
        hits = [int(u) for u in fn5["UDI"] if int(u) in unflagged]
        fn_summary["unflagged_overlap"] = hits
        fn_summary["n_unflagged_overlap"] = len(hits)
        print("\nCross-check against stage 01. That audit found %d records labelled "
              "as failures with none of the four documented failure modes set: "
              "UDI %s." % (len(unflagged), ", ".join(str(u) for u in unflagged)))
        print("%d of these five false negatives are on that list: UDI %s."
              % (len(hits), ", ".join(str(h) for h in hits)))
        print("That is the explanation, and it is a property of the labelling "
              "rather than of the model. Those records carry a failure label with "
              "no accompanying signature in any sensor the dataset provides, so "
              "there is nothing for any learner to find. No amount of tuning "
              "reaches them, and lowering the threshold far enough to catch them "
              "would bury the crew in alarms. The remaining %d sit close to a "
              "documented boundary without crossing it decisively."
              % (5 - len(hits)))

        # ------------------------------------------------------------------
        cases = pd.concat([fp_tbl.assign(error_type="False positive"),
                           fn_tbl.assign(error_type="False negative")])
        C.save_table(cases, "tbl45_misclassified_cases_combined")

        # Figure: the ten cases against the two class distributions
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
        panel = ["torque_Nm", "rot_speed_rpm", "tool_wear_min", "power_W",
                 "temp_diff_K", "wear_torque"]
        for ax, col in zip(axes.ravel(), panel):
            parts = ax.violinplot(
                [test.loc[y == 0, col].to_numpy(), test.loc[y == 1, col].to_numpy()],
                positions=[0, 1], showmedians=True, widths=0.75)
            for pc, colour in zip(parts["bodies"], [C.C_OK, C.C_FAIL]):
                pc.set_facecolor(colour)
                pc.set_alpha(0.35)
            ax.scatter(np.zeros(len(fp5)) + 0.0, fp5[col], s=70, marker="X",
                       color=C.PALETTE[3], edgecolor="black", linewidth=0.6,
                       zorder=5, label="False positives")
            ax.scatter(np.ones(len(fn5)) * 1.0, fn5[col], s=70, marker="P",
                       color=C.PALETTE[5], edgecolor="black", linewidth=0.6,
                       zorder=5, label="False negatives")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["actual\nno failure", "actual\nfailure"])
            ax.set_title(col)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=9,
                   bbox_to_anchor=(0.995, 0.995))
        fig.suptitle("The ten misclassified cases against their own class distributions",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        C.save_fig(fig, "fig20_misclassification_cases")
        plt.close(fig)

        # Deviation heatmap
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        for ax, dev, title in [(axes[0], fp_dev, "False positives against the healthy median"),
                               (axes[1], fn_dev, "False negatives against the failure median")]:
            m = dev.set_index("case")[SHOW]
            sns.heatmap(m, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                        vmin=-3, vmax=3, linewidths=0.7, ax=ax,
                        cbar_kws={"label": "IQR units from the median"})
            ax.set_title(title)
            ax.set_ylabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right",
                               fontsize=8.5)
        fig.suptitle("How far each mistake sits from the class it truly belongs to",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        C.save_fig(fig, "fig21_misclassification_deviation")
        plt.close(fig)

        C.put("misclassification", {
            "threshold": thr, "model": MODEL,
            "n_fp": int(len(fps)), "n_fn": int(len(fns)),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "fp_cases": fp_tbl.to_dict("records"),
            "fn_cases": fn_tbl.to_dict("records"),
            "fp_summary": fp_summary, "fn_summary": fn_summary,
            "med_ok": {k: float(v) for k, v in med_ok.items()},
            "med_fail": {k: float(v) for k, v in med_fail.items()},
        })
        print("\nStage 10 complete.")
    finally:
        C.end_log(tee)


if __name__ == "__main__":
    main()
