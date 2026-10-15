"""
Stage 01 - Provenance and data audit  (Task 2)

Downloads the untouched AI4I 2020 file from the UCI archive, saves the raw copy,
then audits dimensions, types, target balance, missingness, duplicates, odd
values, identifiers, and leakage. The leakage argument for TWF/HDF/PWF/OSF/RNF
is settled empirically here, not asserted.
"""

from __future__ import annotations

import io
import sys
import zipfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as C

STAGE = "01_data_audit"


# ---------------------------------------------------------------- provenance
def fetch_raw() -> pd.DataFrame:
    """Primary route: the zipped CSV published on the UCI archive.
    Fallback: the ucimlrepo package, same dataset id 601."""
    if C.RAW_CSV.exists():
        print("Raw copy already on disk: " + str(C.RAW_CSV))
        return pd.read_csv(C.RAW_CSV)

    try:
        print("Downloading " + C.UCI_ZIP_URL)
        blob = urllib.request.urlopen(C.UCI_ZIP_URL, timeout=120).read()
        zf = zipfile.ZipFile(io.BytesIO(blob))
        member = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        df = pd.read_csv(io.BytesIO(zf.read(member)))
        source = "UCI archive zip -> " + member
    except Exception as exc:                                   # pragma: no cover
        print("Direct download failed (%s: %s); using ucimlrepo." % (type(exc).__name__, exc))
        from ucimlrepo import fetch_ucirepo
        repo = fetch_ucirepo(id=C.UCI_DATASET_ID)
        df = repo.data.original.copy()
        df = df.rename(columns={
            "UID": "UDI",
            "Air temperature": "Air temperature [K]",
            "Process temperature": "Process temperature [K]",
            "Rotational speed": "Rotational speed [rpm]",
            "Torque": "Torque [Nm]",
            "Tool wear": "Tool wear [min]",
        })
        source = "ucimlrepo id=%d" % C.UCI_DATASET_ID

    df.to_csv(C.RAW_CSV, index=False)
    print("Source: " + source)
    print("Raw copy written untouched to " + str(C.RAW_CSV))
    return df


def main() -> None:
    tee = C.start_log(STAGE)
    try:
        C.banner("STAGE 01  DATA AUDIT AND LEAKAGE ASSESSMENT")
        print("Assigned seed for the whole project: SEED = %d" % C.SEED)

        df = fetch_raw()

        # -------------------------------------------------- provenance proof
        C.banner("PROVENANCE PROOF")
        print("Shape: %s" % (df.shape,))
        print("Expected: %s" % (C.EXPECTED_SHAPE,))
        assert df.shape == C.EXPECTED_SHAPE, "Dataset is not the original 10000 x 14 file"
        print("\nColumns:")
        for i, col in enumerate(df.columns, 1):
            print("  %2d. %s" % (i, col))
        print("\nFirst three rows:")
        print(df.head(3).to_string())

        # -------------------------------------------------- variable dictionary
        C.banner("VARIABLE DICTIONARY AND TYPES")
        roles = {
            "UDI": "identifier",
            "Product ID": "identifier",
            "Type": "categorical predictor",
            "Air temperature [K]": "numeric predictor",
            "Process temperature [K]": "numeric predictor",
            "Rotational speed [rpm]": "numeric predictor",
            "Torque [Nm]": "numeric predictor",
            "Tool wear [min]": "numeric predictor",
            "Machine failure": "target",
            "TWF": "post-hoc failure-mode label (leakage)",
            "HDF": "post-hoc failure-mode label (leakage)",
            "PWF": "post-hoc failure-mode label (leakage)",
            "OSF": "post-hoc failure-mode label (leakage)",
            "RNF": "post-hoc failure-mode label (leakage)",
        }
        dic = pd.DataFrame({
            "variable": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "n_unique": [df[c].nunique() for c in df.columns],
            "n_missing": [int(df[c].isna().sum()) for c in df.columns],
            "pct_missing": [round(100 * df[c].isna().mean(), 3) for c in df.columns],
            "example_values": [
                ", ".join(map(str, df[c].dropna().unique()[:3])) for c in df.columns
            ],
            "assigned_role": [roles[c] for c in df.columns],
        })
        C.save_table(dic, "tbl01_variable_dictionary")

        # -------------------------------------------------- target distribution
        C.banner("TARGET DISTRIBUTION")
        counts = df[C.TARGET].value_counts().sort_index()
        n_pos, n_neg = int(counts.get(1, 0)), int(counts.get(0, 0))
        pos_rate = n_pos / len(df)
        print("Machine failure = 0 : %5d  (%.2f%%)" % (n_neg, 100 * n_neg / len(df)))
        print("Machine failure = 1 : %5d  (%.2f%%)" % (n_pos, 100 * pos_rate))
        print("Imbalance ratio     : 1 failure per %.1f healthy records" % (n_neg / n_pos))
        print("Majority-class accuracy floor: %.2f%%" % (100 * (1 - pos_rate)))

        # -------------------------------------------------- quality checks
        C.banner("DATA QUALITY CHECKS")
        print("Missing cells anywhere         : %d" % int(df.isna().sum().sum()))
        dup_full = int(df.duplicated().sum())
        dup_nokey = int(df.drop(columns=C.ID_COLS).duplicated().sum())
        print("Fully duplicated rows          : %d" % dup_full)
        print("Duplicates ignoring identifiers: %d" % dup_nokey)
        print("UDI unique                     : %s (%d values)"
              % (df["UDI"].is_unique, df["UDI"].nunique()))
        print("Product ID unique              : %s (%d values)"
              % (df["Product ID"].is_unique, df["Product ID"].nunique()))

        # Product ID encodes Type in its first character: a subtle redundancy.
        pid_prefix = df["Product ID"].str[0]
        agree = float((pid_prefix == df["Type"]).mean())
        print("\nProduct ID first character equals Type in %.2f%% of rows" % (100 * agree))
        print("So Product ID is not merely an identifier, it silently repeats Type.")
        print(pd.crosstab(pid_prefix, df["Type"]).to_string())

        # Unusual values: physical plausibility rather than blind outlier removal
        C.banner("UNUSUAL OR PROBLEMATIC VALUES")
        num = df[C.NUMERIC_RAW]
        desc = num.describe().T
        desc["skew"] = num.skew()
        desc["kurtosis"] = num.kurtosis()
        q1, q3 = num.quantile(0.25), num.quantile(0.75)
        iqr = q3 - q1
        desc["n_iqr_outliers"] = [
            int(((num[c] < q1[c] - 1.5 * iqr[c]) | (num[c] > q3[c] + 1.5 * iqr[c])).sum())
            for c in num.columns
        ]
        desc["n_nonpositive"] = [int((num[c] <= 0).sum()) for c in num.columns]
        desc = desc.round(3).reset_index().rename(columns={"index": "variable"})
        C.save_table(desc, "tbl02_numeric_quality")

        neg_gap = int((df["Process temperature [K]"] - df["Air temperature [K]"] <= 0).sum())
        print("\nRows where process temperature <= air temperature: %d" % neg_gap)
        print("A negative gap would be physically impossible for a heated process, "
              "so this is a genuine validity check rather than a formality.")
        zero_torque = int((df["Torque [Nm]"] <= 0).sum())
        print("Rows with non-positive torque: %d" % zero_torque)
        rs_hi = int((df["Rotational speed [rpm]"] > 2500).sum())
        print("Rotational speed above 2500 rpm: %d (max %d rpm)"
              % (rs_hi, df["Rotational speed [rpm]"].max()))
        print("High-speed readings sit in the right tail of a skewed but continuous "
              "distribution, so I keep them: they are operating states, not typos.")

        # -------------------------------------------------- LEAKAGE
        C.banner("LEAKAGE INVESTIGATION: TWF, HDF, PWF, OSF, RNF")

        flag_counts = pd.DataFrame({
            "flag": C.FAILURE_FLAGS,
            "n_fired": [int(df[f].sum()) for f in C.FAILURE_FLAGS],
            "pct_of_rows": [round(100 * df[f].mean(), 3) for f in C.FAILURE_FLAGS],
            "n_fired_and_failure": [
                int(((df[f] == 1) & (df[C.TARGET] == 1)).sum()) for f in C.FAILURE_FLAGS
            ],
            "n_fired_but_no_failure": [
                int(((df[f] == 1) & (df[C.TARGET] == 0)).sum()) for f in C.FAILURE_FLAGS
            ],
        })
        flag_counts["pct_fired_that_are_failures"] = (
            100 * flag_counts["n_fired_and_failure"] / flag_counts["n_fired"]
        ).round(2)
        C.save_table(flag_counts, "tbl03_failure_flag_behaviour")

        # Is Machine failure the OR of the four deterministic modes?
        or4 = (df[C.DETERMINISTIC_FLAGS].sum(axis=1) > 0).astype(int)
        or5 = (df[C.FAILURE_FLAGS].sum(axis=1) > 0).astype(int)

        match4 = int((or4 == df[C.TARGET]).sum())
        match5 = int((or5 == df[C.TARGET]).sum())
        fp4 = int(((or4 == 1) & (df[C.TARGET] == 0)).sum())   # flag fires, target 0
        fn4 = int(((or4 == 0) & (df[C.TARGET] == 1)).sum())   # target 1, no flag
        fp5 = int(((or5 == 1) & (df[C.TARGET] == 0)).sum())
        fn5 = int(((or5 == 0) & (df[C.TARGET] == 1)).sum())

        print("OR(TWF,HDF,PWF,OSF) reproduces Machine failure on %d/%d rows (%.2f%%)"
              % (match4, len(df), 100 * match4 / len(df)))
        print("   rows where the OR fires but the target is 0 : %d" % fp4)
        print("   rows where the target is 1 but no mode fires: %d" % fn4)
        print("OR(all five, RNF included) reproduces the target on %d/%d rows (%.2f%%)"
              % (match5, len(df), 100 * match5 / len(df)))
        print("   rows where the OR fires but the target is 0 : %d" % fp5)
        print("   rows where the target is 1 but no mode fires: %d" % fn5)

        # RNF treated separately, because it behaves differently
        rnf_n = int(df["RNF"].sum())
        rnf_and_fail = int(((df["RNF"] == 1) & (df[C.TARGET] == 1)).sum())
        rnf_only = int(((df["RNF"] == 1) &
                        (df[C.DETERMINISTIC_FLAGS].sum(axis=1) == 0)).sum())
        rnf_only_and_fail = int(((df["RNF"] == 1) &
                                 (df[C.DETERMINISTIC_FLAGS].sum(axis=1) == 0) &
                                 (df[C.TARGET] == 1)).sum())
        print("\nRNF fires on %d rows; %d of those are labelled failures."
              % (rnf_n, rnf_and_fail))
        print("RNF fires alone (no other mode) on %d rows; %d of those are labelled failures."
              % (rnf_only, rnf_only_and_fail))

        # Which rows disagree, and why
        amb = df[(or4 == 1) & (df[C.TARGET] == 0)]
        unexp = df[(or4 == 0) & (df[C.TARGET] == 1)]
        print("\nDisagreement rows where a mode fires but the target is 0: %d" % len(amb))
        if len(amb):
            print(amb[C.ID_COLS + C.FAILURE_FLAGS + [C.TARGET]].to_string(index=False))
        print("\nDisagreement rows where the target is 1 but no mode fires: %d" % len(unexp))
        if len(unexp):
            print(unexp[C.ID_COLS + C.FAILURE_FLAGS + [C.TARGET]].to_string(index=False))

        combo = (df.groupby(C.FAILURE_FLAGS, as_index=False)
                   .agg(n_rows=(C.TARGET, "size"), n_failures=(C.TARGET, "sum")))
        combo["failure_rate_pct"] = (100 * combo["n_failures"] / combo["n_rows"]).round(2)
        combo = combo.sort_values("n_rows", ascending=False)
        C.save_table(combo, "tbl04_flag_combination_crosstab")

        leak_summary = pd.DataFrame({
            "check": [
                "OR(TWF,HDF,PWF,OSF) == Machine failure",
                "rows: mode fires, target = 0",
                "rows: target = 1, no mode fires",
                "OR(all five incl. RNF) == Machine failure",
                "RNF rows total",
                "RNF fires alone",
                "RNF fires alone and target = 1",
            ],
            "result": [
                "%d/%d rows (%.2f%%)" % (match4, len(df), 100 * match4 / len(df)),
                str(fp4), str(fn4),
                "%d/%d rows (%.2f%%)" % (match5, len(df), 100 * match5 / len(df)),
                str(rnf_n), str(rnf_only), str(rnf_only_and_fail),
            ],
        })
        C.save_table(leak_summary, "tbl05_leakage_summary")

        # -------------------------------------------------- inclusion decision
        C.banner("INCLUSION DECISION")
        decisions = pd.DataFrame([
            ("UDI", "Exclude", "Row counter 1 to 10000, unique on every row. Carries the "
             "sort order of the file and nothing about machine condition."),
            ("Product ID", "Exclude", "Quality letter plus serial number, 10000 unique "
             "values. Its first character duplicates Type, so it adds a redundant copy "
             "of a variable I already keep, plus thousands of useless levels."),
            ("Type", "Include", "Three-level product quality rating L/M/H, a genuine "
             "design characteristic known before the machine runs."),
            ("Air temperature [K]", "Include", "Ambient process condition, measured live."),
            ("Process temperature [K]", "Include", "Machine-side temperature, measured live."),
            ("Rotational speed [rpm]", "Include", "Live sensor reading."),
            ("Torque [Nm]", "Include", "Live sensor reading."),
            ("Tool wear [min]", "Include", "Cumulative minutes of tool use, known live."),
            ("TWF", "Exclude", "Tool-wear failure mode. Post-hoc label assigned after a "
             "failure is diagnosed; also part of the OR that defines the target."),
            ("HDF", "Exclude", "Heat-dissipation failure mode, same argument."),
            ("PWF", "Exclude", "Power failure mode, same argument."),
            ("OSF", "Exclude", "Overstrain failure mode, same argument."),
            ("RNF", "Exclude", "Random failure flag. Not part of the OR that defines the "
             "target and essentially unpredictable, so it is noise at prediction time."),
        ], columns=["variable", "decision", "reason"])
        C.save_table(decisions, "tbl06_inclusion_decisions")

        C.put("audit", {
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "pos_rate_pct": round(100 * pos_rate, 2),
            "majority_accuracy_pct": round(100 * (1 - pos_rate), 2),
            "imbalance_ratio": round(n_neg / n_pos, 1),
            "n_missing": int(df.isna().sum().sum()),
            "dup_full": dup_full,
            "dup_nokey": dup_nokey,
            "udi_unique": int(df["UDI"].nunique()),
            "pid_unique": int(df["Product ID"].nunique()),
            "pid_type_agreement_pct": round(100 * agree, 2),
            "or4_match": match4,
            "or4_match_pct": round(100 * match4 / len(df), 2),
            "or4_flag_no_target": fp4,
            "or4_target_no_flag": fn4,
            "or5_match": match5,
            "or5_match_pct": round(100 * match5 / len(df), 2),
            "or5_flag_no_target": fp5,
            "or5_target_no_flag": fn5,
            "rnf_n": rnf_n,
            "rnf_and_fail": rnf_and_fail,
            "rnf_only": rnf_only,
            "rnf_only_and_fail": rnf_only_and_fail,
            "neg_temp_gap_rows": neg_gap,
            "n_iqr_out_speed": int(desc.loc[desc["variable"] == "Rotational speed [rpm]",
                                            "n_iqr_outliers"].iloc[0]),
            "n_iqr_out_torque": int(desc.loc[desc["variable"] == "Torque [Nm]",
                                             "n_iqr_outliers"].iloc[0]),
            "flag_counts": {f: int(df[f].sum()) for f in C.FAILURE_FLAGS},
            "flag_fail_counts": {
                f: int(((df[f] == 1) & (df[C.TARGET] == 1)).sum()) for f in C.FAILURE_FLAGS
            },
            "max_rot_speed": float(df["Rotational speed [rpm]"].max()),
            "min_rot_speed": float(df["Rotational speed [rpm]"].min()),
        })
        print("\nStage 01 complete.")
    finally:
        C.end_log(tee)


if __name__ == "__main__":
    main()
