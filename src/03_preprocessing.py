"""
Stage 03 - Preprocessing design and the train/test split  (Task 4)

Drops the identifiers and the leaking failure-mode flags, renames the sensor
columns, makes the stratified split under SEED, and records which preprocessing
steps are common to all four models and which depend on the algorithm.

No transformer is fitted here. Fitting happens inside cross-validation in the
later stages, which is the whole point of putting the steps in a Pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C
import pipelines as P

STAGE = "03_preprocessing"


def main() -> None:
    tee = C.start_log(STAGE)
    try:
        C.banner("STAGE 03  PREPROCESSING DESIGN")
        raw = pd.read_csv(C.RAW_CSV)
        print("Loaded raw file: %s rows, %s columns" % raw.shape)

        # ------------------------------------------------------- 1. selection
        C.banner("VARIABLE SELECTION APPLIED")
        drop_cols = C.ID_COLS + C.FAILURE_FLAGS
        print("Dropping identifiers   : %s" % ", ".join(C.ID_COLS))
        print("Dropping leaking flags : %s" % ", ".join(C.FAILURE_FLAGS))
        df = raw.drop(columns=drop_cols).rename(columns=C.RENAME)
        print("Retained columns       : %s" % ", ".join(df.columns))
        print("Shape after selection  : %s rows, %s columns" % df.shape)

        # ------------------------------------------------------- 2. validity
        C.banner("VALIDITY CHECKS ON THE RETAINED COLUMNS")
        print("Missing cells                    : %d" % int(df.isna().sum().sum()))
        print("Duplicated rows                  : %d" % int(df.duplicated().sum()))
        print("type levels                      : %s"
              % sorted(df["type"].unique().tolist()))
        print("Target levels                    : %s"
              % sorted(df[C.TARGET_CLEAN].unique().tolist()))
        bad = int((df["process_temp_K"] <= df["air_temp_K"]).sum())
        print("Physically impossible temp gaps  : %d" % bad)
        print("\nNo row is dropped and no value is imputed, because nothing needs "
              "either. I keep a median imputer inside the numeric branch of the "
              "pipeline anyway: the deployed model will meet live sensor dropouts "
              "that this static extract does not contain.")

        # ------------------------------------------------------- 3. split
        C.banner("STRATIFIED TRAIN / TEST SPLIT")
        X = df.drop(columns=[C.TARGET_CLEAN])
        y = df[C.TARGET_CLEAN]
        idx_train, idx_test = train_test_split(
            np.arange(len(df)),
            test_size=C.TEST_SIZE,
            random_state=C.SEED,
            stratify=y,
        )
        df["split"] = "train"
        df.loc[df.index[idx_test], "split"] = "test"

        tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
        split_tbl = pd.DataFrame({
            "partition": ["train", "test", "total"],
            "n_rows": [len(tr), len(te), len(df)],
            "n_failures": [int(tr[C.TARGET_CLEAN].sum()), int(te[C.TARGET_CLEAN].sum()),
                           int(df[C.TARGET_CLEAN].sum())],
        })
        split_tbl["failure_rate_pct"] = (
            100 * split_tbl["n_failures"] / split_tbl["n_rows"]).round(3)
        C.save_table(split_tbl, "tbl16_train_test_split")
        print("\ntest_size = %.2f, random_state = %d, stratify on the target."
              % (C.TEST_SIZE, C.SEED))
        print("Stratifying matters at a 3.39% base rate: an unstratified 20% "
              "test set could easily land 15 failures away from the expected count, "
              "which would move recall by several points for reasons that have "
              "nothing to do with the model.")

        # ------------------------------------------------------- 4. features
        C.banner("ENGINEERED FEATURES ADDED (evaluated in stage 04)")
        df = P.add_engineered(df)
        print("Added: %s" % ", ".join(C.ENGINEERED))
        print("Each is a row-wise function of that row's own sensor values. No "
              "mean, no quantile, no target information is involved, so computing "
              "them before the split cannot leak across folds. I state this "
              "explicitly because it is the one place where a reader might "
              "reasonably suspect leakage.")

        ordered = (C.NUMERIC_CLEAN + C.ENGINEERED + C.CAT_CLEAN +
                   [C.TARGET_CLEAN, "split"])
        df = df[ordered]
        df.to_csv(C.MODEL_READY_CSV, index=False)
        print("\nModel-ready dataset written to %s" % C.MODEL_READY_CSV)
        print("Shape: %s rows, %s columns" % df.shape)
        print(df.head(5).to_string(index=False))

        # ------------------------------------------------------- 5. the plan
        C.banner("PREPROCESSING PLAN: COMMON VERSUS MODEL-SPECIFIC")
        plan = pd.DataFrame([
            ("Drop UDI, Product ID", "Common", "All four",
             "Identifiers with no operational meaning; Product ID also duplicates Type."),
            ("Drop TWF, HDF, PWF, OSF, RNF", "Common", "All four",
             "Post-hoc failure-mode labels; the first four reproduce the target on "
             "99.91% of rows, so keeping them would hand the answer to the model."),
            ("Median imputation, numeric block", "Common", "All four",
             "Redundant on this extract (zero missing cells) and retained as an "
             "operational safeguard for live sensor dropouts."),
            ("One-hot encode type, drop first", "Common", "All four",
             "Three levels only. One-hot costs two columns and avoids asserting "
             "that the L to M to H spacing is equal, which an ordinal code would."),
            ("Engineered features", "Common", "All four",
             "power_W, temp_diff_K, wear_torque. Row-wise, so leakage-free."),
            ("Standardisation", "Model-specific", "Logistic Regression only",
             "The L2 penalty shrinks coefficients in proportion to their scale, so "
             "an unscaled torque in Nm and a wear-torque product in the thousands "
             "would be penalised very differently. Trees split on rank order, so "
             "scaling them would change nothing but would hide the units."),
            ("Class-imbalance treatment", "Model-specific", "Compared in stage 07",
             "class_weight for LR/DT/RF, scale_pos_weight for XGBoost, SMOTE inside "
             "the cross-validation pipeline. Never applied to a validation fold."),
        ], columns=["step", "scope", "applies_to", "justification"])
        C.save_table(plan, "tbl17_preprocessing_plan")

        # Show the two concrete preprocessor objects
        C.banner("THE TWO PREPROCESSOR OBJECTS")
        print("Scaled variant (Logistic Regression):")
        print(P.build_preprocessor(scale=True))
        print("\nUnscaled variant (Decision Tree, Random Forest, XGBoost):")
        print(P.build_preprocessor(scale=False))

        # sanity check: fitting on train only produces the expected width
        pre = P.build_preprocessor(scale=True)
        num, cat = P.feature_columns(True)
        train_full = df[df["split"] == "train"]
        Xt = pre.fit_transform(train_full[num + cat])
        print("\nFitted on the training partition only. Design matrix: %s" % (Xt.shape,))
        print("Output columns: %s" % ", ".join(P.expanded_feature_names(True)))
        means = Xt[:, :len(num)].mean(axis=0)
        print("Post-scaling training means (should be ~0): %s"
              % np.round(means, 6).tolist())

        C.put("split", {
            "n_train": len(tr), "n_test": len(te),
            "train_pos": int(tr[C.TARGET_CLEAN].sum()),
            "test_pos": int(te[C.TARGET_CLEAN].sum()),
            "train_neg": int((tr[C.TARGET_CLEAN] == 0).sum()),
            "test_neg": int((te[C.TARGET_CLEAN] == 0).sum()),
            "train_rate_pct": round(100 * tr[C.TARGET_CLEAN].mean(), 3),
            "test_rate_pct": round(100 * te[C.TARGET_CLEAN].mean(), 3),
            "test_size": C.TEST_SIZE,
            "n_features_model": int(Xt.shape[1]),
            "dropped": drop_cols,
            "retained": [c for c in df.columns if c not in (C.TARGET_CLEAN, "split")],
        })
        print("\nStage 03 complete.")
    finally:
        C.end_log(tee)


if __name__ == "__main__":
    main()
