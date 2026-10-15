"""
Shared pipeline components used by stages 05 to 12.

Keeping these in one place is what guarantees that the baseline, the tuning
search, the imbalance study, the evaluation, and the robustness check all fit
exactly the same transformations on exactly the same folds. Every transformer
here is fitted inside cross-validation, never on the full data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C


def feature_columns(use_engineered: bool = True):
    """Numeric and categorical predictor names for a given feature set."""
    numeric = list(C.NUMERIC_CLEAN)
    if use_engineered:
        numeric = numeric + list(C.ENGINEERED)
    return numeric, list(C.CAT_CLEAN)


def build_preprocessor(scale: bool, use_engineered: bool = True) -> ColumnTransformer:
    """Common preprocessing for every model, plus the one model-specific step.

    Common to all four models
        median imputation on the numeric block (a safeguard for live sensor
        dropouts; the training file itself has no missing cells)
        one-hot encoding of the three-level product grade

    Model-specific
        standardisation, applied only when ``scale`` is True. Logistic
        regression needs it because its L2 penalty is scale sensitive. The
        three tree-based models split on order statistics, so scaling would
        change nothing except readability of the fitted objects.
    """
    numeric, categorical = feature_columns(use_engineered)

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), numeric),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                                  sparse_output=False), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_preprocessor_cols(numeric, categorical, scale: bool) -> ColumnTransformer:
    """The same design as build_preprocessor, for an arbitrary column list.
    Used by the feature-set ablation in stage 04 and the robustness check."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), list(numeric))]
    if categorical:
        transformers.append(
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                                  sparse_output=False), list(categorical)))
    return ColumnTransformer(transformers=transformers, remainder="drop",
                             verbose_feature_names_out=False)


def expanded_feature_names(use_engineered: bool = True):
    """Column names after the ColumnTransformer, in output order."""
    numeric, _ = feature_columns(use_engineered)
    # OneHotEncoder(drop='first') on Type with categories H, L, M keeps L and M
    return numeric + ["type_L", "type_M"]


def load_model_ready(use_engineered: bool = True):
    """Return X_train, X_test, y_train, y_test from the stored model-ready file.

    The split column was written once in stage 03 using SEED, so every stage
    that calls this function sees the identical partition.
    """
    df = pd.read_csv(C.MODEL_READY_CSV)
    numeric, categorical = feature_columns(use_engineered)
    cols = numeric + categorical
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    return (train[cols].copy(), test[cols].copy(),
            train[C.TARGET_CLEAN].to_numpy(), test[C.TARGET_CLEAN].to_numpy())


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise physical features. No cross-row statistic is used, so these can
    be computed before the split without leaking anything between folds."""
    out = df.copy()
    out["power_W"] = out["torque_Nm"] * out["rot_speed_rpm"] * 2 * np.pi / 60.0
    out["temp_diff_K"] = out["process_temp_K"] - out["air_temp_K"]
    out["wear_torque"] = out["tool_wear_min"] * out["torque_Nm"]
    return out


def confusion_counts(y_true, y_pred):
    """Return tn, fp, fn, tp as plain integers."""
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def score_row(name, y_true, y_pred, y_proba):
    """One row of the common evaluation framework used everywhere."""
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score)
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc": round(average_precision_score(y_true, y_proba), 4),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


# ---------------------------------------------------------------------------
# Final model assembly, shared by stages 07 to 12 so that every stage builds
# exactly the same objects from the stored tuning and imbalance decisions.
# ---------------------------------------------------------------------------

def build_estimator(name, params, weighted=False, pos_weight=None, seed=None):
    """Rebuild one tuned model, optionally with its class-weighting switch on."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    seed = C.SEED if seed is None else seed
    p = dict(params)
    if name == "Logistic Regression":
        if weighted:
            p["class_weight"] = "balanced"
        return LogisticRegression(max_iter=5000, random_state=seed, **p)
    if name == "Decision Tree":
        if weighted:
            p["class_weight"] = "balanced"
        return DecisionTreeClassifier(random_state=seed, **p)
    if name == "Random Forest":
        if weighted:
            p["class_weight"] = "balanced"
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **p)
    if name == "XGBoost":
        if weighted:
            p["scale_pos_weight"] = pos_weight
        return XGBClassifier(random_state=seed, eval_metric="logloss",
                             n_jobs=-1, tree_method="hist", **p)
    raise ValueError(name)


def build_treated_pipeline(name, params, scale, treatment, pos_weight,
                           use_engineered=True, seed=None):
    """A full pipeline: preprocessing, the chosen imbalance treatment, the model."""
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline

    seed = C.SEED if seed is None else seed
    pre = build_preprocessor(scale=scale, use_engineered=use_engineered)

    if treatment == "none":
        return Pipeline([("pre", pre),
                         ("clf", build_estimator(name, params, seed=seed))])
    if treatment == "class weighting":
        return Pipeline([("pre", pre),
                         ("clf", build_estimator(name, params, weighted=True,
                                                 pos_weight=pos_weight, seed=seed))])
    if treatment == "SMOTE":
        return ImbPipeline([("pre", pre),
                            ("smote", SMOTE(random_state=seed, k_neighbors=5)),
                            ("clf", build_estimator(name, params, seed=seed))])
    if treatment == "SMOTE + undersample":
        return ImbPipeline([("pre", pre),
                            ("smote", SMOTE(random_state=seed, k_neighbors=5,
                                            sampling_strategy=0.3)),
                            ("under", RandomUnderSampler(random_state=seed,
                                                         sampling_strategy=0.5)),
                            ("clf", build_estimator(name, params, seed=seed))])
    raise ValueError(treatment)


def build_final_models(use_engineered=True, seed=None):
    """Every model in its final form: tuned parameters plus retained treatment."""
    tuning = C.get("tuning")["chosen"]
    imb = C.get("imbalance")
    out = {}
    for name, cfg in tuning.items():
        out[name] = build_treated_pipeline(
            name, cfg["params"], cfg["scale"], imb["retained"][name],
            imb["pos_weight"], use_engineered=use_engineered, seed=seed)
    return out


MODEL_ORDER = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
