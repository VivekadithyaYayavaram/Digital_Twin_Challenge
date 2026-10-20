"""Patient-level splitting, preprocessing, candidate models and a rule baseline."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODEL_NAMES = {
    "rule_baseline": "Rule baseline (current or 60-min projected CGM >= threshold)",
    "logistic_regression": "Logistic regression (baseline ML)",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Histogram gradient boosting",
}


def split_patients(ehr: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Assign every patient to exactly one of train / val / test, stratified by
    diabetes status so each split has both prediabetes and T2D patients."""
    rng = np.random.default_rng(int(cfg["seed"]))
    sc = cfg["split"]
    parts = []
    for _, grp in ehr.groupby("diabetes_status", sort=True):
        ids = grp["patient_id"].sort_values().to_numpy()
        ids = ids[rng.permutation(len(ids))]
        n_tr = int(round(sc["train_frac"] * len(ids)))
        n_va = int(round(sc["val_frac"] * len(ids)))
        lab = np.array(["train"] * n_tr + ["val"] * n_va + ["test"] * (len(ids) - n_tr - n_va))
        parts.append(pd.DataFrame({"patient_id": ids, "split": lab}))
    return pd.concat(parts).sort_values("patient_id").reset_index(drop=True)


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    transformers = []
    if numeric:
        transformers.append(("num", num, numeric))
    if categorical:
        transformers.append(("cat", cat, categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_estimator(name: str, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=3000, C=1.0)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                                      n_jobs=-1, random_state=seed)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                              min_samples_leaf=40, l2_regularization=1.0,
                                              early_stopping=False, random_state=seed)
    raise ValueError(f"unknown model {name}")


def make_pipeline(name: str, seed: int, numeric: list[str] | None = None,
                  categorical: list[str] | None = None) -> Pipeline:
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    categorical = CATEGORICAL_FEATURES if categorical is None else categorical
    return Pipeline([("prep", make_preprocessor(numeric, categorical)),
                     ("model", make_estimator(name, seed))])


def rule_baseline_score(df: pd.DataFrame, threshold: float) -> np.ndarray:
    """Non-ML heuristic: max(current CGM, CGM projected 60 min ahead on the
    15-min trend), mapped to (0, 1) with a logistic so ROC/PR can be computed.
    Score 0.5 <=> projected glucose == threshold."""
    cur = df["cgm_current"].to_numpy(dtype=float)
    roc = np.nan_to_num(df["cgm_roc_15"].to_numpy(dtype=float), nan=0.0)
    proj = np.maximum(cur, cur + 60 * np.clip(roc, -3, 3))
    proj = np.where(np.isnan(proj), threshold - 40, proj)  # no CGM at all -> low score
    return 1 / (1 + np.exp(-(proj - threshold) / 15.0))
