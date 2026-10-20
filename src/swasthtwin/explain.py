"""Model-agnostic local explanations by single-feature and block perturbation.

For one prediction row we replace a feature (or a whole data stream) with the
value of a "typical training patient" (median / mode, stored in the model
metadata) and re-score with the SAME pipeline. The change in predicted
probability is that input's contribution *relative to the typical patient*.

Caveats (also shown in the dashboard):
* contributions are not additive and depend on the reference chosen;
* they describe what the model associates with risk, not what causes it.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .features import (ALL_FEATURES, CONTEXT_FEATURES, EHR_CATEGORICAL, EHR_NUMERIC, FEATURE_LABELS,
                       FEATURE_STREAM, INTERACTION_FEATURES, SENSOR_FEATURES)


def recompute_interactions(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["cgm_minus_fasting"] = X["cgm_current"] - pd.to_numeric(X["fasting_glucose_mg_dl"], errors="coerce")
    X["hba1c_x_carbs_logged"] = pd.to_numeric(X["hba1c_pct"], errors="coerce") * X["carbs_logged_120"] / 100
    return X


def local_explanation(pipe, row: pd.DataFrame, reference: dict[str, Any]) -> pd.DataFrame:
    """One row in -> table of per-feature contributions (probability points)."""
    row = row[ALL_FEATURES].iloc[[0]]
    p0 = float(pipe.predict_proba(row)[0, 1])
    variants = []
    for f in ALL_FEATURES:
        v = row.copy()
        v[f] = reference.get(f, v[f].iloc[0])
        variants.append(v)
    probs = pipe.predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
    out = pd.DataFrame({
        "feature": ALL_FEATURES,
        "label": [FEATURE_LABELS.get(f, f) for f in ALL_FEATURES],
        "stream": [FEATURE_STREAM[f] for f in ALL_FEATURES],
        "value": [row[f].iloc[0] for f in ALL_FEATURES],
        "reference": [reference.get(f) for f in ALL_FEATURES],
        "contribution": p0 - probs,
    })
    out["abs"] = out["contribution"].abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


STREAM_BLOCKS = {
    "EHR (historical)": EHR_NUMERIC + EHR_CATEGORICAL,
    "Sensor (dynamic)": SENSOR_FEATURES,
    "Context (time of day)": CONTEXT_FEATURES,
}


def stream_contributions(pipe, row: pd.DataFrame, reference: dict[str, Any]) -> dict[str, Any]:
    """Probability with each whole stream swapped for the typical patient's values."""
    row = row[ALL_FEATURES].iloc[[0]]
    p0 = float(pipe.predict_proba(row)[0, 1])
    res: dict[str, Any] = {"actual": p0, "streams": {}}
    for name, cols in STREAM_BLOCKS.items():
        v = row.copy()
        for c in cols:
            v[c] = reference.get(c, v[c].iloc[0])
        v = recompute_interactions(v)
        p = float(pipe.predict_proba(v[ALL_FEATURES])[0, 1])
        res["streams"][name] = {"prob_if_typical": p, "contribution": p0 - p}
    allref = pd.DataFrame([{c: reference.get(c) for c in ALL_FEATURES}])
    res["typical_patient_prob"] = float(pipe.predict_proba(allref[ALL_FEATURES])[0, 1])
    return res


__all__ = ["local_explanation", "stream_contributions", "recompute_interactions", "INTERACTION_FEATURES"]
