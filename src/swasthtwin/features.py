"""Leakage-safe feature engineering and label construction.

Rule enforced everywhere in this module: a feature at time ``t`` may use only
observations with timestamp <= t. All rolling windows are *trailing* windows on
a regular 5-minute grid. Logged meals are assigned to the first grid point at
or after the moment they were logged (ceil), so a meal logged at 10:03 is not
visible to a 10:00 prediction. Labels (future CGM) are computed in a separate
function and are never passed to the feature builder.

The same ``build_patient_features`` function is used for (a) the offline
training table and (b) the live Digital Twin, so training and inference cannot
silently drift apart.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

GRID = "5min"
BIN_MIN = 5

EHR_NUMERIC = [
    "age_years", "bmi_kg_m2", "years_since_diagnosis", "hba1c_pct",
    "fasting_glucose_mg_dl", "egfr_ml_min_1_73m2", "ldl_mg_dl",
    "triglycerides_mg_dl", "systolic_bp_mm_hg", "hypertension",
]
EHR_CATEGORICAL = ["sex", "diabetes_status", "medication_class"]

SENSOR_FEATURES = [
    # CGM
    "cgm_current", "cgm_minutes_since_last", "cgm_roc_15", "cgm_slope_30",
    "cgm_mean_60", "cgm_std_60", "cgm_mean_120", "cgm_max_120", "cgm_min_120",
    "cgm_tir_120", "cgm_coverage_120",
    # wearable
    "hr_mean_30", "hr_mean_120", "hrv_mean_60", "hrv_rel_24h",
    "steps_30", "steps_120", "active_min_120", "wear_coverage_120",
    # sleep
    "sleep_last_h", "sleep_quality_last", "hours_since_wake",
    # optional meal log (self-reported)
    "carbs_logged_120", "minutes_since_meal_log",
    ]
CONTEXT_FEATURES = ["hour_sin", "hour_cos", "is_weekend"]  # clock time, not a sensor
INTERACTION_FEATURES = ["cgm_minus_fasting", "hba1c_x_carbs_logged"]

NUMERIC_FEATURES = EHR_NUMERIC + SENSOR_FEATURES + CONTEXT_FEATURES + INTERACTION_FEATURES
CATEGORICAL_FEATURES = EHR_CATEGORICAL
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_STREAM = {f: "EHR (historical)" for f in EHR_NUMERIC + EHR_CATEGORICAL}
FEATURE_STREAM.update({f: "Sensor (dynamic)" for f in SENSOR_FEATURES})
FEATURE_STREAM.update({f: "Context (time of day)" for f in CONTEXT_FEATURES})
FEATURE_STREAM.update({f: "Fusion (EHR x sensor)" for f in INTERACTION_FEATURES})

FEATURE_LABELS = {
    "cgm_current": "Latest CGM glucose (mg/dL)",
    "cgm_minutes_since_last": "Minutes since last CGM reading",
    "cgm_roc_15": "CGM rate of change, last 15 min (mg/dL/min)",
    "cgm_slope_30": "CGM trend, last 30 min (mg/dL/min)",
    "cgm_mean_60": "CGM mean, last 60 min",
    "cgm_std_60": "CGM variability (SD), last 60 min",
    "cgm_mean_120": "CGM mean, last 120 min",
    "cgm_max_120": "CGM max, last 120 min",
    "cgm_min_120": "CGM min, last 120 min",
    "cgm_tir_120": "Time in 70-180 range, last 120 min (fraction)",
    "cgm_coverage_120": "CGM data coverage, last 120 min",
    "hr_mean_30": "Heart rate, last 30 min (bpm)",
    "hr_mean_120": "Heart rate, last 120 min (bpm)",
    "hrv_mean_60": "HRV (RMSSD), last 60 min (ms)",
    "hrv_rel_24h": "HRV last 60 min vs own 24 h mean",
    "steps_30": "Steps, last 30 min",
    "steps_120": "Steps, last 120 min",
    "active_min_120": "Active minutes, last 120 min",
    "wear_coverage_120": "Wearable coverage, last 120 min",
    "sleep_last_h": "Last night's sleep (h)",
    "sleep_quality_last": "Last night's sleep quality (0-100)",
    "hours_since_wake": "Hours since waking",
    "carbs_logged_120": "Logged carbs, last 120 min (g)",
    "minutes_since_meal_log": "Minutes since last logged meal",
    "hour_sin": "Time of day (sin)",
    "hour_cos": "Time of day (cos)",
    "is_weekend": "Weekend",
    "cgm_minus_fasting": "Latest CGM minus EHR fasting glucose",
    "hba1c_x_carbs_logged": "HbA1c x logged carbs (load x control)",
    "age_years": "Age (years)",
    "bmi_kg_m2": "BMI (kg/m2)",
    "years_since_diagnosis": "Years since diagnosis",
    "hba1c_pct": "HbA1c (%)",
    "fasting_glucose_mg_dl": "Fasting glucose (mg/dL)",
    "egfr_ml_min_1_73m2": "eGFR",
    "ldl_mg_dl": "LDL cholesterol",
    "triglycerides_mg_dl": "Triglycerides",
    "systolic_bp_mm_hg": "Systolic BP",
    "hypertension": "Hypertension recorded",
    "sex": "Sex",
    "diabetes_status": "Diabetes status",
    "medication_class": "Medication class",
}

MEAL_LOG_CAP_MIN = 720.0


def _regularise(sensor_p: pd.DataFrame) -> pd.DataFrame:
    """Sort, de-duplicate and reindex one patient's stream onto a 5-min grid."""
    s = sensor_p.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last")
    s = s.set_index("timestamp")
    s.index = s.index.floor(GRID)
    s = s[~s.index.duplicated(keep="last")]
    full = pd.date_range(s.index.min(), s.index.max(), freq=GRID)
    s = s.reindex(full)
    s.index.name = "timestamp"
    for col in ("cgm_mg_dl", "heart_rate_bpm", "hrv_rmssd_ms", "steps"):
        if col not in s:
            s[col] = np.nan
        s[col] = pd.to_numeric(s[col], errors="coerce")
    # treat physiologically impossible values as missing (data-quality guard)
    s.loc[~s["cgm_mg_dl"].between(20, 600), "cgm_mg_dl"] = np.nan
    s.loc[~s["heart_rate_bpm"].between(25, 250), "heart_rate_bpm"] = np.nan
    s.loc[~s["hrv_rmssd_ms"].between(1, 300), "hrv_rmssd_ms"] = np.nan
    s.loc[s["steps"] < 0, "steps"] = np.nan
    return s


def _minutes_since_last(valid: pd.Series) -> pd.Series:
    pos = np.arange(len(valid), dtype=float)
    last = pd.Series(np.where(valid.to_numpy(), pos, np.nan), index=valid.index).ffill()
    return (pos - last) * BIN_MIN


def build_patient_features(
    sensor_p: pd.DataFrame,
    sleep_p: pd.DataFrame,
    meal_log_p: pd.DataFrame,
    ehr_row: dict[str, Any] | pd.Series,
    sample_times: Iterable[pd.Timestamp] | None = None,
    sample_every_min: int = 30,
    min_history_min: int = 120,
) -> pd.DataFrame:
    """Compute the feature table for ONE patient.

    ``sample_times``: timestamps at which to emit a row. If None, emit one row
    every ``sample_every_min`` minutes after ``min_history_min`` of history.
    Only data with timestamp <= sample time influences that row.
    """
    s = _regularise(sensor_p)
    if s.empty:
        raise ValueError("empty sensor stream")
    cgm, hr, hrv, st = s["cgm_mg_dl"], s["heart_rate_bpm"], s["hrv_rmssd_ms"], s["steps"]
    roll = lambda x, k: x.rolling(k, min_periods=1)  # noqa: E731  (trailing only)

    f = pd.DataFrame(index=s.index)
    cur = cgm.ffill()
    f["cgm_current"] = cur
    f["cgm_minutes_since_last"] = _minutes_since_last(cgm.notna())
    f["cgm_roc_15"] = (cur - cur.shift(3)) / 15.0
    f["cgm_slope_30"] = (roll(cgm, 3).mean() - roll(cgm.shift(6), 3).mean()) / 30.0
    f["cgm_mean_60"] = roll(cgm, 12).mean()
    f["cgm_std_60"] = cgm.rolling(12, min_periods=3).std()
    f["cgm_mean_120"] = roll(cgm, 24).mean()
    f["cgm_max_120"] = roll(cgm, 24).max()
    f["cgm_min_120"] = roll(cgm, 24).min()
    in_range = ((cgm >= 70) & (cgm < 180)).astype(float).where(cgm.notna())
    f["cgm_tir_120"] = roll(in_range, 24).mean()
    f["cgm_coverage_120"] = roll(cgm.notna().astype(float), 24).mean()

    f["hr_mean_30"] = roll(hr, 6).mean()
    f["hr_mean_120"] = roll(hr, 24).mean()
    f["hrv_mean_60"] = roll(hrv, 12).mean()
    f["hrv_rel_24h"] = f["hrv_mean_60"] / roll(hrv, 288).mean()
    f["steps_30"] = roll(st, 6).sum()
    f["steps_120"] = roll(st, 24).sum()
    f["active_min_120"] = roll((st >= 250).astype(float), 24).sum() * BIN_MIN
    f["wear_coverage_120"] = roll(hr.notna().astype(float), 24).mean()

    # ---- sleep: most recent sleep episode that has ENDED by t --------------
    idx_df = pd.DataFrame({"timestamp": s.index})
    if sleep_p is not None and len(sleep_p):
        sl = sleep_p[["sleep_end", "sleep_duration_h", "sleep_quality_score"]].copy()
        sl["sleep_end"] = pd.to_datetime(sl["sleep_end"]).astype(idx_df["timestamp"].dtype)
        sl = sl.sort_values("sleep_end")
        m = pd.merge_asof(idx_df, sl, left_on="timestamp", right_on="sleep_end", direction="backward")
        f["sleep_last_h"] = m["sleep_duration_h"].to_numpy()
        f["sleep_quality_last"] = m["sleep_quality_score"].to_numpy()
        f["hours_since_wake"] = ((m["timestamp"] - m["sleep_end"]).dt.total_seconds() / 3600).to_numpy()
    else:
        f["sleep_last_h"] = np.nan
        f["sleep_quality_last"] = np.nan
        f["hours_since_wake"] = np.nan

    # ---- meal log: available from the first grid point AT/AFTER logging ----
    carbs = pd.Series(0.0, index=s.index)
    if meal_log_p is not None and len(meal_log_p):
        ml = meal_log_p.copy()
        ml["bin"] = pd.to_datetime(ml["timestamp"]).dt.ceil(GRID)
        ml = ml[(ml["bin"] >= s.index.min()) & (ml["bin"] <= s.index.max())]
        if len(ml):
            agg = ml.groupby("bin")["carbs_g_estimated"].sum()
            agg.index = agg.index.astype(s.index.dtype)
            carbs = carbs.add(agg, fill_value=0.0).reindex(s.index).fillna(0.0)
    f["carbs_logged_120"] = carbs.rolling(24, min_periods=1).sum()
    f["minutes_since_meal_log"] = _minutes_since_last(carbs > 0).fillna(MEAL_LOG_CAP_MIN) \
        .clip(upper=MEAL_LOG_CAP_MIN)

    hours = s.index.hour + s.index.minute / 60.0
    f["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    f["is_weekend"] = (s.index.dayofweek >= 5).astype(int)

    # ---- static EHR (historical) + fusion features -------------------------
    ehr = dict(ehr_row)
    for col in EHR_NUMERIC + EHR_CATEGORICAL:
        f[col] = ehr.get(col, np.nan)
    f["cgm_minus_fasting"] = f["cgm_current"] - pd.to_numeric(f["fasting_glucose_mg_dl"], errors="coerce")
    f["hba1c_x_carbs_logged"] = pd.to_numeric(f["hba1c_pct"], errors="coerce") * f["carbs_logged_120"] / 100.0

    # ---- choose output rows -------------------------------------------------
    if sample_times is None:
        start = s.index.min() + pd.Timedelta(minutes=min_history_min)
        times = s.index[(s.index >= start) & (s.index.minute % sample_every_min == 0)]
    else:
        times = pd.DatetimeIndex(pd.to_datetime(list(sample_times))).floor(GRID)
        times = times.astype(s.index.dtype)
        times = times[(times >= s.index.min()) & (times <= s.index.max())]
    out = f.loc[times].copy()
    out.insert(0, "patient_id", ehr.get("patient_id"))
    out.index.name = "timestamp"
    return out.reset_index()


def make_labels(sensor_p: pd.DataFrame, threshold: float, horizon_min: int,
                min_coverage: float) -> pd.DataFrame:
    """Label for each grid time t: does any observed CGM in (t, t+horizon] reach
    ``threshold``? NaN if too few future readings exist. Uses FUTURE data — never
    pass these columns to the model as inputs."""
    s = _regularise(sensor_p)
    g = s["cgm_mg_dl"].to_numpy()
    k = horizon_min // BIN_MIN
    pad = np.concatenate([g[1:], np.full(k, np.nan)])  # value at t+1 ... (shifted)
    win = np.lib.stride_tricks.sliding_window_view(pad, k)[: len(g)]
    cover = np.sum(~np.isnan(win), axis=1) / k
    with np.errstate(all="ignore"):
        fmax = np.where(cover > 0, np.nanmax(np.where(np.isnan(win), -np.inf, win), axis=1), np.nan)
    label = np.where(cover >= min_coverage, (fmax >= threshold).astype(float), np.nan)
    return pd.DataFrame({"timestamp": s.index, "future_max_cgm_120": fmax,
                         "future_coverage": cover, "label": label})


def build_dataset(cohort: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> pd.DataFrame:
    """Feature + label table for the whole cohort (one row per prediction time)."""
    fc, tc = cfg["features"], cfg["target"]
    sensor = cohort["sensor"]
    sleep_g = dict(tuple(cohort["sleep"].groupby("patient_id")))
    meal_g = dict(tuple(cohort["meal_log"].groupby("patient_id")))
    start = sensor["timestamp"].min().normalize()
    rows = []
    for _, ehr in cohort["ehr"].iterrows():
        pid = ehr["patient_id"]
        sp = sensor[sensor["patient_id"] == pid]
        feats = build_patient_features(sp, sleep_g.get(pid), meal_g.get(pid), ehr,
                                       sample_every_min=int(fc["sample_every_min"]),
                                       min_history_min=int(fc["min_history_min"]))
        lab = make_labels(sp, float(tc["threshold_mg_dl"]), int(tc["horizon_min"]),
                          float(tc["min_future_coverage"]))
        lab["timestamp"] = lab["timestamp"].astype(feats["timestamp"].dtype)
        rows.append(feats.merge(lab, on="timestamp", how="left"))
    df = pd.concat(rows, ignore_index=True)
    df["study_day"] = (df["timestamp"] - start).dt.days
    df = df[df["label"].notna()].reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df
