"""Deterministic synthetic cohort generator.

Produces, for a configurable number of FICTIONAL patients:

* ``ehr_patients.csv``      — one row per patient: static / historical EHR profile
* ``sensor_stream.csv.gz``  — 5-minute CGM, heart rate, HRV and steps
* ``sleep_log.csv``         — one row per night (wearable sleep summary)
* ``meal_log.csv``          — meals the patient chose to log in an app (incomplete, noisy)
* ``meals_ground_truth.csv``— every simulated meal (NOT used as a model input; kept
                              only so the simulation is auditable)

The glucose signal is a simple additive simulator, not a physiological model:

    glucose(t) = basal (from fasting glucose)            [EHR-linked]
               + daily offset + slow AR(1) drift          [unexplained variation]
               + dawn-phenomenon bump                     [patient-specific]
               + sum(meal carbs x carb sensitivity x meal kernel)
               - sum(activity x exercise sensitivity x exercise kernel)
               + stress episodes (also visible in HR / HRV)
               + poor-sleep penalty on next-day sensitivity and basal level

Carb sensitivity (mg/dL peak rise per gram) rises with HbA1c and BMI and is reduced by medication class,
so the EHR genuinely changes how a patient responds to the same day. All
coefficients are hand-picked assumptions documented in docs/data_provenance.md.
They are NOT fitted to any real dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from . import config as C

BINS_PER_DAY = 288  # 5-minute bins
MED_CLASSES = [
    "none",
    "metformin",
    "metformin_dpp4i",
    "metformin_sglt2i",
    "basal_insulin_plus_oral",
]
MED_EFFECT = {  # multiplier on carb sensitivity (assumption, not a drug-effect claim)
    "none": 1.0,
    "metformin": 0.90,
    "metformin_dpp4i": 0.85,
    "metformin_sglt2i": 0.85,
    "basal_insulin_plus_oral": 0.80,
}


@dataclass
class Latent:
    """Hidden simulation parameters (never shown to the model)."""

    basal: float
    carb_sens: float
    meal_peak_min: float
    ex_sens: float
    dawn_amp: float
    sleep_sens: float
    stress_amp: float
    rhr: float
    hrv_base: float
    steps_per_day: float
    p_walk: float
    carb_scale: float
    p_log: float
    sleep_mean_h: float
    bed_min: float


# ----------------------------------------------------------------------------
# EHR
# ----------------------------------------------------------------------------

def _gen_ehr(rng: np.random.Generator, pid: str, p_log_mean: float = 0.7
             ) -> tuple[dict[str, Any], Latent]:
    t2d = rng.random() < 0.70
    sex = str(rng.choice(["F", "M"]))
    age = float(np.clip(rng.normal(56, 10) if t2d else rng.normal(45, 10), 25, 80))
    bmi = float(np.clip(rng.normal(27.5, 4.2), 19, 42))
    years = float(np.clip(rng.gamma(2.0, 3.5), 0.2, max(age - 25, 0.5))) if t2d else 0.0
    if t2d:
        hba1c = float(np.clip(6.4 + rng.gamma(2.2, 0.55) + 0.03 * years + 0.04 * (bmi - 27), 6.5, 12.5))
    else:
        hba1c = float(rng.uniform(5.7, 6.4))
    eag = 28.7 * hba1c - 46.7  # ADAG estimated average glucose (mg/dL)
    fpg = float(np.clip(0.82 * eag + rng.normal(0, 8) + (5 if t2d else 0), 88, 300))

    if not t2d:
        med = "metformin" if rng.random() < 0.2 else "none"
    else:
        p = np.array([0.08, 0.45, 0.20, 0.15, 0.12])
        if hba1c > 9.0:  # sicker patients more often on insulin
            p = np.array([0.04, 0.30, 0.18, 0.18, 0.30])
        med = str(rng.choice(MED_CLASSES, p=p / p.sum()))

    egfr = float(np.clip(rng.normal(95 - 0.6 * (age - 40) - 0.5 * years, 12), 30, 125))
    ldl = float(np.clip(rng.normal(112, 30), 50, 220))
    tg = float(np.clip(rng.normal(140 + 12 * (hba1c - 6) + 2 * (bmi - 27), 45), 50, 450))
    sbp = float(np.clip(rng.normal(126 + 0.35 * (age - 50) + 0.8 * (bmi - 27), 12), 95, 190))
    htn = bool(sbp >= 140 or rng.random() < 0.15)

    ehr = {
        "patient_id": pid,
        "age_years": round(age),
        "sex": sex,
        "bmi_kg_m2": round(bmi, 1),
        "diabetes_status": "type2_diabetes" if t2d else "prediabetes",
        "years_since_diagnosis": round(years, 1),
        "hba1c_pct": round(hba1c, 1),
        "fasting_glucose_mg_dl": round(fpg),
        "egfr_ml_min_1_73m2": round(egfr),
        "ldl_mg_dl": round(ldl),
        "triglycerides_mg_dl": round(tg),
        "systolic_bp_mm_hg": round(sbp),
        "hypertension": int(htn),
        "medication_class": med,
    }

    carb_sens = (0.45 + 0.45 * (hba1c - 5.5) + 0.02 * (bmi - 25)) * MED_EFFECT[med]
    carb_sens *= float(np.exp(rng.normal(0, 0.12)))
    lat = Latent(
        basal=0.0,  # set below
        carb_sens=max(carb_sens, 0.4),
        meal_peak_min=float(np.clip(45 + 5 * (hba1c - 6) + rng.normal(0, 5), 35, 80)),
        ex_sens=float(rng.uniform(4.5, 9.0) * (1.2 if med == "basal_insulin_plus_oral" else 1.0)),
        dawn_amp=float(rng.uniform(0, 25) if t2d else rng.uniform(0, 8)),
        sleep_sens=float(rng.uniform(0.03, 0.08)),
        stress_amp=float(rng.uniform(8, 25)),
        rhr=float(np.clip(rng.normal(70, 6) + 0.4 * (bmi - 27), 52, 92)),
        hrv_base=float(np.clip(rng.normal(44 - 0.35 * (age - 45) - 0.4 * years, 6), 12, 80)),
        steps_per_day=float(rng.uniform(2500, 8000)),
        p_walk=float(rng.uniform(0.2, 0.8)),
        carb_scale=float(rng.uniform(0.8, 1.25)),
        p_log=float(np.clip(rng.beta(10 * p_log_mean, 10 * (1 - p_log_mean)), 0.05, 0.98)),
        sleep_mean_h=float(rng.uniform(5.8, 7.8)),
        bed_min=float(rng.normal(23 * 60, 35)),
    )
    lat.basal = fpg - 0.5 * lat.dawn_amp
    return ehr, lat


# ----------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------

def _gamma_kernel(peak_min: float, length_min: int) -> np.ndarray:
    """Shape t/tp * exp(1 - t/tp): 0 at t=0, peak 1 at t=tp, long tail."""
    t = np.arange(0, length_min + 1, 5, dtype=float)
    return (t / peak_min) * np.exp(1 - t / peak_min)


def _apply_kernel(impulses: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.convolve(impulses, kernel)[: len(impulses)]


# ----------------------------------------------------------------------------
# Per-patient time series
# ----------------------------------------------------------------------------

def _gen_patient_series(
    rng: np.random.Generator, ehr: dict[str, Any], lat: Latent, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dcfg = cfg["data"]
    n_days = int(dcfg["n_days"])
    n = n_days * BINS_PER_DAY
    start = pd.Timestamp(dcfg["start_date"])
    ts = pd.date_range(start, periods=n, freq="5min")
    tmin = np.arange(n) * 5  # minutes since start
    tod = tmin % 1440
    day = tmin // 1440
    pid = ehr["patient_id"]

    asleep = np.zeros(n, dtype=bool)
    steps = np.zeros(n)
    meal_imp = np.zeros(n)
    stress = np.zeros(n)
    day_factor = np.ones(n_days)
    day_basal_shift = np.zeros(n_days)
    sleep_rows, meal_rows, log_rows = [], [], []

    for d in range(n_days):
        weekend = ts[d * BINS_PER_DAY].dayofweek >= 5
        # ---- the night ending on the morning of day d --------------------
        sleep_h = float(np.clip(rng.normal(lat.sleep_mean_h + (0.6 if weekend else 0), 0.9), 3.5, 10))
        bed_abs = (d - 1) * 1440 + lat.bed_min + rng.normal(0, 30) + (30 if weekend else 0)
        wake_abs = bed_abs + sleep_h * 60
        quality = float(np.clip(rng.normal(78, 9) - 6 * max(0, 6.5 - sleep_h), 20, 100))
        sleep_rows.append({
            "patient_id": pid,
            "night_of": (start + pd.Timedelta(days=d - 1)).date().isoformat(),
            "sleep_start": (start + pd.Timedelta(minutes=bed_abs)).floor("min"),
            "sleep_end": (start + pd.Timedelta(minutes=wake_abs)).floor("min"),
            "sleep_duration_h": round(sleep_h, 2),
            "sleep_quality_score": round(quality),
        })
        i0 = int(max(bed_abs, 0) // 5)
        i1 = int(min(wake_abs, n * 5) // 5)
        if i1 > i0:
            asleep[i0:i1] = True
        # poor sleep -> lower insulin sensitivity next day (simulation assumption)
        deficit = max(0.0, 7.0 - sleep_h)
        day_factor[d] = 1 + lat.sleep_sens * deficit + 0.002 * (75 - quality)
        day_basal_shift[d] = 4.0 * deficit

        wake_tod = wake_abs - d * 1440
        # ---- meals ---------------------------------------------------------
        meals = [
            ("breakfast", wake_tod + rng.normal(60, 20), rng.uniform(40, 70)),
            ("lunch", 13.5 * 60 + rng.normal(0, 40), rng.uniform(55, 100)),
            ("dinner", 20.5 * 60 + rng.normal(0, 40), rng.uniform(50, 90)),
        ]
        if rng.random() < 0.5:
            meals.append(("snack", 17 * 60 + rng.normal(0, 50), rng.uniform(15, 35)))
        if weekend:
            meals = [(m, t, c * 1.15) for m, t, c in meals]
        for mtype, mtod, carbs in meals:
            carbs *= lat.carb_scale
            mabs = d * 1440 + float(np.clip(mtod, 0, 1435))
            idx = int(mabs // 5)
            if idx >= n:
                continue
            meal_imp[idx] += carbs * lat.carb_sens * day_factor[d]
            mts = (start + pd.Timedelta(minutes=mabs)).floor("min")
            meal_rows.append({"patient_id": pid, "timestamp": mts, "meal_type": mtype,
                              "carbs_g": round(carbs)})
            if rng.random() < lat.p_log:  # patient logs it (time + carb estimate are noisy)
                lts = mts + pd.Timedelta(minutes=float(np.round(rng.normal(0, 10))))
                log_rows.append({"patient_id": pid, "timestamp": lts, "meal_type": mtype,
                                 "carbs_g_estimated": int(round(carbs * np.exp(rng.normal(0, 0.2))))})
            # post-meal walk
            if rng.random() < lat.p_walk * 0.45:
                ws = idx + int(rng.integers(4, 10))
                wl = int(rng.integers(4, 9))
                steps[ws: ws + wl] += rng.normal(480, 60, size=len(steps[ws: ws + wl]))
        # ---- stress episode -----------------------------------------------
        if rng.random() < 0.25:
            s0 = d * BINS_PER_DAY + int(rng.uniform(9 * 12, 20 * 12))
            sl = int(rng.integers(12, 30))
            stress[s0: s0 + sl] = 1.0

    # ---- background activity (awake only), bursty -------------------------
    awake = ~asleep
    lam = lat.steps_per_day / (16 * 12)
    steps += np.where(awake, rng.gamma(0.6, lam / 0.6, size=n), 0)
    steps = np.clip(steps, 0, 900)

    # ---- glucose ----------------------------------------------------------
    meal_k = _gamma_kernel(lat.meal_peak_min, 300)
    ex_k = _gamma_kernel(25, 150)
    stress_s = _apply_kernel(stress, _gamma_kernel(30, 120)) / 4.0
    glucose = (
        lat.basal
        + np.repeat(rng.normal(0, 6, n_days), BINS_PER_DAY)
        + np.repeat(day_basal_shift, BINS_PER_DAY)
        + lat.dawn_amp * np.exp(-(((tod - 390) / 70.0) ** 2))
        + _apply_kernel(meal_imp, meal_k)
        - _apply_kernel(lat.ex_sens * steps / 500.0, ex_k)
        + lat.stress_amp * np.clip(stress_s, 0, 1.5)
        + lfilter([1.0], [1.0, -0.985], rng.normal(0, 2.2, n))
    )
    glucose = np.clip(glucose, 55, 400)
    cgm = glucose * (1 + rng.normal(0, 0.025, n)) + rng.normal(0, 3, n)
    cgm = np.clip(np.round(cgm), 40, 400)

    # CGM dropouts (sensor gaps of 15 min to 3 h)
    gap_starts = np.where(rng.random(n) < float(dcfg["cgm_dropout_rate"]))[0]
    for g in gap_starts:
        cgm[g: g + int(rng.integers(3, 37))] = np.nan

    # ---- heart rate / HRV -------------------------------------------------
    act = np.clip(steps / 500.0, 0, 1.6)
    act_s = _apply_kernel(act, np.array([0.5, 0.3, 0.2]))
    s_now = np.clip(stress_s, 0, 1.2)
    hr = np.where(asleep, lat.rhr - 6, lat.rhr + 6) + 30 * act_s + 12 * s_now + rng.normal(0, 2.5, n)
    night_deficit = np.repeat(np.maximum(0, 7 - np.array([r["sleep_duration_h"] for r in sleep_rows])),
                              BINS_PER_DAY)
    hrv = lat.hrv_base * np.where(asleep, 1.25, 1.0) * (1 - 0.3 * np.clip(act_s, 0, 1))
    hrv *= (1 - 0.35 * np.clip(s_now, 0, 1)) * (1 - 0.03 * night_deficit)
    hrv *= np.exp(rng.normal(0, 0.12, n))
    hr, hrv = np.round(hr), np.round(hrv, 1)

    # watch off (charging) ~1 h on most days -> HR / HRV / steps missing
    for d in range(n_days):
        if rng.random() < 0.6:
            w0 = d * BINS_PER_DAY + int(rng.uniform(8 * 12, 21 * 12))
            hr[w0: w0 + 12] = np.nan
            hrv[w0: w0 + 12] = np.nan
            steps[w0: w0 + 12] = np.nan

    sensor = pd.DataFrame({
        "patient_id": pid,
        "timestamp": ts,
        "cgm_mg_dl": cgm,
        "heart_rate_bpm": hr,
        "hrv_rmssd_ms": hrv,
        "steps": np.round(steps),
    })
    return sensor, pd.DataFrame(sleep_rows), pd.DataFrame(log_rows), pd.DataFrame(meal_rows)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def generate_cohort(cfg: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """Generate the full synthetic cohort in memory (deterministic for a seed)."""
    cfg = cfg or C.load_config()
    seed = int(cfg["seed"])
    ehrs, sensors, sleeps, logs, truths = [], [], [], [], []
    for i in range(int(cfg["data"]["n_patients"])):
        rng = np.random.default_rng([seed, i])  # per-patient stream -> stable IDs
        pid = f"SYN-{i + 1:04d}"
        ehr, lat = _gen_ehr(rng, pid, float(cfg["data"]["meal_log_probability"]))
        s, sl, lg, mt = _gen_patient_series(rng, ehr, lat, cfg)
        ehrs.append(ehr); sensors.append(s); sleeps.append(sl); logs.append(lg); truths.append(mt)
    meal_cols = ["patient_id", "timestamp", "meal_type", "carbs_g_estimated"]
    log_df = pd.concat([x for x in logs if len(x)], ignore_index=True) if any(len(x) for x in logs) \
        else pd.DataFrame(columns=meal_cols)
    return {
        "ehr": pd.DataFrame(ehrs),
        "sensor": pd.concat(sensors, ignore_index=True),
        "sleep": pd.concat(sleeps, ignore_index=True),
        "meal_log": log_df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True),
        "meal_truth": pd.concat(truths, ignore_index=True),
    }


def validate_cohort(data: dict[str, pd.DataFrame]) -> list[str]:
    """Return a list of validation problems (empty list = valid)."""
    problems: list[str] = []
    ehr, s = data["ehr"], data["sensor"]
    if ehr["patient_id"].duplicated().any():
        problems.append("duplicate patient_id in EHR")
    ids = set(ehr["patient_id"])
    for name in ("sensor", "sleep", "meal_log", "meal_truth"):
        extra = set(data[name]["patient_id"]) - ids
        if extra:
            problems.append(f"{name} has patient_ids not in EHR: {sorted(extra)[:3]}")
    if set(s["patient_id"]) != ids:
        problems.append("some EHR patients have no sensor stream")
    ranges = {"cgm_mg_dl": (40, 400), "heart_rate_bpm": (30, 220), "hrv_rmssd_ms": (1, 250),
              "steps": (0, 1000)}
    for col, (lo, hi) in ranges.items():
        v = s[col].dropna()
        if len(v) and (v.min() < lo or v.max() > hi):
            problems.append(f"{col} outside [{lo}, {hi}]")
    for pid, g in s.groupby("patient_id"):
        dt = g["timestamp"].diff().dropna()
        if not (dt == pd.Timedelta(minutes=5)).all():
            problems.append(f"{pid}: irregular or unsorted timestamps")
            break
    if (data["sleep"]["sleep_end"] <= data["sleep"]["sleep_start"]).any():
        problems.append("sleep_end not after sleep_start")
    if not ehr["hba1c_pct"].between(4, 15).all():
        problems.append("hba1c out of range")
    return problems


def save_cohort(data: dict[str, pd.DataFrame], out_dir: Path | None = None) -> Path:
    out = out_dir or C.DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    data["ehr"].to_csv(out / C.EHR_FILE, index=False)
    data["sensor"].to_csv(out / C.SENSOR_FILE, index=False, compression={"method": "gzip", "mtime": 0})
    data["sleep"].to_csv(out / C.SLEEP_FILE, index=False)
    data["meal_log"].to_csv(out / C.MEAL_LOG_FILE, index=False)
    data["meal_truth"].to_csv(out / C.MEAL_TRUTH_FILE, index=False)
    return out


def load_cohort(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    d = data_dir or C.DATA_DIR
    missing = [f for f in (C.EHR_FILE, C.SENSOR_FILE, C.SLEEP_FILE, C.MEAL_LOG_FILE)
               if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Synthetic data not found in {d} (missing {missing}). Run `make data` first.")
    return {
        "ehr": pd.read_csv(d / C.EHR_FILE),
        "sensor": pd.read_csv(d / C.SENSOR_FILE, parse_dates=["timestamp"]),
        "sleep": pd.read_csv(d / C.SLEEP_FILE, parse_dates=["sleep_start", "sleep_end"]),
        "meal_log": pd.read_csv(d / C.MEAL_LOG_FILE, parse_dates=["timestamp"]),
    }
