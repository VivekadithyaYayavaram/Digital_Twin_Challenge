"""Patient-specific Digital Twin state manager.

A ``PatientTwin`` keeps a virtual state for ONE patient:

1. the historical EHR profile (static);
2. the sensor observations received so far (everything up to ``cursor``);
3. rolling-window features recomputed from (2) with the SAME function used in
   training (``features.build_patient_features``);
4. the trained prediction pipeline and its metadata;
5. the latest forecast with its timestamp, risk band and data-quality warnings;
6. a history of every forecast the twin has produced.

Where do new observations come from? This prototype has no live device
integration. Observations arrive from either
* a deterministic **replay** of the pre-generated synthetic stream
  (``advance``), which stands in for a CGM / wearable feed; or
* a **manual observation** pushed into the twin (``ingest_observation``).

``scenario`` answers "what would the model say if these recent inputs had been
different?" by editing a COPY of the observed inputs and re-scoring with the
same model. It shows model sensitivity, not a causal or physiological effect.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .explain import local_explanation, stream_contributions
from .features import ALL_FEATURES, EHR_CATEGORICAL, EHR_NUMERIC, build_patient_features

SENSOR_FIELDS = ("cgm_mg_dl", "heart_rate_bpm", "hrv_rmssd_ms", "steps")
VALID_RANGES = {"cgm_mg_dl": (40, 400), "heart_rate_bpm": (30, 220),
                "hrv_rmssd_ms": (1, 250), "steps": (0, 1000)}
SCENARIOS = {
    "walk_30min": "Add a brisk 30-minute walk (~480 steps per 5 min) to the last 30 minutes",
    "short_sleep": "Replace last night's sleep with a short, poor night",
    "logged_meal": "Log a carbohydrate-rich meal right now",
    "no_activity_2h": "Set steps in the last 2 hours to zero (sedentary)",
}


def load_model(artifact_dir: Path | None = None):
    d = artifact_dir or C.ARTIFACT_DIR
    mp, meta_p = d / C.MODEL_FILE, d / C.MODEL_META_FILE
    if not mp.exists() or not meta_p.exists():
        raise FileNotFoundError(f"Trained model not found in {d}. Run `make train` first.")
    return joblib.load(mp), json.loads(meta_p.read_text())


def risk_band(p: float, cfg: dict[str, Any]) -> str:
    rb = cfg["risk_bands"]
    if p < rb["low_max"]:
        return "Low"
    if p < rb["moderate_max"]:
        return "Moderate"
    return "Elevated"


@dataclass
class Forecast:
    as_of: pd.Timestamp
    probability: float
    risk_band: str
    above_alert_threshold: bool
    latest_cgm: float | None
    latest_cgm_time: pd.Timestamp | None
    source: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PatientTwin:
    def __init__(self, patient_id: str, cohort: dict[str, pd.DataFrame], pipe, meta: dict[str, Any],
                 cfg: dict[str, Any] | None = None, start_time: pd.Timestamp | None = None):
        self.cfg = cfg or C.load_config()
        ehr = cohort["ehr"]
        if patient_id not in set(ehr["patient_id"]):
            raise KeyError(f"unknown patient_id {patient_id!r}")
        self.patient_id = patient_id
        self.ehr: dict[str, Any] = ehr[ehr.patient_id == patient_id].iloc[0].to_dict()
        self._feed = cohort["sensor"][cohort["sensor"].patient_id == patient_id] \
            .sort_values("timestamp").reset_index(drop=True)
        self._sleep = cohort["sleep"][cohort["sleep"].patient_id == patient_id].reset_index(drop=True)
        self._meals = cohort["meal_log"][cohort["meal_log"].patient_id == patient_id].reset_index(drop=True)
        self._overlay: list[dict[str, Any]] = []      # manually ingested observations
        self._overlay_meals: list[dict[str, Any]] = []
        self.pipe, self.meta = pipe, meta
        self.threshold = float(meta["operating_threshold"])
        self.feed_start = self._feed["timestamp"].min()
        self.feed_end = self._feed["timestamp"].max()
        self.min_cursor = self.feed_start + pd.Timedelta(minutes=int(self.cfg["features"]["min_history_min"]))
        default = self.feed_start.normalize() + pd.Timedelta(days=2, hours=7)
        self.cursor: pd.Timestamp = self._clamp(start_time if start_time is not None else default)
        self.history: list[Forecast] = []
        self.latest: Forecast | None = None
        self.latest_features: pd.DataFrame | None = None

    # ------------------------------------------------------------------ state
    def _clamp(self, t) -> pd.Timestamp:
        t = pd.Timestamp(t).floor("5min")
        return min(max(t, self.min_cursor), self.feed_end)

    def observed_sensor(self) -> pd.DataFrame:
        """Everything the twin has 'received' up to the cursor (feed + manual)."""
        obs = self._feed[self._feed.timestamp <= self.cursor].set_index("timestamp").copy()
        for o in self._overlay:
            t = o["timestamp"]
            if t > self.cursor:
                continue
            if t not in obs.index:
                obs.loc[t] = {"patient_id": self.patient_id, **{f: np.nan for f in SENSOR_FIELDS}}
            for f in SENSOR_FIELDS:
                if o.get(f) is not None:
                    obs.loc[t, f] = o[f]
        return obs.sort_index().reset_index()

    def observed_sleep(self) -> pd.DataFrame:
        return self._sleep[self._sleep.sleep_end <= self.cursor]

    def observed_meals(self) -> pd.DataFrame:
        m = self._meals[self._meals.timestamp <= self.cursor]
        if self._overlay_meals:
            m = pd.concat([m, pd.DataFrame(self._overlay_meals)], ignore_index=True)
            m = m[m.timestamp <= self.cursor]
        return m

    def revealed_future(self, minutes: int = 120) -> pd.DataFrame:
        """Synthetic 'what actually happened next' for demo/teaching. NEVER used
        as model input; only available because the stream is a replay."""
        end = self.cursor + pd.Timedelta(minutes=minutes)
        return self._feed[(self._feed.timestamp > self.cursor) & (self._feed.timestamp <= end)]

    # -------------------------------------------------------------- features
    def compute_features(self, sensor=None, sleep=None, meals=None) -> pd.DataFrame:
        sensor = self.observed_sensor() if sensor is None else sensor
        sleep = self.observed_sleep() if sleep is None else sleep
        meals = self.observed_meals() if meals is None else meals
        if sensor.empty:
            raise ValueError("no sensor observations available yet")
        f = build_patient_features(sensor, sleep, meals, self.ehr, sample_times=[self.cursor])
        if f.empty:
            raise ValueError(f"cannot compute features at {self.cursor}")
        return f

    def _quality_warnings(self, f: pd.Series) -> list[str]:
        w = []
        stale = float(self.cfg["features"]["max_staleness_min"])
        if pd.isna(f["cgm_current"]):
            w.append("No CGM reading available — prediction relies on EHR, wearable and time context only.")
        elif f["cgm_minutes_since_last"] > stale:
            w.append(f"Latest CGM reading is {f['cgm_minutes_since_last']:.0f} min old (stale > {stale:.0f} min).")
        if f["cgm_coverage_120"] < 0.5:
            w.append(f"Only {f['cgm_coverage_120']:.0%} of expected CGM readings in the last 2 h.")
        if pd.isna(f["wear_coverage_120"]) or f["wear_coverage_120"] < 0.5:
            w.append("Wearable (HR/HRV/steps) mostly missing in the last 2 h — watch may be off.")
        if pd.isna(f["sleep_last_h"]):
            w.append("No completed sleep record yet.")
        miss = [c for c in EHR_NUMERIC + EHR_CATEGORICAL if pd.isna(self.ehr.get(c))]
        if miss:
            w.append(f"EHR fields missing (imputed): {', '.join(miss)}")
        return w

    def _score(self, f: pd.DataFrame, source: str) -> Forecast:
        p = float(self.pipe.predict_proba(f[ALL_FEATURES])[0, 1])
        r = f.iloc[0]
        cgm_t = None
        if not pd.isna(r["cgm_current"]):
            cgm_t = self.cursor - pd.Timedelta(minutes=float(r["cgm_minutes_since_last"]))
        return Forecast(as_of=self.cursor, probability=p, risk_band=risk_band(p, self.cfg),
                        above_alert_threshold=p >= self.threshold,
                        latest_cgm=None if pd.isna(r["cgm_current"]) else float(r["cgm_current"]),
                        latest_cgm_time=cgm_t, source=source, warnings=self._quality_warnings(r))

    def refresh(self, source: str = "replay") -> Forecast:
        """Recompute features from current observations and re-run the model."""
        f = self.compute_features()
        fc = self._score(f, source)
        self.latest, self.latest_features = fc, f
        self.history.append(fc)
        return fc

    # ---------------------------------------------------------------- updates
    def advance(self, minutes: int = 30) -> Forecast:
        """Replay the next ``minutes`` of the synthetic sensor stream, then refresh."""
        if minutes <= 0:
            raise ValueError("minutes must be positive")
        self.cursor = self._clamp(self.cursor + pd.Timedelta(minutes=minutes))
        return self.refresh("replay")

    def jump_to(self, t) -> Forecast:
        self.cursor = self._clamp(t)
        return self.refresh("replay")

    @property
    def at_end(self) -> bool:
        return self.cursor >= self.feed_end

    def ingest_observation(self, timestamp=None, **values: float) -> Forecast:
        """Push a manual observation (e.g. a fingerstick-confirmed CGM value).
        Validates ranges; unknown fields or impossible values raise ValueError."""
        unknown = set(values) - set(SENSOR_FIELDS)
        if unknown:
            raise ValueError(f"unknown sensor fields: {sorted(unknown)}")
        if not any(v is not None for v in values.values()):
            raise ValueError("no values supplied")
        for k, v in values.items():
            if v is None:
                continue
            lo, hi = VALID_RANGES[k]
            if not (isinstance(v, (int, float)) and np.isfinite(v) and lo <= v <= hi):
                raise ValueError(f"{k}={v!r} outside valid range [{lo}, {hi}]")
        t = pd.Timestamp(timestamp).floor("5min") if timestamp is not None \
            else self.cursor + pd.Timedelta(minutes=5)
        if t < self.cursor - pd.Timedelta(hours=2):
            raise ValueError("observation is more than 2 h older than the twin's clock")
        self._overlay.append({"timestamp": t, **values})
        self.cursor = max(self.cursor, min(t, self.feed_end))
        return self.refresh("manual")

    def log_meal(self, carbs_g: float, timestamp=None) -> Forecast:
        if not (0 < carbs_g <= 300):
            raise ValueError("carbs_g must be in (0, 300]")
        t = pd.Timestamp(timestamp) if timestamp is not None else self.cursor
        self._overlay_meals.append({"patient_id": self.patient_id, "timestamp": t,
                                    "meal_type": "manual", "carbs_g_estimated": carbs_g})
        return self.refresh("manual")

    # -------------------------------------------------------------- scenarios
    def scenario(self, kind: str, sleep_h: float = 4.5, carbs_g: float = 80) -> dict[str, Any]:
        """Score an edited COPY of the recent inputs. Twin state is not changed."""
        if kind not in SCENARIOS:
            raise ValueError(f"unknown scenario {kind!r}; choose from {list(SCENARIOS)}")
        base_f = self.compute_features()
        base = self._score(base_f, "baseline")
        sensor, sleep, meals = self.observed_sensor(), self.observed_sleep().copy(), self.observed_meals().copy()
        if kind == "walk_30min":
            m = sensor.timestamp > self.cursor - pd.Timedelta(minutes=30)
            sensor.loc[m, "steps"] = sensor.loc[m, "steps"].fillna(0) + 480
            sensor["steps"] = sensor["steps"].clip(upper=1000)
            sensor.loc[m, "heart_rate_bpm"] = sensor.loc[m, "heart_rate_bpm"] + 25
        elif kind == "no_activity_2h":
            m = sensor.timestamp > self.cursor - pd.Timedelta(minutes=120)
            sensor.loc[m, "steps"] = 0
        elif kind == "short_sleep":
            if sleep.empty:
                raise ValueError("no completed sleep record to modify")
            i = sleep.sleep_end.idxmax()
            sleep.loc[i, "sleep_duration_h"] = float(sleep_h)
            sleep.loc[i, "sleep_quality_score"] = 45
        elif kind == "logged_meal":
            meals = pd.concat([meals, pd.DataFrame([{"patient_id": self.patient_id, "timestamp": self.cursor,
                                                     "meal_type": "scenario",
                                                     "carbs_g_estimated": float(carbs_g)}])],
                              ignore_index=True)
        scen_f = self.compute_features(sensor, sleep, meals)
        scen = self._score(scen_f, f"scenario:{kind}")
        changed = [c for c in ALL_FEATURES
                   if not (pd.isna(base_f[c].iloc[0]) and pd.isna(scen_f[c].iloc[0]))
                   and base_f[c].iloc[0] != scen_f[c].iloc[0]]
        return {"kind": kind, "description": SCENARIOS[kind], "baseline": base, "scenario": scen,
                "delta": scen.probability - base.probability,
                "changed_features": pd.DataFrame({
                    "feature": changed,
                    "baseline": [base_f[c].iloc[0] for c in changed],
                    "scenario": [scen_f[c].iloc[0] for c in changed]})}

    # ------------------------------------------------------------ explanation
    def explain(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        f = self.latest_features if self.latest_features is not None else self.compute_features()
        ref = self.meta["reference_row"]
        return local_explanation(self.pipe, f, ref), stream_contributions(self.pipe, f, ref)

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame([h.to_dict() for h in self.history])

    def state(self) -> dict[str, Any]:
        return {"patient_id": self.patient_id, "clock": self.cursor, "ehr": self.ehr,
                "latest_forecast": self.latest.to_dict() if self.latest else None,
                "n_forecasts": len(self.history), "model_version": self.meta.get("model_version"),
                "selected_model": self.meta.get("selected_model_label"),
                "manual_observations": len(self._overlay) + len(self._overlay_meals)}
