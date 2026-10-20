# Data dictionary

All files in `data/` are **synthetic**, produced by `make data` (`src/swasthtwin/data_generator.py`) from the
seed in `config/model_config.yaml`. No field describes a real person.

## `data/ehr_patients.csv` — historical / static EHR (one row per patient)

| Column | Unit | How it is produced |
|---|---|---|
| `patient_id` | — | `SYN-0001` … sequential fictional ID |
| `age_years` | years | Normal(56, 10) for T2D, Normal(45, 10) for prediabetes, clipped 25–80 |
| `sex` | F / M | 50 / 50 |
| `bmi_kg_m2` | kg/m² | Normal(27.5, 4.2), clipped 19–42 |
| `diabetes_status` | `type2_diabetes` / `prediabetes` | 70 % / 30 % |
| `years_since_diagnosis` | years | Gamma(2, 3.5) for T2D, 0 for prediabetes |
| `hba1c_pct` | % | T2D: 6.4 + Gamma(2.2, 0.55) + small age/BMI terms, 6.5–12.5; prediabetes: Uniform(5.7, 6.4) |
| `fasting_glucose_mg_dl` | mg/dL | 0.82 × ADAG estimated average glucose (28.7 × HbA1c − 46.7) + noise |
| `egfr_ml_min_1_73m2` | mL/min/1.73 m² | Declines with age and diabetes duration + noise |
| `ldl_mg_dl` | mg/dL | Normal(112, 30) |
| `triglycerides_mg_dl` | mg/dL | Rises with HbA1c and BMI |
| `systolic_bp_mm_hg` | mmHg | Rises with age and BMI |
| `hypertension` | 0/1 | SBP ≥ 140 or 15 % random |
| `medication_class` | category | `none`, `metformin`, `metformin_dpp4i`, `metformin_sglt2i`, `basal_insulin_plus_oral`; insulin more likely when HbA1c > 9 |

## `data/sensor_stream.csv.gz` — dynamic sensor stream (5-minute grid)

| Column | Unit | Notes |
|---|---|---|
| `patient_id` | — | links to EHR |
| `timestamp` | local time, no time zone | regular 5-min grid, 14 days starting 2026-03-02 (fictional) |
| `cgm_mg_dl` | mg/dL | simulated glucose + 2.5 % multiplicative and ±3 mg/dL additive noise, clipped 40–400, rounded; **missing** during sensor gaps (15 min–3 h) |
| `heart_rate_bpm` | bpm | resting HR ± sleep, + activity, + stress; missing when the watch is off (~1 h on 60 % of days) |
| `hrv_rmssd_ms` | ms | baseline falls with age and diabetes duration; higher asleep; lower with activity, stress and short sleep |
| `steps` | steps per 5 min | 0 while asleep; bursty background + walks; missing when the watch is off |

## `data/sleep_log.csv` — one row per night (wearable summary)

| Column | Unit | Notes |
|---|---|---|
| `night_of` | date | evening the sleep started |
| `sleep_start`, `sleep_end` | timestamp | `sleep_end` is when the record becomes available to the model |
| `sleep_duration_h` | hours | patient mean 5.8–7.8 h, +0.6 h on weekends, SD 0.9 h |
| `sleep_quality_score` | 0–100 | lower for short nights |

## `data/meal_log.csv` — meals the patient chose to log (optional input)

| Column | Unit | Notes |
|---|---|---|
| `timestamp` | timestamp | true meal time ± ~10 min |
| `meal_type` | breakfast / lunch / snack / dinner | |
| `carbs_g_estimated` | g | patient's estimate: true carbs × log-normal error (~20 %) |

Each patient logs a meal with a patient-specific probability (mean 0.7). Unlogged meals still change glucose.

## `data/meals_ground_truth.csv` — every simulated meal (audit only)

Kept so the simulation can be inspected. **Never used as a model input.**

## Derived tables (not saved; rebuilt by `make train` / `make evaluate`)

The feature table has one row per patient every 30 minutes (after 2 h of history). Feature definitions are in
`src/swasthtwin/features.py` (`FEATURE_LABELS`) and in [model_card.md](model_card.md). Labels:

| Column | Definition |
|---|---|
| `label` | 1 if any observed CGM in (t, t + 120 min] ≥ 180 mg/dL; NaN (row dropped) if < 50 % of future readings exist |
| `future_max_cgm_120` | max observed CGM in that window (analysis only) |
