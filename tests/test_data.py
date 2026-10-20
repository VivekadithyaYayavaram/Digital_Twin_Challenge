"""Synthetic data: determinism, schema, units, timestamps, linkage, privacy."""
import copy
import re

import pandas as pd

from swasthtwin import data_generator as G


def test_generation_is_deterministic(small_cfg):
    a, b = G.generate_cohort(small_cfg), G.generate_cohort(small_cfg)
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])


def test_different_seed_changes_data(small_cfg):
    cfg2 = copy.deepcopy(small_cfg); cfg2["seed"] = small_cfg["seed"] + 1
    a, b = G.generate_cohort(small_cfg), G.generate_cohort(cfg2)
    assert not a["sensor"]["cgm_mg_dl"].equals(b["sensor"]["cgm_mg_dl"])


def test_validation_passes_and_detects_problems(built):
    data = G.generate_cohort(built["cfg"])
    assert G.validate_cohort(data) == []
    bad = {k: v.copy() for k, v in data.items()}
    bad["sensor"].loc[0, "cgm_mg_dl"] = 9999
    bad["sleep"].loc[0, "patient_id"] = "SYN-9999"
    probs = G.validate_cohort(bad)
    assert any("cgm_mg_dl" in p for p in probs) and any("sleep" in p for p in probs)


def test_schema_units_and_ranges(built):
    c = built["cohort"]
    assert list(c["sensor"].columns) == ["patient_id", "timestamp", "cgm_mg_dl", "heart_rate_bpm",
                                         "hrv_rmssd_ms", "steps"]
    s = c["sensor"]
    assert s.cgm_mg_dl.dropna().between(40, 400).all()
    assert s.heart_rate_bpm.dropna().between(30, 220).all()
    assert (s.steps.dropna() >= 0).all()
    assert c["ehr"].hba1c_pct.between(5.0, 13.0).all()
    assert set(c["ehr"].diabetes_status) <= {"type2_diabetes", "prediabetes"}
    # missing data is present on purpose (sensor gaps, watch off)
    assert s.cgm_mg_dl.isna().any() and s.heart_rate_bpm.isna().any()


def test_timestamps_regular_and_linked(built):
    c = built["cohort"]
    ids = set(c["ehr"].patient_id)
    assert c["ehr"].patient_id.is_unique
    assert set(c["sensor"].patient_id) == ids
    assert set(c["sleep"].patient_id) <= ids and set(c["meal_log"].patient_id) <= ids
    for _, g in c["sensor"].groupby("patient_id"):
        assert (g.timestamp.diff().dropna() == pd.Timedelta(minutes=5)).all()
    assert (c["sleep"].sleep_end > c["sleep"].sleep_start).all()


def test_no_personal_identifiers(built):
    c = built["cohort"]
    assert all(re.fullmatch(r"SYN-\d{4}", p) for p in c["ehr"].patient_id)
    banned = {"name", "phone", "email", "address", "aadhaar", "mrn"}
    for df in c.values():
        assert not banned & {col.lower() for col in df.columns}


def test_saved_files_roundtrip(built):
    loaded = G.load_cohort(built["data"])
    assert len(loaded["sensor"]) == len(built["cohort"]["sensor"])
    assert pd.api.types.is_datetime64_any_dtype(loaded["sensor"].timestamp)
