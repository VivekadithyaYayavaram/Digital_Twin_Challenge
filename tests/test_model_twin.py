"""Training, persistence, inference, evaluation reproducibility and the Digital Twin."""
import json

import numpy as np
import pandas as pd
import pytest

from swasthtwin import config as C
from swasthtwin import features as F
from swasthtwin import pipeline as P
from swasthtwin import twin as T


def test_model_saved_and_loads(built):
    pipe, meta = T.load_model(built["artifacts"])
    assert meta["selected_model"] in built["cfg"]["model"]["candidates"]
    assert 0 < meta["operating_threshold"] < 1
    assert meta["data"]["synthetic"] is True
    assert set(meta["features"]["numeric"]) == set(F.NUMERIC_FEATURES)


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="make train"):
        T.load_model(tmp_path)


def test_predictions_valid_probabilities(built):
    pipe, _ = T.load_model(built["artifacts"])
    df = F.build_dataset(built["cohort"], built["cfg"]).head(300)
    p = pipe.predict_proba(df[F.ALL_FEATURES])
    assert p.shape == (300, 2)
    assert np.all((p >= 0) & (p <= 1)) and np.allclose(p.sum(1), 1)
    df2 = df.copy()
    df2[F.SENSOR_FEATURES] = np.nan        # all sensors missing -> still a valid probability
    q = pipe.predict_proba(df2[F.ALL_FEATURES])[:, 1]
    assert np.all((q >= 0) & (q <= 1))


def test_metrics_derived_and_selection_on_validation(built):
    m = built["metrics"]
    best = m["selected_model"]
    sel = json.loads((built["artifacts"] / "model_selection.json").read_text())
    assert best == max(sel["candidates"], key=lambda k: sel["candidates"][k]["pr_auc"])
    t = m["test"][best]
    c = t["confusion"]
    assert c["tp"] + c["fp"] + c["tn"] + c["fn"] == t["n"]
    assert abs(t["sensitivity"] - c["tp"] / (c["tp"] + c["fn"])) < 1e-3
    assert "SYNTHETIC" in m["disclaimer"]


def test_evaluation_is_reproducible(built, tmp_path):
    m2 = P.step_evaluate(built["cfg"], built["data"], built["artifacts"], tmp_path, make_figures=False)
    a = json.loads((built["reports"] / C.METRICS_FILE).read_text())
    b = json.loads((tmp_path / C.METRICS_FILE).read_text())
    assert a == b
    assert m2["selected_model"] == built["metrics"]["selected_model"]


@pytest.fixture()
def twin(built):
    pipe, meta = T.load_model(built["artifacts"])
    pid = built["cohort"]["ehr"].patient_id.iloc[0]
    tw = T.PatientTwin(pid, G_load(built), pipe, meta, built["cfg"])
    tw.refresh()
    return tw


def G_load(built):
    from swasthtwin import data_generator as G
    return G.load_cohort(built["data"])


def test_twin_state_updates_with_new_observations(twin):
    t0, n0 = twin.cursor, len(twin.history)
    fc = twin.advance(30)
    assert twin.cursor == t0 + pd.Timedelta(minutes=30) and len(twin.history) == n0 + 1
    assert 0 <= fc.probability <= 1 and fc.as_of == twin.cursor
    assert twin.observed_sensor().timestamp.max() <= twin.cursor  # never sees the future


def test_twin_prediction_equals_model_on_same_inputs(twin):
    f = twin.compute_features()
    direct = float(twin.pipe.predict_proba(f[F.ALL_FEATURES])[0, 1])
    assert abs(twin.refresh().probability - direct) < 1e-12


def test_twin_manual_observation_changes_inputs(twin):
    fc = twin.ingest_observation(cgm_mg_dl=350)
    assert fc.latest_cgm == 350 and fc.source == "manual"
    with pytest.raises(ValueError):
        twin.ingest_observation(cgm_mg_dl=900)
    with pytest.raises(ValueError):
        twin.ingest_observation(blood_pressure=120)


def test_twin_scenarios_use_model_and_do_not_mutate_state(twin):
    before_cursor, before_hist = twin.cursor, len(twin.history)
    for kind in T.SCENARIOS:
        r = twin.scenario(kind)
        assert 0 <= r["scenario"].probability <= 1
        assert abs(r["delta"] - (r["scenario"].probability - r["baseline"].probability)) < 1e-12
        assert len(r["changed_features"]) > 0
    assert twin.cursor == before_cursor and len(twin.history) == before_hist
    with pytest.raises(ValueError):
        twin.scenario("teleport")


def test_twin_stale_sensor_warning(built):
    pipe, meta = T.load_model(built["artifacts"])
    c = G_load(built)
    pid = c["ehr"].patient_id.iloc[0]
    t = pd.Timestamp("2026-03-04 12:00")
    s = c["sensor"]
    c["sensor"] = s.assign(cgm_mg_dl=s.cgm_mg_dl.where(
        ~((s.patient_id == pid) & (s.timestamp > t - pd.Timedelta(hours=2)) & (s.timestamp <= t))))
    tw = T.PatientTwin(pid, c, pipe, meta, built["cfg"], start_time=t)
    fc = tw.refresh()
    assert any("stale" in w or "CGM readings" in w for w in fc.warnings)


def test_explanations(twin):
    expl, streams = twin.explain()
    assert set(expl.feature) == set(F.ALL_FEATURES)
    assert set(streams["streams"]) == {"EHR (historical)", "Sensor (dynamic)", "Context (time of day)"}


def test_unknown_patient(built):
    pipe, meta = T.load_model(built["artifacts"])
    with pytest.raises(KeyError):
        T.PatientTwin("SYN-9999", G_load(built), pipe, meta, built["cfg"])
