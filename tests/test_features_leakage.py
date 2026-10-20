"""Features use only the past; labels use only the future; splits are by patient."""
import numpy as np
import pandas as pd
import pytest

from swasthtwin import features as F
from swasthtwin import modeling as M


def _patient(c, i=0):
    ehr = c["ehr"].iloc[i]
    pid = ehr.patient_id
    return (ehr, c["sensor"][c["sensor"].patient_id == pid], c["sleep"][c["sleep"].patient_id == pid],
            c["meal_log"][c["meal_log"].patient_id == pid])


def test_features_identical_when_future_is_removed(built):
    """Strongest leakage test: computing features on the full stream vs on the
    stream truncated at t must give the same row for t."""
    ehr, s, sl, ml = _patient(built["cohort"], 1)
    times = pd.to_datetime(["2026-03-03 08:00", "2026-03-04 13:30", "2026-03-05 21:00"])
    full = F.build_patient_features(s, sl, ml, ehr, sample_times=times)
    for t in times:
        trunc = F.build_patient_features(s[s.timestamp <= t], sl[sl.sleep_end <= t], ml[ml.timestamp <= t],
                                         ehr, sample_times=[t])
        a = full[full.timestamp == t][F.ALL_FEATURES].reset_index(drop=True)
        b = trunc[F.ALL_FEATURES].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_future_changes_do_not_affect_features(built):
    ehr, s, sl, ml = _patient(built["cohort"], 2)
    t = pd.Timestamp("2026-03-04 10:00")
    s2 = s.copy()
    s2.loc[s2.timestamp > t, ["cgm_mg_dl", "heart_rate_bpm", "steps"]] = [399, 200, 999]
    ml2 = pd.concat([ml, pd.DataFrame([{"patient_id": ehr.patient_id, "timestamp": t + pd.Timedelta(minutes=3),
                                        "meal_type": "x", "carbs_g_estimated": 150}])])
    a = F.build_patient_features(s, sl, ml, ehr, sample_times=[t])[F.ALL_FEATURES]
    b = F.build_patient_features(s2, sl, ml2, ehr, sample_times=[t])[F.ALL_FEATURES]
    pd.testing.assert_frame_equal(a, b)


def test_meal_logged_after_t_not_visible(built):
    ehr, s, sl, _ = _patient(built["cohort"], 0)
    t = pd.Timestamp("2026-03-03 10:00")
    ml = pd.DataFrame([{"patient_id": ehr.patient_id, "timestamp": t + pd.Timedelta(minutes=3),
                        "meal_type": "x", "carbs_g_estimated": 60}])
    at_t = F.build_patient_features(s, sl, ml, ehr, sample_times=[t]).iloc[0]
    later = F.build_patient_features(s, sl, ml, ehr, sample_times=[t + pd.Timedelta(minutes=5)]).iloc[0]
    assert at_t.carbs_logged_120 == 0 and later.carbs_logged_120 == 60


def test_label_matches_future_window_only(built):
    ehr, s, *_ = _patient(built["cohort"], 3)
    lab = F.make_labels(s, 180, 120, 0.5).set_index("timestamp")
    g = s.set_index("timestamp").cgm_mg_dl
    for t in g.index[100:3000:97]:
        fut = g[(g.index > t) & (g.index <= t + pd.Timedelta(minutes=120))]
        if fut.notna().mean() >= 0.5:
            assert lab.loc[t, "label"] == float(fut.max() >= 180)
            assert lab.loc[t, "future_max_cgm_120"] == fut.max()  # excludes current reading


def test_label_columns_not_model_inputs():
    forbidden = {"label", "future_max_cgm_120", "future_coverage", "study_day", "split", "patient_id",
                 "timestamp"}
    assert not forbidden & set(F.ALL_FEATURES)


def test_dataset_has_both_classes(built):
    df = F.build_dataset(built["cohort"], built["cfg"])
    assert set(df.label.unique()) == {0, 1}
    assert df.timestamp.dt.minute.isin([0, 30]).all()


def test_patient_level_split_is_disjoint(built):
    sp = M.split_patients(built["cohort"]["ehr"], built["cfg"])
    assert sp.patient_id.is_unique
    groups = {k: set(v) for k, v in sp.groupby("split").patient_id}
    assert set(groups) == {"train", "val", "test"}
    assert not (groups["train"] & groups["test"]) and not (groups["train"] & groups["val"])
    assert not (groups["val"] & groups["test"])
    saved = pd.read_csv(built["artifacts"] / "patient_split.csv")
    pd.testing.assert_frame_equal(saved, sp)


def test_missing_values_and_bad_inputs_handled(built):
    ehr, s, sl, ml = _patient(built["cohort"], 0)
    s2 = s.copy()
    s2["cgm_mg_dl"] = s2["cgm_mg_dl"].astype(object)
    s2.iloc[50:80, s2.columns.get_loc("cgm_mg_dl")] = "error"   # corrupt strings
    s2.iloc[90, s2.columns.get_loc("cgm_mg_dl")] = 5000          # impossible value
    out = F.build_patient_features(s2, None, None, ehr)
    assert len(out) > 0 and np.isfinite(out.cgm_current.dropna()).all()
    assert out.cgm_current.max() <= 400
    with pytest.raises(ValueError):
        F.build_patient_features(s.iloc[0:0], sl, ml, ehr)
