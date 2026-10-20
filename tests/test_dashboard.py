"""Dashboard smoke tests against the committed demo data + model (headless)."""
from pathlib import Path

import pytest

from swasthtwin import config as C

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (C.ARTIFACT_DIR / C.MODEL_FILE).exists() or not (C.DATA_DIR / C.SENSOR_FILE).exists(),
    reason="demo data/model not generated (run `make pipeline`)")


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)


def test_dashboard_starts_without_exceptions():
    at = _app().run()
    assert not at.exception
    assert len(at.tabs) == 7


def test_dashboard_twin_controls_and_scenarios():
    at = _app().run()
    clock0 = [m.value for m in at.sidebar.markdown if m.value.startswith("**")][0]
    at.button[0].click().run()                       # +30 min
    assert not at.exception
    clock1 = [m.value for m in at.sidebar.markdown if m.value.startswith("**")][0]
    assert clock0 != clock1                          # clock refreshed
    for i in range(4):
        sb = [s for s in at.selectbox if s.label == "Scenario"][0]
        sb.select_index(i).run()
        assert not at.exception


def test_committed_metrics_match_report():
    import json
    m = json.loads((C.REPORT_DIR / C.METRICS_FILE).read_text())
    md = (C.REPORT_DIR / "metrics.md").read_text()
    best = m["selected_model"]
    assert f"{m['test'][best]['pr_auc']:.3f}" in md
