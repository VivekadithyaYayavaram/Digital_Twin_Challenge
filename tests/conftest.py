"""Shared fixtures: a SMALL synthetic cohort + trained model in a temp folder,
so tests are fast and never touch the committed demo artifacts."""
from __future__ import annotations

import copy

import pytest

from swasthtwin import config as C
from swasthtwin import pipeline as P


@pytest.fixture(scope="session")
def small_cfg():
    cfg = copy.deepcopy(C.load_config())
    cfg["data"]["n_patients"] = 20
    cfg["data"]["n_days"] = 5
    cfg["split"]["temporal_train_days"] = 3
    cfg["model"]["bootstrap_reps"] = 20
    return cfg


@pytest.fixture(scope="session")
def built(tmp_path_factory, small_cfg):
    root = tmp_path_factory.mktemp("swasthtwin")
    d, a, r = root / "data", root / "artifacts", root / "reports"
    cohort = P.step_data(small_cfg, d)
    meta = P.step_train(small_cfg, d, a)
    metrics = P.step_evaluate(small_cfg, d, a, r, make_figures=False)
    return {"root": root, "data": d, "artifacts": a, "reports": r, "cohort": cohort,
            "meta": meta, "metrics": metrics, "cfg": small_cfg}
