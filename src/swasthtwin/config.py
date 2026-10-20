"""Paths and YAML configuration loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

# File names (single source of truth for every module + the dashboard)
EHR_FILE = "ehr_patients.csv"
SENSOR_FILE = "sensor_stream.csv.gz"
SLEEP_FILE = "sleep_log.csv"
MEAL_LOG_FILE = "meal_log.csv"
MEAL_TRUTH_FILE = "meals_ground_truth.csv"
SPLIT_FILE = "patient_split.csv"
MODEL_FILE = "model_pipeline.joblib"
MODEL_META_FILE = "model_metadata.json"
METRICS_FILE = "metrics.json"


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load the modelling config (config/model_config.yaml by default)."""
    return load_yaml(path or CONFIG_DIR / "model_config.yaml")


def load_team_info() -> dict[str, Any]:
    p = CONFIG_DIR / "team_info.yaml"
    return load_yaml(p) if p.exists() else {}
