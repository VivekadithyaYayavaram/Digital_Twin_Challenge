# Architecture

Diagram: [`submission/architecture_diagram.pdf`](../submission/architecture_diagram.pdf) ·
editable [`submission/architecture_diagram.pptx`](../submission/architecture_diagram.pptx) ·
source script [`scripts/build_architecture.py`](../scripts/build_architecture.py).

## Modules (`src/swasthtwin/`)

| Module | Responsibility |
|---|---|
| `config.py` | paths, file names, YAML loading |
| `data_generator.py` | deterministic synthetic EHR + sensor + sleep + meal-log cohort, validation, save/load |
| `features.py` | regularise to 5-min grid, trailing-window features, fusion features, future-window labels |
| `modeling.py` | patient-level split, preprocessing, candidate models, rule baseline |
| `evaluation.py` | metrics, threshold choice, patient bootstrap, report figures |
| `pipeline.py` | `data → train → evaluate` steps, metadata, `reports/metrics.{json,md}` |
| `explain.py` | single-feature and whole-stream perturbation explanations |
| `twin.py` | `PatientTwin`: state, replay `advance`, `ingest_observation`, `log_meal`, `scenario`, history, warnings |
| `charts.py` | Plotly figures for the dashboard |
| `cli.py` | `python -m swasthtwin.cli {data,train,evaluate,all}` |

`app.py` is the Streamlit dashboard; it imports the twin and loads saved artifacts only.

## Training-time vs runtime

* **Offline:** `make data` → `make train` (fit on train patients, select on validation patients, save
  pipeline + metadata) → `make evaluate` (test patients, temporal hold-out, ablation, bootstrap, figures).
* **Runtime:** `PatientTwin` holds one patient's EHR and the observations received up to its clock,
  calls `features.build_patient_features` (the same function used for training) at the clock time, and scores
  with the loaded pipeline. No retraining happens at runtime.

## Integration status

No hospital EHR, FHIR server, CGM vendor API or wearable is connected. The sensor "feed" is a deterministic
replay of the generated stream, plus manual observations entered in the dashboard. A real integration would
replace `PatientTwin._feed` with a device/EHR adapter that writes into the same schema.
