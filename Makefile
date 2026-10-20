PY ?= python
export PYTHONPATH := src

.PHONY: install data train evaluate pipeline test app submission check all clean

install:            ## install Python dependencies
	$(PY) -m pip install -r requirements.txt

data:               ## generate the synthetic cohort into data/
	$(PY) -m swasthtwin.cli data

train:              ## build features, compare models, save the pipeline
	$(PY) -m swasthtwin.cli train

evaluate:           ## held-out + temporal evaluation, figures, reports/
	$(PY) -m swasthtwin.cli evaluate

pipeline: data train evaluate   ## data -> train -> evaluate

test:               ## run the automated test suite
	$(PY) -m pytest -q

app:                ## launch the clinician dashboard
	$(PY) -m streamlit run app.py

submission:         ## rebuild architecture diagram + slides (PPTX + PDF)
	$(PY) scripts/build_architecture.py
	$(PY) scripts/build_presentation.py
	@echo "Tip: refresh dashboard screenshots first with scripts/capture_screenshots.py (needs Playwright)"

all: pipeline test submission

clean:              ## remove generated data, model and reports
	rm -rf data/*.csv data/*.csv.gz artifacts/* reports/*.json reports/*.md reports/figures/*

check:              ## audit links, files, metric consistency, placeholders
	$(PY) scripts/check_submission.py
