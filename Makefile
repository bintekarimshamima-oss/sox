.PHONY: setup replicate data validate features models robustness report all

PY := python
export PYTHONPATH := src

setup:
	$(PY) -c "import pandas, numpy, sklearn, statsmodels, xgboost, arch, pyarrow, httpx; print('environment OK')"

replicate:
	$(PY) src/soxstudy/ingest/replicate_thesis.py

data:
	$(PY) src/soxstudy/ingest/fred.py
	$(PY) src/soxstudy/ingest/french_gpr.py
	$(PY) src/soxstudy/ingest/comtrade.py
	$(PY) src/soxstudy/ingest/sec.py

features:
	$(PY) src/soxstudy/transform/panel_calendar.py
	$(PY) src/soxstudy/transform/features.py
	$(PY) src/soxstudy/transform/merge_extended_blocks.py

models:
	$(PY) src/soxstudy/models/backtest.py

robustness:
	$(PY) src/soxstudy/evaluation/descriptive.py
	$(PY) src/soxstudy/evaluation/metrics.py
	$(PY) src/soxstudy/evaluation/robustness.py

report:
	$(PY) src/soxstudy/reporting/figures.py
	$(PY) src/soxstudy/reporting/manuscript.py

all: setup replicate data features models robustness report
	@echo "Pipeline complete."
