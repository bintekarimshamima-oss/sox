# SOX Fundamentals Study

Release-aware, multivariate redesign of a univariate SOX (PHLX Semiconductor Index)
forecasting thesis. See `manuscript/manuscript_draft.md` for the current manuscript
and `config/protocol.yml` for the pre-registered primary hypothesis/test.

## Reproduction

```
python src/soxstudy/ingest/replicate_thesis.py      # Stage 1: reproduce thesis audit
python src/soxstudy/ingest/fred.py                  # Stage 2: FRED macro/market series
python src/soxstudy/ingest/french_gpr.py            # Stage 2: Fama-French factors + GPR
python src/soxstudy/ingest/comtrade.py              # Stage 2: UN Comtrade trade flows (slow, ~3h)
python src/soxstudy/ingest/sec.py                   # Stage 2: SEC XBRL corporate facts (slow)
python src/soxstudy/transform/panel_calendar.py     # Stage 3: harmonize core monthly panel
python src/soxstudy/transform/features.py           # Stage 4: stationary features + outcomes
python src/soxstudy/transform/merge_extended_blocks.py  # Stage 4b: merge Comtrade/SEC once ready
python src/soxstudy/models/backtest.py              # Stage 6: nested expanding-window backtest
python src/soxstudy/evaluation/metrics.py           # Stage 7: primary test + metrics
python src/soxstudy/evaluation/descriptive.py       # Stage 5: measurement/descriptive analysis
python src/soxstudy/evaluation/robustness.py        # Stage 7: regime + robustness checks
python src/soxstudy/reporting/figures.py            # Stage 8: figures
python src/soxstudy/reporting/manuscript.py         # Stage 8: manuscript numbers (hashed)
```

Run from the repository root with `PYTHONPATH=src` set (or `pip install -e .`).

## Status

Core FRED/French/GPR-based analysis is complete (see `manuscript/manuscript_draft.md`).
UN Comtrade and SEC XBRL extension blocks are ingested via `config/sources.yml` and
merged via `merge_extended_blocks.py` once their raw ingestion completes — both are
long-running, rate-limited jobs; see `outputs/logs/comtrade_ingest_status.md` and
`outputs/logs/sec_ingest_status.md` for live progress and resumption instructions.

## Guardrails

- Raw snapshots in `data/raw/` are immutable; never edit by hand.
- All feature/model fitting uses only information available at each forecast origin
  (see `config/protocol.yml` for the no-future-information rules).
- The primary comparison (h=3 excess return, elastic net vs. market-only benchmark,
  Clark-West test) is fixed and was not changed after seeing results.
