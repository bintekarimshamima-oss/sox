# UN Comtrade ingestion status

Run completed (script exit) at: 2026-09-13T15:05:17.719025+00:00

## Overall progress

- Total combos planned (reporters x hs_families x flows x months, 2006-01 through 2026-08): **7936**
- Total combos completed (across all runs, cumulative): **7936** (100.0%)
- Already done before this run: 0
- Attempted this run: 7936
- Succeeded this run: 7936
- Failed this run: 0

## Completed

All planned combos were attempted in this run (cumulative across all runs). See coverage report for per-cell completeness.

## Coverage by reporter / hs_family / flow

| reporter | hs_family | flow | attempted | retrieved | coverage | below 70% |
|---|---|---|---|---|---|---|
| USA | 8542 | M | 248 | 248 | 100.00% | no |
| USA | 8542 | X | 248 | 248 | 100.00% | no |
| USA | 8486 | M | 248 | 248 | 100.00% | no |
| USA | 8486 | X | 248 | 248 | 100.00% | no |
| KOR | 8542 | M | 248 | 248 | 100.00% | no |
| KOR | 8542 | X | 248 | 248 | 100.00% | no |
| KOR | 8486 | M | 248 | 248 | 100.00% | no |
| KOR | 8486 | X | 248 | 248 | 100.00% | no |
| TWN | 8542 | M | 248 | 248 | 100.00% | no |
| TWN | 8542 | X | 248 | 248 | 100.00% | no |
| TWN | 8486 | M | 248 | 248 | 100.00% | no |
| TWN | 8486 | X | 248 | 248 | 100.00% | no |
| JPN | 8542 | M | 248 | 248 | 100.00% | no |
| JPN | 8542 | X | 248 | 248 | 100.00% | no |
| JPN | 8486 | M | 248 | 248 | 100.00% | no |
| JPN | 8486 | X | 248 | 248 | 100.00% | no |
| DEU | 8542 | M | 248 | 248 | 100.00% | no |
| DEU | 8542 | X | 248 | 248 | 100.00% | no |
| DEU | 8486 | M | 248 | 248 | 100.00% | no |
| DEU | 8486 | X | 248 | 248 | 100.00% | no |
| NLD | 8542 | M | 248 | 248 | 100.00% | no |
| NLD | 8542 | X | 248 | 248 | 100.00% | no |
| NLD | 8486 | M | 248 | 248 | 100.00% | no |
| NLD | 8486 | X | 248 | 248 | 100.00% | no |
| CHN | 8542 | M | 248 | 248 | 100.00% | no |
| CHN | 8542 | X | 248 | 248 | 100.00% | no |
| CHN | 8486 | M | 248 | 248 | 100.00% | no |
| CHN | 8486 | X | 248 | 248 | 100.00% | no |
| SGP | 8542 | M | 248 | 248 | 100.00% | no |
| SGP | 8542 | X | 248 | 248 | 100.00% | no |
| SGP | 8486 | M | 248 | 248 | 100.00% | no |
| SGP | 8486 | X | 248 | 248 | 100.00% | no |

Cells below the protocol's ~70% rolling-coverage trust threshold: **0 / 32**. These are flagged, not dropped, per the task guardrails.

## File locations

- Raw snapshots: `data/raw/comtrade/<reporter>_<hsfamily>_<flow>_<period>.json`
- Call manifest: `data/manifests/comtrade_manifest.jsonl`
- Resume state: `data/manifests/comtrade_resume_state.json`
- Long-format table: `data/interim/comtrade_trade_flows.parquet`
- Coverage report: `data/interim/comtrade_coverage_report.csv`
