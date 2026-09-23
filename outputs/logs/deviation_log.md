# Deviation Log

Per protocol.yml, any departure from the pre-registered design must be logged here
with date, rationale, whether outer-test results were already visible, and impact.

## 2026-09-13 — Scoping deviation from the full 16-week protocol

**Deviation:** The original research proposal specifies a 16-week protocol including
UN Comtrade trade flows, SEC XBRL corporate fundamentals, Census M3 orders/inventory,
and the Caldara-Iacoviello GPR index, evaluated jointly in a single fundamentals model.
This session executed the FRED macro/market block, Kenneth French factors, and GPR
index first (fast, reliable public sources), ran the full pre-registered primary
comparison and backtest on that reduced feature set, and is ingesting Comtrade/SEC
in parallel as background jobs to be merged in a subsequent revision.

**Rationale:** Comtrade (~7,900 API calls) and SEC XBRL (~500 firms x multiple filings)
are rate-limited, multi-hour ingestion jobs. Running the core analysis on the
faster-to-obtain sources first allows a genuine, complete primary-hypothesis test to
be reported now rather than leaving the entire study blocked on the slowest sources.

**Outer-test visibility:** No outer-test results were visible before this decision was
made — the decision to sequence data sources this way was made at Stage 2 (ingestion),
before any model was fit or any backtest was run.

**Impact:** The primary hypothesis test reported in this draft uses a macro/market
feature set only (12 fundamentals features drawn from FRED/French/GPR), not the full
trade-flow + corporate-fundamentals set specified in the original proposal. The
primary decision rule, target, and test were NOT changed based on this deviation and
were fixed in config/protocol.yml before the backtest was run. Once Comtrade/SEC
ingestion completes, the extended model will be reported as a separate, clearly
labeled analysis (M2-extended) rather than silently replacing the current primary
result.

## 2026-09-13 (later) — SEC XBRL coverage below pre-specified threshold

**Deviation:** All 8 SEC XBRL concepts (revenue, inventory, capex, R&D, cost of
revenue, operating income, receivables) fell below the pre-specified 70%
firm-quarter coverage threshold (actual range: 5.4%-16.2%) once ingestion completed
(505-firm universe, 251 fetched successfully, 97,609 fact-rows).

**Rationale:** This is an anticipated protocol stop-condition, not an ad hoc choice:
config/protocol.yml (via the original proposal's section 7.1) explicitly states that
if "SEC concept coverage is too heterogeneous," the response is to "publish the
corporate block as a robustness module." That rule was applied exactly as written.

**Outer-test visibility:** The primary h=3 backtest (Section 4.1) had already been
run and its results were known before the SEC coverage report was computed. However,
the decision of how to handle low SEC coverage was fixed in the protocol BEFORE the
SEC data was ever ingested, so this is a pre-committed contingency being executed,
not a post-hoc reaction to the primary result.

**Impact:** The corporate-fundamentals block (src/soxstudy/models/backtest_corporate_robustness.py)
is reported only as a secondary, removable robustness module (Table 9, manuscript
Section 4.4) and was never merged into the primary M2 model or the primary
hypothesis test. Its own result is also a null (OOS R^2 = -0.017, Clark-West
p = 0.67), consistent with, and not overturning, the primary finding.
