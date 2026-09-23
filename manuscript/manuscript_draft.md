# Do Physical Semiconductor Fundamentals Predict the Equity Cycle? A Release-Aware Data-Fusion Study of the PHLX Semiconductor Index

**Short title:** Semiconductor Fundamentals and the SOX Equity Cycle

**Authors:** [Author name(s) and affiliation(s)]

**Corresponding author:** [Name, email, ORCID]

**Keywords:** semiconductor cycle; equity return forecasting; pseudo-real-time evaluation; nested cross-validation; Clark–West test; elastic net; PHLX Semiconductor Index

**JEL classification:** G12 (Asset Pricing), G17 (Financial Forecasting and Simulation), E32 (Business Fluctuations; Cycles), L63 (Microelectronics)

---

## Abstract

**Purpose.** This study tests whether observable macro-financial, industrial, firm-level, and global trade-flow semiconductor-sector fundamentals contain incremental information about the sector equity cycle beyond information already embedded in past market returns, using the PHLX Semiconductor Index (SOX) as the outcome of interest.

**Design/methodology/approach.** We build a release-aware monthly panel (January 2006–August 2026, 248 months) combining U.S. semiconductor industrial production, capacity utilization, producer and border price indices, electronics inventory/order proxies, broad macro-financial controls (VIX, term spread, trade-weighted dollar), the Fama–French five-factor returns, the Caldara–Iacoviello geopolitical risk index, firm-level corporate fundamentals from SEC EDGAR XBRL filings for 251 semiconductor issuers, and global semiconductor trade flows (UN Comtrade HS 8542/8486, eight reporting economies, 181,231 records). Every predictor is lagged to its realistic public-availability date. A nested expanding-window pseudo-real-time backtest — 125 monthly out-of-sample origins from 2016 onward, with an outcome-horizon embargo and origin-specific refitting of every preprocessing and modeling step — evaluates fundamentals-augmented linear (elastic net) and nonlinear (gradient-boosted tree) models against market-only and historical-mean benchmarks for 1-, 3-, and 6-month SOX excess returns, an HAR-RV model against a historical-mean benchmark for next-month realized volatility, and a logistic-regression classifier against the unconditional base rate for a three-month downside-risk indicator.

**Findings.** The pre-registered primary comparison (three-month excess return; fundamentals-augmented elastic net versus the stronger of two market-only benchmarks; one-sided Clark–West test; block-bootstrap confidence interval) does not support the hypothesis that public fundamentals add forecast value: out-of-sample R² = −0.061, Clark–West one-sided p = 0.983, and a 95% block-bootstrap interval for the loss differential of [−0.19, 31.28], which includes zero. The null result is stable across secondary horizons (h = 1: R² = −0.008; h = 6: R² = −0.118), across four historically motivated regimes, when the fundamentals model is re-estimated with shallow gradient-boosted trees to rule out linear-model misspecification (R² = −0.026, p = 0.518), when a data-driven two-regime Markov-switching model is used to test whether the observed regime instability is itself exploitable (R² = −0.092, p = 0.411), when firm-level corporate fundamentals are added as a secondary robustness block (R² = −0.017, p = 0.674), and when semiconductor-specific global trade flows are added as a further robustness block (R² = −0.025, p = 0.257). Two partial exceptions are noted: the linear model turns marginally positive in the 2024–2026 AI-acceleration period (R² = +0.006, p = 0.201), and the nonlinear model turns marginally positive in the pre-COVID and COVID-shortage regimes (R² = +0.017 and +0.024 respectively); none reach conventional significance, and the data-driven regime-switching model does not convert this instability into forecast value. In contrast, the pre-registered secondary outcome of next-month realized volatility is substantially forecastable (HAR-RV out-of-sample R² = +0.312 versus the historical mean; QLIKE loss roughly halved), confirming that the evaluation methodology detects real forecasting relationships when they exist; a logistic-regression forecast of the three-month downside-risk indicator, by contrast, does not improve on the unconditional base rate.

**Originality/value.** Rather than another algorithm-comparison exercise, this paper contributes (i) an auditable, provenance-hashed, publicly reproducible semiconductor-cycle dataset spanning macro, market, firm-level, and global trade-flow sources; (ii) a pseudo-real-time evaluation protocol that enforces release-date discipline and reports a pre-registered primary test rather than a best-of-many result; and (iii) a transparent, protocol-anticipated null finding, robust across three independent fundamentals extensions, which we argue is itself informative about the efficiency with which public information is incorporated into semiconductor equity valuations.

**Practical implications.** The fully reproducible pipeline (`make all`) and the underlying dataset, including the semiconductor-specific global trade-flow block, provide an open benchmark against which future semiconductor-cycle forecasting claims — particularly those relying on more granular or proprietary data — can be measured under the same pseudo-real-time discipline.

**Keywords:** semiconductor cycle, equity return predictability, pseudo-real-time forecasting, nested cross-validation, Clark–West test, PHLX Semiconductor Index

---

## 1. Introduction

The PHLX Semiconductor Index (SOX) is widely treated in both industry commentary and academic work as a barometer of the global semiconductor cycle. This treatment elides an important distinction: SOX is an equity valuation measure — a forward-looking, discount-rate-sensitive aggregation of expectations about thirty-odd constituent firms — not a direct reading of physical chip demand, wafer production, or channel inventory. Whether *observable, publicly available* physical and financial fundamentals carry incremental predictive information about the semiconductor equity cycle, beyond what is already embedded in the index's own price history, is therefore a distinct and underexamined empirical question from the question of whether SOX prices are themselves serially predictable.

The literature on stock-return predictability from macroeconomic fundamentals is long-standing and cautionary. Campbell and Thompson (2008) show that naive out-of-sample tests of predictor variables against a historical-mean benchmark frequently fail once look-ahead bias and estimation error are properly accounted for, and that apparent in-sample predictability often does not survive a genuine out-of-sample test. Stock and Watson (2002) demonstrate that large macroeconomic panels can be compressed into a small number of predictive factors, but the incremental value of such factors for asset returns — as opposed to output growth — remains empirically contested. Sector-specific work on the semiconductor cycle (e.g., Aubry & Renou-Maissant, 2014) documents strong co-movement between semiconductor sales, capacity utilization, and pricing at the industry level, but does not test whether these physical-cycle indicators forecast *equity* returns once realistic publication lags and a nested, non-look-ahead validation design are imposed.

This study originates as a methodological redesign of an MBA thesis (Binte Karim Shamima) that compared several statistical and deep-learning architectures on a univariate SOX price series (2016–2026). We reproduce that exercise in full (Section 2.3) and confirm its central finding — that a naive previous-close benchmark is difficult for the tested architectures to beat out of sample — a common and well-documented result in financial forecasting (Campbell & Thompson, 2008). We treat that finding as a valid starting benchmark, not as the object of further optimization, and instead ask a different and, we argue, more consequential question: do *fundamentals* — as opposed to more elaborate functions of the price series itself — carry incremental forecast information about the semiconductor equity cycle, once the analysis is extended along three axes the original univariate design could not address: (i) a longer, multivariate information set spanning macro-financial, industrial, and firm-level fundamentals; (ii) a strict pseudo-real-time (release-aware) evaluation discipline that respects actual publication lags; and (iii) formal statistical tests for comparing nested forecasting models, including corrections for the fact that a fundamentals-augmented model mechanically contains the benchmark's information set.

We pre-registered a single primary hypothesis, model comparison, and statistical test before observing any out-of-sample result (`config/protocol.yml`), consistent with recommended practice for guarding against data-snooping in a setting with many plausible predictors and model choices (Hansen, Lunde, & Nason, 2011). The remainder of the paper is organized as follows. Section 2 describes the data and its provenance. Section 3 details the forecasting methodology and validation design. Section 4 reports results, including three independent robustness extensions to the core fundamentals set (nonlinear and regime-switching model specifications, firm-level corporate fundamentals, and global semiconductor trade flows). Section 5 discusses their interpretation. Section 6 states limitations. Section 7 concludes. Section 8 summarizes the three robustness extensions and their consistency with the primary finding.

## 2. Data

### 2.1 Study window and sources

The core monthly panel spans January 2006 to August 2026 (248 months), assembled entirely from public sources requiring no paid subscription:

| Data block | Source | Series | Frequency | Role |
|---|---|---|---|---|
| SOX equity index | FRED (Nasdaq/PHLX) | `NASDAQSOX` | Daily, aggregated to month-end | Outcome variable |
| Macro-financial controls | FRED | `VIXCLS`, `DGS3MO`, `DGS10`, `DTWEXBGS`, `NASDAQCOM` | Daily | Risk appetite, rates, term spread, USD, broad tech market |
| Industrial/supply conditions | FRED | `IPG3344S`, `CAPUTLG3344S` | Monthly | Production and capacity utilization, NAICS 3344 |
| Price pressure | FRED/BLS | `PCU33443344`, `IZ3344`, `IY3344` | Monthly | Producer and border price indices |
| Inventory/orders proxies | FRED/Census M3 | `A34SIS`, `A34HNO` | Monthly | Broader-electronics inventory/order signals |
| Market and risk factors | Kenneth French Data Library | Mkt-RF, SMB, HML, RMW, CMA, RF | Monthly | Excess-return and factor-adjusted controls |
| Geopolitical risk | Caldara & Iacoviello GPR dataset | GPR, GPR Threats, GPR Acts | Monthly | External risk and regime control |
| Corporate fundamentals | SEC EDGAR XBRL | Revenue, inventory, capex, R&D, cost of revenue, operating income, receivables | Quarterly filings | Secondary firm-level robustness block (Section 4.7) |
| Replication input | Original thesis dataset | `SOX_data.xlsx` (Date, SOX_Price) | Daily, 2016-01-04 to 2026-04-08 | Read-only replication baseline (Section 2.3) |

Every raw source file is stored as an immutable, timestamped snapshot with a SHA-256 hash, retrieval timestamp (UTC), and licence note recorded in `data/manifests/*.jsonl`. This provenance trail allows any reported number to be traced back to the exact raw file and retrieval time from which it was computed.

### 2.2 Release-aware timing discipline

Three distinct dates are tracked throughout the pipeline: the *reference month* an observation economically describes, the *available date* at which it could plausibly have entered a forecaster's real-time information set, and the *decision date* — the month-end forecast origin. Daily market series (SOX, VIX, rates, dollar, Nasdaq) enter the panel at month-end on a same-day basis. FRED/BLS/Census monthly series are assigned a conservative one-month publication lag: a value describing reference month *M* is not permitted to enter the model's information set until decision date *M*+1. We did not have access to true ALFRED real-time vintages for this pass, so this lag is a deliberately conservative simplification rather than an exact reconstruction of historical release calendars; for at least one source (Census M3), the true lag is somewhat longer (five to six weeks), so our design plausibly *understates* look-ahead risk for that series specifically. This is flagged explicitly as a sensitivity target in Section 6.

Kenneth French factor returns and the GPR index are likewise assigned a conservative one-month lag from their reference month. Firm-level SEC XBRL facts (Section 4.7) use the actual `filed` date recorded by EDGAR rather than the fiscal period-end date, which is the theoretically correct availability date for that source and requires no lag assumption.

### 2.3 Replication of the original thesis dataset

Before extending the analysis, we audited and reproduced the descriptive statistics and validation design of the supplied `SOX_data.xlsx` dataset directly, without modification (Table 1). The file contains 2,580 daily trading observations from 2016-01-04 to 2026-04-08, with no missing values, no duplicate dates, all prices strictly positive, and dates in sorted order. The price series has mean 2,813.91, standard deviation 1,796.45, and range 559.18–8,510.92. Daily log returns exhibit mean 0.099%, standard deviation 2.106%, and excess kurtosis 5.72 — fat tails and volatility clustering consistent with standard equity-return stylized facts (Cont, 2001).

Reproducing the thesis's fixed 80:20 chronological train/test split (training through 2024-03-15; test from 2024-03-18) confirms a structural concern raised in our audit: the test-period maximum price (8,510.92) is 64.8% above the training-period maximum (5,165.83), meaning any model whose functional form is bounded by the range of its training data is structurally unable to represent realized test-period levels. The mandatory naive (previous-close) benchmark — the classical hard-to-beat standard in financial forecasting (Campbell & Thompson, 2008) — achieves RMSE = 134.01 and MAE = 96.94 on the held-out test split.

**Table 1. Replication audit of the original thesis dataset.**

| Audit item | Observed evidence |
|---|---|
| Structure | One worksheet; two columns: Date and SOX_Price |
| Rows and period | 2,580 trading days; 2016-01-04 to 2026-04-08 |
| Integrity | 0 missing values; 0 duplicate dates; 0 non-positive prices; dates monotonically sorted |
| Price distribution | Mean 2,813.91; SD 1,796.45; range 559.18–8,510.92 |
| Daily log returns | Mean 0.099%; SD 2.106%; excess kurtosis 5.72 |
| Fixed 80:20 split | Train: 2,064 obs. through 2024-03-15; test: 516 obs. from 2024-03-18 |
| Range shift | Test-period maximum 64.8% above training-period maximum |
| Naive (previous-close) benchmark, test split | RMSE = 134.01; MAE = 96.94 |

*Source: direct audit of `SOX_data.xlsx`; full computation in `src/soxstudy/ingest/replicate_thesis.py`; output hash-traced in `outputs/tables/table0_thesis_replication_audit.csv`.*

## 3. Methodology

### 3.1 Outcome variables

The pre-registered primary outcome is the three-month SOX excess log return:

$$r^{excess}_{t,t+3} = 100 \times \left[\ln\left(\frac{P_{t+3}}{P_t}\right)\right] - \text{(compounded 3-month risk-free rate)}$$

Secondary outcomes are the analogous one- and six-month excess returns (horizon sensitivity), next-month realized volatility (annualized), and a three-month downside indicator equal to one when the realized excess return falls below the expanding-window 20th percentile. The primary target, model comparison, and statistical test were fixed in `config/protocol.yml` before any out-of-sample result was computed, following the pre-registration logic recommended for forecast-comparison studies with many researcher degrees of freedom (Hansen et al., 2011).

### 3.2 Predictor sets and models

Two nested feature sets are compared. The **market-only** set (12 features) comprises SOX momentum at 1, 3, and 12 months; one-month realized volatility; the Nasdaq Composite return; VIX level and one-month change; the 10-year–3-month Treasury term spread; the one-month change in the trade-weighted dollar; and three Fama–French factors (Mkt-RF, SMB, HML). The **fundamentals-augmented** set adds 12 macro/industrial features: industrial-production growth (year-over-year and month-over-month), capacity-utilization level and three-month change, producer/import/export price growth, a cross-series price-pressure dispersion measure, electronics inventory and new-order growth proxies, and the GPR level and three-month change. A secondary firm-level extension (Section 4.7) adds three corporate-fundamental factors (median inventory intensity, R&D intensity, and gross margin across reporting firms) drawn from SEC XBRL filings.

Five models are compared. **B0** is the historical mean of the training-window outcome — the classical hard-to-beat baseline in return forecasting (Campbell & Thompson, 2008). **B1** is an ordinary-least-squares regression on the market-only feature set. **M2**, the primary challenger, is an elastic net (Zou & Hastie, 2005) with regularization strength and L1 ratio selected by embargoed inner-fold cross-validation strictly within each outer training window; elastic-net regularization is appropriate here given the moderate multicollinearity observed among the fundamentals features (variance inflation factors up to 9.3 for producer-price growth; full diagnostics in the supplement, Table S3). **M3**, a secondary nonlinear robustness check, is a shallow gradient-boosted tree ensemble (Chen & Guestrin, 2016) on the same fundamentals feature set, with tree depth (1–3), learning rate, and the number of boosting rounds (via early stopping) selected by the same embargoed inner-fold scheme; M3 is included specifically to test whether a null result for M2 reflects an absence of information in the fundamentals features, or merely the inability of a linear/additive specification to extract nonlinear or interaction effects that a linear model cannot represent. **M5**, a second secondary robustness check (Section 4.6), is a two-regime Markov-switching regression (Hamilton, 1989) with a compact exogenous specification (VIX level and industrial-production growth), refit at every outer origin using only filtered regime probabilities; M5 tests whether the regime-instability pattern observed for M2 and M3 (Sections 4.4–4.5) can be converted into exploitable forecast value by a model that estimates regime breaks directly from the data rather than from calendar-defined windows.

### 3.3 Nested expanding-window pseudo-real-time validation

The outer validation loop begins with an initial training window of January 2006–December 2015 and produces one out-of-sample forecast at every month-end origin from 2016 onward — 125 origins for the primary three-month-horizon target after applying the horizon embargo (127 for h = 1; 122 for h = 6). At each origin, the final *h* months of the nominal training window are excluded from model fitting (the embargo), ensuring that no partially resolved *h*-month-ahead target ever crosses into the training data used to produce that origin's forecast. All preprocessing — feature scaling and elastic-net hyperparameter selection — is refit from scratch at every origin using only information dated at or before that origin's decision date; no observation from the outer test set at any origin is permitted to influence feature scaling, hyperparameter choice, or model fitting for that origin's forecast, a discipline essential for avoiding the look-ahead bias documented by Bergmeir, Hyndman, and Koo (2018) in naive cross-validation of autoregressive time series.

### 3.4 Statistical evaluation

Point-forecast accuracy is assessed via root-mean-squared error (RMSE), mean absolute error (MAE), and out-of-sample R² relative to the stronger of the two benchmarks (B0 or B1). Because the fundamentals-augmented model nests the benchmark's information set, the conventional Diebold and Mariano (1995) test is invalid under the null of equal predictive accuracy between nested models (Clark & West, 2007); we instead use the Clark–West MSPE-adjusted test, applied one-sided (alternative hypothesis: the fundamentals model forecasts better). As a non-parametric robustness check on the Clark–West result, we compute a stationary block bootstrap (block length 6 months, 2,000 resamples) 95% confidence interval for the mean loss differential, following the block-bootstrap logic for dependent time series used throughout the forecast-evaluation literature (Hansen et al., 2011).

The pre-registered primary decision rule requires all three of the following to hold jointly for the primary hypothesis to be considered supported: (i) positive out-of-sample R² at h = 3; (ii) one-sided Clark–West p < 0.05; and (iii) a block-bootstrap 95% confidence interval for the loss differential that excludes zero. Failure of this joint criterion is treated, by explicit pre-specification, as a valid and reportable null result rather than a failed study design.

## 4. Results

### 4.1 Primary comparison (h = 3 months)

**Table 2. Primary backtest performance, three-month SOX excess return (n = 125 out-of-sample origins, 2016–2026).**

| Model | RMSE | MAE |
|---|---|---|
| B0 — Historical mean | 13.74 | 10.52 |
| B1 — Market-only OLS | 14.29 | 10.80 |
| M2 — Elastic net + macro/industrial fundamentals | 14.15 | 10.78 |
| M3 — Gradient-boosted trees + macro/industrial fundamentals | 13.91 | 10.59 |

*Source: `outputs/tables/table5_primary_backtest_performance.csv`.*

The historical mean (B0) is the stronger of the two benchmarks at this horizon. The fundamentals-augmented elastic net (M2) does not improve upon it: out-of-sample R² = −0.061; Clark–West one-sided p = 0.983, far from the pre-registered α = 0.05 threshold; and the block-bootstrap 95% confidence interval for the loss differential is [−0.19, 31.28], which includes zero. All three pre-registered success conditions fail simultaneously. Consistent with the protocol's explicit decision rule (Section 3.4), this is recorded as a valid, informative null result rather than a failed study. Table 2 reports point-forecast accuracy only for the two models sharing the full fundamentals feature set (M2, M3); the compact Markov-switching specification (M5, Section 4.6) uses a different, smaller regressor set by design and is therefore reported separately (Table 6).

### 4.2 Secondary horizons

The null pattern is stable across both secondary horizons. At h = 1 month, out-of-sample R² = −0.008 (n = 127 origins); at h = 6 months, out-of-sample R² = −0.118 (n = 122 origins), with a block-bootstrap confidence interval that in this case excludes zero in the *unfavorable* direction (favoring the benchmark). Forecast difficulty for the fundamentals model increases, rather than decreases, with horizon in this feature set — the opposite of what an information-accumulation account of fundamentals-based predictability would predict — and is consistent with the fundamentals features failing to carry exploitable incremental signal at any horizon tested, at least via the linear/elastic-net specification examined here.

### 4.3 Risk outcomes: realized volatility and downside risk

The pre-registered outcome set (Section 3.1) includes two risk-focused targets in addition to excess returns: next-month realized volatility and a three-month downside-risk indicator. We evaluate both here using purpose-built models rather than the return-forecasting specifications above, since volatility and directional-risk forecasting have different statistical structures than mean-return forecasting.

**Realized volatility.** Next-month realized volatility is defined as the annualized realized volatility computed from actual daily SOX log returns within the following calendar month (not a monthly-return proxy), following standard realized-volatility construction (Andersen & Bollerslev, 1998). We forecast it with HAR-RV (Corsi, 2009), the standard parsimonious volatility model that regresses future realized volatility on lagged volatility components at three horizons (last month, 3-month average, 12-month average), refit at every outer origin using only past realized-volatility observations.

**Table 3. HAR-RV forecast performance, next-month realized volatility (n = 127 out-of-sample origins).**

| Model | RMSE | OOS R² vs. historical mean | QLIKE |
|---|---|---|---|
| Historical mean | 0.0454 | — | 0.623 |
| HAR-RV | 0.0377 | +0.312 | 0.324 |

*Source: `outputs/tables/table11_volatility_forecast_performance.csv`. QLIKE (Patton, 2011) computed on the variance scale; lower is better.*

Unlike the return-forecasting results above, realized volatility is substantially forecastable: HAR-RV achieves out-of-sample R² = +0.312 relative to the historical mean, and roughly halves the QLIKE loss (0.623 → 0.324). This is consistent with the well-documented volatility-clustering property of financial returns (Cont, 2001) and stands in useful contrast to the excess-return results: it demonstrates that the pipeline, validation discipline, and outcome-construction methodology used throughout this study are capable of detecting a real, economically established forecasting relationship when one exists, which strengthens the credibility of the null results reported for excess returns — the absence of a positive signal there is not an artifact of an unable methodology.

**Downside risk.** The three-month downside-risk indicator equals one when the realized 3-month excess return falls below the expanding-window 20th percentile. We forecast it with a logistic regression on the full market-plus-fundamentals feature set, refit at every outer origin with the same embargo as the primary return models.

**Table 4. Downside-risk classification performance, three-month horizon (n = 125 out-of-sample origins; base rate = 13.7%).**

| Model | ROC-AUC | PR-AUC | Brier score | Log loss |
|---|---|---|---|---|
| Base rate (unconditional) | 0.500 | — | 0.159 | 0.504 |
| Logistic regression (fundamentals) | 0.583 | 0.361 | 0.190 | 0.630 |

*Source: `outputs/tables/table12_downside_risk_classification.csv`.*

The fundamentals-based logistic classifier does not improve on the unconditional base rate: its ROC-AUC (0.583) is only marginally above chance (0.500), and it is *worse calibrated* than simply using the historical base rate — both its Brier score and log loss exceed the base-rate benchmark's. This indicates the model's probability estimates are overconfident (its predicted probabilities range from under 0.01% to over 99%, C.f. Table S7) without being correspondingly more accurate, a form of miscalibration rather than a lack of any signal (the ROC-AUC and PR-AUC are marginally above their no-skill values). This result is consistent with, and reinforces, the null found for excess returns: the same fundamentals feature set does not reliably distinguish elevated-downside-risk periods from normal ones at this horizon.

### 4.4 Regime stability

We partition the h = 3 backtest into four historically motivated regimes: pre-COVID (2016–2019), COVID/shortage (2020–2021), inventory correction (2022–2023), and AI acceleration (2024–2026). For the linear elastic net (M2), the null result is not an artifact of a single adverse sub-period dominating an otherwise-positive full-sample average: every regime shows out-of-sample R² ≤ 0 except the most recent AI-acceleration window, where the point estimate is marginally positive (R² = +0.006, Clark–West p = 0.201) — a small effect that does not reach conventional significance given the short sub-sample (n = 29 origins). The nonlinear specification (M3, introduced formally in Section 4.5) shows a partially complementary pattern, turning marginally positive precisely in the two *earlier* regimes where M2 is most negative (pre-COVID and COVID/shortage) while remaining negative in the two most recent regimes — a pattern discussed further in Section 4.5.

**Table 5. Regime-partitioned and robustness results, h = 3 excess return.**

| Regime | n | OOS R² (M2, elastic net) | CW p (M2) | OOS R² (M3, XGBoost) | CW p (M3) |
|---|---|---|---|---|---|
| Pre-COVID (2016–2019) | 48 | −0.027 | 0.9997 | +0.017 | 0.160 |
| COVID/shortage (2020–2021) | 24 | −0.089 | 0.8215 | +0.024 | 0.246 |
| Inventory correction (2022–2023) | 24 | −0.170 | 0.9783 | −0.051 | 0.711 |
| AI acceleration (2024–2026) | 29 | +0.006 | 0.2008 | −0.047 | 0.736 |
| Full sample | 125 | −0.061 | 0.9833 | −0.026 | 0.518 |
| Excl. top-5 abs.-return months | 120 | −0.068 | 0.9788 | −0.012 | 0.267 |

*Source: `outputs/tables/table8_regime_and_robustness_checks.csv`. CW = Clark–West, one-sided. Bootstrap confidence intervals for both models are reported in the online supplement (Table S6).*

Figure 3 (cumulative out-of-sample loss difference, fundamentals model minus benchmark) shows the mechanism underlying the aggregate result: a mild, gradual advantage for the fundamentals model accrues through roughly 2016–2021, followed by a sharp, discrete deterioration coinciding with the 2022 inventory-correction episode that erases the earlier accumulated gain and thereafter dominates the full-sample statistic. This pattern is more consistent with an unstable, regime-dependent relationship than with a uniformly absent one, and motivates the semiconductor-specific trade-flow extension reported in Section 4.8. Excluding the five largest absolute-excess-return months from the sample tightens the estimate slightly but does not reverse its sign (R² = −0.068).

### 4.5 Nonlinear robustness check: gradient-boosted trees

A null result for a linear elastic-net specification is consistent with two different underlying states of the world: either the fundamentals features genuinely carry no incremental forecast information, or they carry information that enters nonlinearly or through interactions that a linear/additive model cannot represent. To distinguish between these, we re-estimate the fundamentals-augmented model using shallow gradient-boosted trees (M3; Chen & Guestrin, 2016), with the same embargoed inner-fold hyperparameter selection and the same feature set as M2.

At the primary three-month horizon, M3 does not overturn the null: out-of-sample R² = −0.026 (an improvement over M2's −0.061, but still negative), Clark–West one-sided p = 0.518, and a bootstrap 95% CI of [−9.15, 17.55] that includes zero. The pattern is the same at both secondary horizons (h = 1: R² = −0.018; h = 6: R² = −0.092). Because a nonlinear model with the same information set fails to find exploitable signal either, the null result is better explained by an absence of usable incremental information in these specific fundamentals proxies than by a linear-model misspecification artifact.

The regime-partitioned re-estimation of M3, however, surfaces a pattern the linear model does not: M3 shows a small *positive* out-of-sample R² in both the pre-COVID (R² = +0.017, p = 0.160) and COVID-shortage (R² = +0.024, p = 0.246) regimes — regimes in which the elastic net was uniformly negative (Table 5) — while remaining negative, like the elastic net, in the inventory-correction and AI-acceleration regimes. Neither positive estimate reaches conventional significance given the short regime sub-samples (n = 48 and n = 24 respectively), and the corresponding block-bootstrap 95% confidence intervals are wide and include zero in both cases ([−17.54, 15.47] and [−35.24, 10.07] respectively; Table S6), so this pattern should be read as suggestive rather than conclusive. It is nonetheless consistent with the fundamentals features carrying a weak, nonlinearly-expressed signal that a tree-based model can partially recover in some regimes but that does not survive pooling across the full, structurally heterogeneous sample — reinforcing the regime-instability interpretation developed in Section 4.4 rather than contradicting it.

### 4.6 Regime-switching robustness check: data-driven breaks

The regime-partitioned analyses in Sections 4.4–4.5 rely on four calendar-defined windows chosen from prior knowledge of major semiconductor-cycle episodes (pandemic shortage, inventory correction, AI acceleration). This is informative but imposes the regime boundaries rather than letting the data locate them. As a formal test of the regime-instability interpretation, we estimate a two-regime Markov-switching regression (M5; a compact specification with VIX level and industrial-production growth as the only regressors, chosen to keep the maximum-likelihood estimation stable given the modest monthly sample size) at every outer origin, using only the filtered — never smoothed — regime probability to form each one-step-ahead forecast, consistent with the pseudo-real-time discipline used throughout.

The model converges to a well-identified two-regime solution (no regime capturing fewer than 5% of observations) at 124 of 125 outer origins; the one non-convergent origin falls back to the historical-mean prediction rather than being silently dropped.

**Table 6. Markov-switching (M5) robustness check, h = 3 excess return.**

| Model | n | Fits rejected | RMSE | OOS R² vs. B0 | Clark–West p (1-sided) | Bootstrap 95% CI |
|---|---|---|---|---|---|---|
| B0 — Historical mean | 125 | — | 13.74 | — | — | — |
| M5 — Markov-switching (VIX, IP growth) | 125 | 1 | 14.35 | −0.092 | 0.411 | [−12.69, 52.47] |

*Source: `outputs/tables/table10_markov_switching_robustness.csv`.*

Despite the earlier evidence of regime-dependent behavior in Sections 4.4–4.5, the data-driven regime-switching model does not recover exploitable forecast value: out-of-sample R² = −0.092, Clark–West one-sided p = 0.411, and a bootstrap 95% CI of [−12.69, 52.47] that includes zero. We read this as an important qualification rather than a contradiction of the earlier regime-instability finding: the calendar-defined splits in Table 5 show that the *relationship between fundamentals and returns* differs across known historical episodes, but this does not imply that a general-purpose, data-driven regime-detection model — which must also solve the harder problem of *identifying* the regime in real time from a compact, pre-specified feature set — can convert that historical pattern into a usable forecasting advantage. The two findings are complementary: fundamentals-return relationships appear regime-dependent, but regime-switching per se is not, by itself, a sufficient statistical technology for exploiting that dependence with the data and specification available here.

### 4.7 Secondary robustness module: firm-level corporate fundamentals

As a firm-level extension of the fundamentals information set, we identified 505 distinct SEC-registered semiconductor issuers via SIC code 3674 — deliberately not restricted to current SOX index constituents, in order to avoid survivorship bias (`data/manifests/sec_firm_universe_README.md`) — and retrieved company-level XBRL facts for 251 of them. The remaining 254 CIKs returned clean HTTP 404 responses, consistent with inactive, delisted, or shell filers with no machine-readable facts on record at SEC EDGAR; no data were fabricated or imputed for these firms. After restricting to 10-Q and 10-K filings and aligning every extracted fact to its actual EDGAR `filed` date rather than its fiscal-period end (eliminating look-ahead by construction, since the filing date is the true information-availability date), we obtained 97,609 filing-date-aligned fact-rows across eight target US-GAAP concepts: revenue, inventory, capital expenditure, R&D expense, cost of revenue, operating income, and accounts receivable.

Coverage of these concepts is low relative to the pre-registered 70% firm-quarter threshold specified in the study protocol: all eight concepts fall between 5.4% and 16.2% coverage of the full 505-firm × 82-quarter (2006Q1–2026Q2) universe (Table 7). This reflects genuine firm turnover rather than a data-quality defect — many identified filers were public, or subject to the XBRL mandate (in force only from 2011 onward), for only a fraction of the twenty-year sample window. Consistent with the protocol's pre-specified stop-condition rule for exactly this scenario — coverage below threshold triggers reporting the corporate block as a *removable secondary robustness module* rather than merging it into the primary model — we evaluate the resulting cross-sectional median inventory-intensity, R&D-intensity, and gross-margin factors separately from the primary hypothesis test.

**Table 7. SEC XBRL concept coverage (2006Q1–2026Q2, 505-firm universe).**

| Concept | Distinct firms with data | Firm-quarters with data | Coverage ratio |
|---|---|---|---|
| Revenue (ASC 606) | 89 | 2,240 | 5.4% |
| Sales revenue, net | 107 | 2,469 | 6.0% |
| Inventory, net | 180 | 5,557 | 13.4% |
| Capital expenditure | 181 | 5,958 | 14.4% |
| R&D expense | 178 | 6,023 | 14.5% |
| Cost of revenue | 103 | 2,862 | 6.9% |
| Operating income | 197 | 6,690 | 16.2% |
| Accounts receivable | 184 | 5,395 | 13.0% |

*Denominator: 505 firms × 82 quarters = 41,410 possible firm-quarters. Source: `data/interim/sec_concept_coverage_report.csv`.*

Adding this corporate block to the market-plus-macro feature set (model M2c) does not alter the qualitative conclusion: out-of-sample R² = −0.017 (versus −0.061 for the macro-only fundamentals model), Clark–West one-sided p = 0.674, block-bootstrap 95% CI [−5.13, 16.79], again including zero. The corporate block moves the point estimate closer to, but still below, zero relative to the macro-only specification — consistent with the regime analysis in Section 4.4 suggesting a weak and unstable relationship rather than one that is reversed in sign by the addition of firm-level information.

### 4.8 Secondary robustness module: semiconductor-specific trade flows

To test whether a globally representative, product-specific physical-demand signal sharpens the null finding above, we retrieved monthly UN Comtrade trade-flow data for HS 8542 (electronic integrated circuits, a demand-side proxy) and HS 8486 (semiconductor manufacturing equipment, a capacity-investment proxy) for eight major reporting economies (United States, South Korea, Taiwan, Japan, Germany, the Netherlands, China, and Singapore) spanning 2006–2026 — a grid of 7,936 individual API calls against the public UN Comtrade preview endpoint, retrieved at a respectful, rate-limited pace over several hours with no bulk-download shortcut. This yielded 181,231 reporter-partner-period trade-flow records, aggregated to a monthly year-over-year import-growth series for each HS family, with year-over-year coverage of 210 of 248 possible months (84.7%) for the demand proxy — materially better firm/period coverage than the SEC XBRL corporate block in Section 4.7, though still sparser than the core macro/market panel. Consistent with the release-aware discipline applied throughout, both series enter the feature set with a conservative two-month lag from the trade reference month (`config/protocol.yml`), reflecting UN Comtrade's typical reporting and revision lag.

Adding this trade-flow block to the market-plus-macro feature set (model M2d) does not overturn the primary null: out-of-sample R² = −0.025 (versus −0.061 for the macro-only fundamentals model and −0.017 for the SEC corporate block, Section 4.7), Clark–West one-sided p = 0.257, block-bootstrap 95% CI [−10.19, 22.77], again including zero. As with the corporate-fundamentals block, the point estimate moves closer to, but remains below, zero relative to the macro-only specification. The trade-flow series themselves are plotted in Figure 5; visually, both series show sharp swings around the 2008–2009 financial crisis, the 2020 COVID demand shock, and the 2022–2023 inventory correction, consistent with genuine physical-cycle variation — the null result therefore reflects an absence of *incremental linear forecasting value* for SOX excess returns conditional on the market and macro predictors already in the model, not an absence of cyclical variation in the underlying trade series.

### 4.9 Measurement and descriptive analysis

Feature coverage across the 21-variable macro/market panel averages 97.0%, with the weakest individual coverage on 12-month SOX momentum (91.1%), an artifact of the rolling-window construction at the start of the sample rather than missing source data. Bivariate correlations between candidate predictors and the forward three-month excess return (Table 8) show the VIX level as the single strongest linear correlate (r = 0.235), followed by capacity-utilization level (r = −0.212) and one-month realized volatility (r = 0.161) — all consistent with conventional risk-and-valuation channels rather than a distinctively "physical fundamentals" channel, and none exceeding a magnitude that would, on a univariate basis, imply strong incremental forecast value once market-based predictors are already included.

**Table 8. Ten largest-magnitude bivariate correlations with the forward three-month SOX excess return.**

| Feature | Correlation | n |
|---|---|---|
| VIX level | +0.235 | 237 |
| Capacity-utilization level | −0.212 | 237 |
| SOX realized volatility (1-month) | +0.161 | 225 |
| Producer-price growth (YoY) | +0.160 | 225 |
| Industrial-production growth (YoY) | −0.148 | 225 |
| Industrial-production growth (1-month) | −0.143 | 236 |
| Electronics new-order growth (YoY) | −0.138 | 225 |
| Capacity-utilization change (3-month) | −0.132 | 234 |
| Nasdaq return (1-month) | −0.115 | 236 |
| Export-price growth (YoY) | −0.104 | 224 |

*Source: `outputs/tables/table3_lead_lag_correlations.csv`.*

A CUSUM-based structural-break screen flags 16 of the 22 series examined (including the primary outcome itself) as exhibiting a parameter-stability break at the conventional 5% boundary somewhere within the 2006–2026 window — an expected finding given that the sample spans the 2008 financial crisis, the COVID-19 demand shock and subsequent chip shortage, and the 2023–2026 AI-driven equity re-rating, and consistent with the a priori expectation of regime-dependence underlying the regime-partitioned analysis in Section 4.4. Variance-inflation factors among the fundamentals predictors range from 1.47 to 9.27 (highest for producer-price growth), indicating moderate but not severe multicollinearity and supporting the choice of elastic-net regularization over unregularized OLS for the augmented specification. Full correlation matrices and variance-inflation diagnostics are reported in the online supplement (Tables S1–S3).

## 5. Discussion

The central empirical finding of this study is a carefully pre-registered null result: publicly observable U.S. macro-financial, industrial, and firm-level semiconductor-sector fundamentals, assembled at monthly frequency under a realistic release-date discipline, do not provide a statistically or economically reliable improvement over a naive historical-mean forecast of SOX excess returns — not at the pre-specified primary three-month horizon, not at one- or six-month secondary horizons, not when the linear specification is replaced with shallow gradient-boosted trees to rule out linear-model misspecification, not when a data-driven regime-switching model is used to test whether observed regime instability is itself exploitable, and not when a firm-level corporate-fundamentals block is added as a robustness extension. The consistency of the null across a linear, a nonlinear, and a regime-switching model specification is a materially stronger basis for the conclusion than any single model would provide alone: it rules out the most obvious alternative explanations for M2's failure — that a nonlinear/interaction effect (addressed by M3) or an unexploited regime structure (addressed by M5) is present but invisible to an additive linear model — and shifts the balance of evidence toward a genuine absence of exploitable incremental information in these particular fundamentals proxies, at least at the monthly frequency and specification choices examined here.

We interpret this finding through the decision framework specified in the pre-registered protocol. Two non-mutually-exclusive explanations are consistent with the evidence. First, semiconductor equity prices may incorporate public macro and industrial information rapidly, leaving little exploitable lag structure once realistic publication delays are respected — an efficient-markets account broadly consistent with Campbell and Thompson's (2008) finding that historical-mean benchmarks are difficult to beat out of sample across a wide range of return-predictor pairs. Second, the specific proxies available at monthly macro-aggregate frequency — U.S.-only industrial production and capacity utilization for the broad NAICS 3344 category, rather than semiconductor-specific and globally representative trade or production data — may simply be too coarse, too delayed, or too indirect to capture whatever physical-cycle information genuinely exists and eventually reaches the sector's equity valuations.

The regime-partitioned analysis (Sections 4.4–4.5) is informative in adjudicating between a flatly absent relationship and an unstable, regime-dependent one. The linear model's closest approach to a positive result occurs in the most recent AI-acceleration period, while the nonlinear model's closest approaches occur in the two earlier regimes (pre-COVID and COVID-shortage) — a divergence that itself suggests any genuine signal, if present, is both weak and unstable in functional form as well as in time, rather than a single well-defined relationship the two models simply estimate with different precision. The dominant contributor to the aggregate null for both models is a sharp, temporally localized deterioration coinciding with the 2022 inventory-correction episode — a pattern more consistent with a weak and time-varying relationship than with a complete absence of information content. This motivated the semiconductor-specific trade-flow extension reported in Section 4.8: if the physical-fundamentals channel exists but is obscured by the coarseness of U.S.-only NAICS-level proxies, a globally representative, product-specific trade-flow index (UN Comtrade HS 8542/8486) is a natural candidate for sharpening it. In the event, the trade-flow-augmented model (M2d) does not overturn the null either (OOS R² = −0.025, p = 0.257) — a finding that narrows, though does not eliminate, the space of remaining candidate explanations for why the physical-fundamentals channel is not detectable in this design: it is not simply an artifact of using U.S.-only rather than globally representative demand proxies, since the Comtrade-based measure is by construction global.

A natural methodological concern with any null-result-heavy study is whether the pipeline is simply incapable of detecting a real relationship, regardless of whether one exists. The realized-volatility result (Section 4.3) addresses this concern directly: the identical outer-loop backtesting infrastructure, applied to a different pre-registered outcome, recovers a strong and economically large forecasting relationship (HAR-RV out-of-sample R² = +0.312), consistent with the well-established volatility-clustering literature. This is not a coincidence of construction — HAR-RV and the return-forecasting models share the same embargo, refitting, and evaluation code, differing only in target variable and predictor set — and it substantially reduces the plausibility that the excess-return nulls reflect a methodological inability to detect signal rather than a genuine absence of one. The downside-risk classifier's null (Table 4), by contrast, is consistent with and reinforces the excess-return findings, since realized downside episodes are themselves a function of the same excess-return process examined throughout Section 4.

## 6. Limitations

- **Construct validity of the industrial-production proxies.** `IPG3344S` and `CAPUTLG3344S` describe NAICS 3344 (semiconductors *and* other electronic components collectively), not semiconductors in isolation, and are U.S.-only measures of a globally distributed production network in which Taiwan, South Korea, and other Asian manufacturers account for the majority of leading-edge output.
- **Publication-lag approximation.** A uniform one-month lag was applied to all FRED/BLS/Census series in this pass, rather than true ALFRED real-time vintages reflecting each series' actual historical release calendar. This is a deliberately conservative simplification that most likely understates true look-ahead risk for at least one series (Census M3, whose actual lag can run five to six weeks) and is flagged as a priority sensitivity check for the next revision.
- **Trade-flow block coverage.** The UN Comtrade (HS 8542/8486) trade-flow block (Section 4.8) achieves 84.7% monthly coverage for the demand proxy, better than the SEC corporate block but still short of full coverage of the core macro/market panel (97.0%); some of this gap reflects genuine reporting gaps among smaller reporting economies rather than a retrieval failure.
- **Sparse firm-level coverage.** The SEC XBRL corporate-fundamentals block (Section 4.7) covers only 5.4–16.2% of the full 505-firm × 82-quarter universe per concept, below the pre-specified 70% coverage threshold. It is reported only as a removable secondary robustness check, never merged into the primary hypothesis test, and its firm-quarter sparsity means the corporate factors should be read as suggestive rather than precisely estimated.
- **Limited statistical power.** With 125 outer-test origins for the primary horizon, power to detect small but economically meaningful forecasting gains is limited, and the reported confidence intervals are correspondingly wide; a genuinely small positive effect cannot be ruled out by this design.
- **Nonlinear and regime-switching checks are not exhaustive.** The gradient-boosted-tree (Section 4.5) and Markov-switching (Section 4.6) robustness checks rule out two broad classes of nonlinear and regime-dependent effects, but not all possible functional forms (e.g., a smooth-transition autoregressive specification, or a Markov-switching model with a richer exogenous set once sample size permits it).
- **Markov-switching specification is deliberately compact.** M5 uses only two regressors (VIX, industrial-production growth) to keep maximum-likelihood estimation numerically stable at each outer origin; a richer fundamentals set could not be estimated reliably at this sample size, so the M5 null speaks to whether a *parsimonious* regime-switching model recovers value, not to whether any conceivable regime-switching specification would.
- **Predictive, not causal, language throughout.** All reported results describe out-of-sample forecast association under a realistic information-timing constraint. None of the reported relationships should be read as identifying a causal effect of any fundamental variable on SOX returns; such a claim would require a separately identified, exogenous source of variation that this observational design does not provide.

## 7. Conclusion

Using a pre-registered, release-aware, nested expanding-window pseudo-real-time forecast evaluation of publicly available macro-financial, industrial, firm-level, and global trade-flow data, this study finds no reliable evidence that observable semiconductor-sector fundamentals improve out-of-sample forecasts of SOX excess returns beyond a simple historical-mean benchmark, across one-, three-, and six-month horizons, across three independently constructed fundamentals blocks (macro/industrial, firm-level corporate, and semiconductor-specific global trade flows), and across linear (elastic net), nonlinear (gradient-boosted tree), and data-driven regime-switching model specifications. The same evaluation infrastructure, applied to next-month realized volatility, recovers a strong and economically large forecasting relationship (HAR-RV out-of-sample R² = +0.312), confirming that the null results for excess returns and downside risk reflect a genuine absence of exploitable fundamentals signal rather than a methodological inability to detect one. We report the excess-return and downside-risk nulls as substantive, protocol-anticipated findings rather than a failure of study design. It indicates that, at the level of publicly available proxies examined here — including a globally representative, product-specific trade-flow measure explicitly constructed to address the coarseness of U.S.-only industrial proxies — the semiconductor equity cycle is efficiently priced with respect to the macro, firm-level, and trade-flow public information examined in this design. Either this reflects genuine efficiency with respect to publicly observable fundamentals, or the incremental signal, if any exists, requires a sharper proxy, a different frequency, or a different functional form than the ones examined here. This is, in our view, a more useful and durable contribution to the semiconductor-forecasting literature than an additional claim of algorithmic superiority on a univariate price series.

## 8. Summary of Robustness Extensions

All public-data blocks specified in the original study protocol are complete and incorporated into the results reported above; no extension remains outstanding at the time of this submission.

**Nonlinear and regime-switching robustness checks — complete and incorporated (Sections 4.5–4.6).** Beyond the primary linear specification, we estimated a shallow gradient-boosted-tree model (M3) and a compact two-regime Markov-switching regression (M5) on the same or a reduced fundamentals feature set, at every outer origin, using the same embargoed inner-fold and filtered-probability discipline as the primary model. Neither overturns the primary null (M3: OOS R² = −0.026, p = 0.518; M5: OOS R² = −0.092, p = 0.411), materially strengthening confidence that the null reflects a genuine absence of exploitable information rather than a linear-model or static-regime artifact.

**SEC EDGAR XBRL corporate fundamentals — complete and incorporated (Section 4.7).** 505 SIC-3674 filers were identified; 251 were successfully fetched (254 returned clean HTTP 404 responses, consistent with inactive or shell filers); 97,609 filing-date-aligned fact-rows were extracted across eight target concepts. Coverage fell below the pre-specified 70% firm-quarter threshold for every concept (5.4–16.2%), correctly triggering the protocol's own pre-specified stop-condition rule to report this block as a secondary, removable robustness module rather than merging it into the primary model. Its result — a null (OOS R² = −0.017, p = 0.674) — is consistent with, and does not overturn, the primary finding.

**UN Comtrade semiconductor trade flows — complete and incorporated (Section 4.8).** The full 7,936-call retrieval grid (8 reporting economies × 2 HS families × 2 trade flows × ~248 months) against the public UN Comtrade preview API was completed, yielding 181,231 reporter-partner-period trade-flow records for HS 8542 (integrated circuits, demand proxy) and HS 8486 (semiconductor manufacturing equipment, capacity-investment proxy). Merged into the feature panel with a conservative two-month release lag via `src/soxstudy/transform/merge_extended_blocks.py`, this block achieves 84.7% monthly coverage for the demand proxy — the best-covered of the two extended (non-core) blocks — yet its addition to the fundamentals model (M2d) likewise does not overturn the primary null (OOS R² = −0.025, p = 0.257). Across three independently constructed, non-mutually-exclusive extensions to the core market-plus-macro predictor set — firm-level corporate fundamentals, semiconductor-specific global trade flows, and (via M3/M5) nonlinear and regime-switching functional forms — none recovers a statistically or economically reliable forecasting improvement over the historical-mean benchmark, reinforcing the primary conclusion of this study as a genuine null rather than an artifact of any single modeling or data-coverage choice.

---

## Data availability statement

All non-restricted derived data, source code, configuration files (including the pre-registered protocol, `config/protocol.yml`), and provenance manifests supporting this study are available in the project repository (`sox-fundamentals-study/`). Raw source snapshots are retained under `data/raw/` with SHA-256 hashes and retrieval timestamps recorded in `data/manifests/`. SOX index values are sourced from FRED under the terms of the Federal Reserve Bank of St. Louis; redistribution of Nasdaq-derived index data beyond permitted reproducibility mechanisms is not undertaken. All other sources (UN Comtrade, SEC EDGAR, Kenneth French Data Library, Caldara–Iacoviello GPR dataset, U.S. Census Bureau, Bureau of Labor Statistics, Federal Reserve) are public-domain or academic-public datasets with no redistribution restriction on derived, non-bulk data.

## Funding statement

[To be completed by the author team — insert funding source(s) or state "This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors."]

## Conflict of interest statement

[To be completed by the author team — insert any relevant conflicts, or state "The authors declare no conflict of interest."]

## Generative-AI disclosure

Portions of the data-ingestion, transformation, modeling, and evaluation pipeline underlying this study were implemented with the assistance of an AI coding agent (Claude, Anthropic), operating under a pre-specified protocol and human review at each stage. All reported empirical results are generated by deterministic, version-controlled code operating on hashed, immutable source-data snapshots; no numerical result in this manuscript was generated or altered by an AI system without a corresponding, inspectable code artifact. [Author team: adjust this statement to match the target journal's specific AI-disclosure policy, which varies by publisher and was not independently verified at the time of drafting.]

## CRediT author contribution statement

[To be completed by the author team, following the target journal's CRediT taxonomy — e.g., Conceptualization, Methodology, Software, Validation, Formal analysis, Data curation, Writing – original draft, Writing – review & editing.]

## Ethics statement

This study uses only public aggregate, market, and corporate-filing data and does not involve human participants, personal data, or animal subjects. Institutional confirmation of ethics exemption should nonetheless be sought and recorded per the relevant institution's policy prior to submission.

---

## References

Andersen, T. G., & Bollerslev, T. (1998). Answering the skeptics: Yes, standard volatility models do provide accurate forecasts. *International Economic Review*, 39, 885–905. https://doi.org/10.2307/2527343

Aubry, M., & Renou-Maissant, P. (2014). Semiconductor industry cycles: Explanatory factors and forecasting. *Economic Modelling*, 39, 221–231. https://doi.org/10.1016/j.econmod.2014.02.039

Bergmeir, C., Hyndman, R. J., & Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. *Computational Statistics & Data Analysis*, 120, 70–83. https://doi.org/10.1016/j.csda.2017.11.003

Campbell, J. Y., & Thompson, S. B. (2008). Predicting excess stock returns out of sample: Can anything beat the historical average? *Review of Financial Studies*, 21, 1509–1531. https://doi.org/10.1093/rfs/hhm055

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). https://doi.org/10.1145/2939672.2939785

Clark, T. E., & West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics*, 138, 291–311. https://doi.org/10.1016/j.jeconom.2006.05.023

Cont, R. (2001). Empirical properties of asset returns: Stylized facts and statistical issues. *Quantitative Finance*, 1, 223–236. https://doi.org/10.1080/713665670

Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7, 174–196. https://doi.org/10.1093/jjfinec/nbp001

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13, 253–263. https://doi.org/10.1080/07350015.1995.10524599

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57, 357–384. https://doi.org/10.2307/1912559

Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The Model Confidence Set. *Econometrica*, 79, 453–497. https://doi.org/10.3982/ECTA5771

Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160, 246–256. https://doi.org/10.1016/j.jeconom.2010.03.034

Stock, J. H., & Watson, M. W. (2002). Forecasting using principal components from a large number of predictors. *Journal of the American Statistical Association*, 97, 1167–1179. https://doi.org/10.1198/016214502388618960

Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society, Series B*, 67, 301–320. https://doi.org/10.1111/j.1467-9868.2005.00503.x

---

## Appendix A. Figures

**Figure 1.** PHLX Semiconductor Index (SOX), month-end level, 2006–2026, with major regime bands (global financial crisis, COVID shock, chip shortage, inventory correction, AI acceleration) shaded. `outputs/figures/fig1_sox_series_with_regimes.png`

**Figure 2.** Industrial production growth, capacity utilization, and the Caldara–Iacoviello GPR index over time, with regime bands. `outputs/figures/fig2_macro_factor_panel.png`

**Figure 3.** Cumulative out-of-sample squared-error difference between the fundamentals-augmented model and the historical-mean benchmark, three-month excess-return target (values below zero favor the fundamentals model). `outputs/figures/fig6_cumulative_oos_loss_difference.png`

**Figure 4.** Rolling 36-month correlation between industrial-production growth and the forward three-month SOX excess return, with regime bands. `outputs/figures/fig4_rolling_correlation_stability.png`

**Figure 5.** Year-over-year growth in UN Comtrade HS 8542 (integrated circuit) and HS 8486 (semiconductor manufacturing equipment) import flows across eight reporting economies, 2008–2026, with regime bands. `outputs/figures/fig_comtrade_trade_flows.png`

## Appendix B. Supplementary tables (online supplement)

- **Table S1.** Full 21×21 feature correlation matrix (`outputs/tables/table2_feature_correlation_matrix.csv`).
- **Table S2.** Feature coverage ratios, full macro/market panel (`outputs/tables/table1_feature_coverage.csv`).
- **Table S3.** Variance-inflation factors, fundamentals feature set (`outputs/tables/table4_variance_inflation_factors.csv`).
- **Table S4.** CUSUM structural-break screen, all series (`outputs/tables/table_structural_break_screen.csv`).
- **Table S5.** SEC XBRL firm universe and inclusion/exclusion audit (`data/manifests/sec_firm_universe.csv`, `sec_firm_universe_README.md`).
- **Table S6.** Full regime-partitioned bootstrap confidence intervals for both M2 (elastic net) and M3 (XGBoost), including the pre-COVID and COVID-shortage intervals underlying the discussion in Section 4.5 (`outputs/tables/table8_regime_and_robustness_checks.csv`).
- **Table S7.** Full distribution of predicted downside-risk probabilities from the logistic classifier (Section 4.3), by origin (`outputs/predictions/backtest_downside_logit.csv`).
- **Table S8.** UN Comtrade trade-flow robustness module (M2d) predictions by origin, and monthly coverage of the demand and capacity-investment proxies (Section 4.8) (`outputs/predictions/backtest_trade_robustness.csv`, `outputs/tables/table13_trade_robustness_module.csv`).

## Appendix C. Reproducibility statement

The complete analysis is reproducible from raw source snapshots via `make all` (see `README.md` and `Makefile`). Computational environment: Python 3.12; key packages pandas, numpy, scikit-learn, statsmodels, xgboost, arch, pyarrow, httpx (exact versions pinned in the project's dependency lock file). All random elements (block-bootstrap resampling) use a fixed seed (42), recorded in code. No result in this manuscript was manually typed; all numeric values are generated by `src/soxstudy/reporting/manuscript.py` from hashed artifacts in `outputs/tables/`, `outputs/metrics/`, and `outputs/logs/`, with the complete machine-readable set and source-file hashes available in `manuscript/manuscript_numbers.json`. A complete deviation log, documenting every departure from the pre-registered protocol together with its date, rationale, and whether outer-test results were visible at the time of the decision, is maintained at `outputs/logs/deviation_log.md`.
