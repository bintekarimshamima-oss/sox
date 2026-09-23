"""Secondary robustness model: a two-regime Markov-switching regression (M5)
that lets the data estimate regime breaks directly, rather than imposing the
four calendar-defined windows used in the regime-partitioned analysis
(Section 4.3). This formally tests the regime-instability interpretation
motivated by the divergence between the linear (M2) and nonlinear (M3)
robustness results, instead of only illustrating it with ad hoc splits.

Per protocol.yml / Appendix B of the study protocol: 2 regimes primary,
filtered (not smoothed) probabilities only in pseudo-real-time use, reject
degenerate regimes (one regime capturing <5% of observations), and keep the
exogenous specification compact given the modest monthly sample size.

The exogenous set here is deliberately small (VIX level as a market-risk
proxy; industrial-production YoY growth as a compact fundamentals proxy) to
keep the EM/MLE estimation stable at each of ~125 outer-loop refits -- a
full 24-feature specification does not converge reliably at this sample
size, consistent with the protocol's own warning against overparameterized
regime models.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from soxstudy.models.backtest import historical_mean_benchmark, embargoed_inner_folds

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"
OUT_PRED = ROOT / "outputs" / "predictions"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_LOGS = ROOT / "outputs" / "logs"

MS_FEATURES = ["vix_level", "ip_growth_yoy"]
TARGET = "sox_excess_ret_h3"
HORIZON = 3
MIN_REGIME_SHARE = 0.05  # protocol: reject degenerate regimes


def fit_markov_switching(X_train: np.ndarray, y_train: np.ndarray):
    """Fit a 2-regime switching regression; return None on non-convergence
    or a degenerate regime split rather than silently accepting a bad fit."""
    if len(y_train) < 40:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MarkovRegression(y_train, k_regimes=2, exog=X_train, switching_variance=True)
            res = model.fit(maxiter=500, disp=False)
    except Exception:
        return None

    if not res.mle_retvals.get("converged", False):
        return None

    filtered = np.asarray(res.filtered_marginal_probabilities)
    regime_share = filtered.mean(axis=0)
    if regime_share.min() < MIN_REGIME_SHARE:
        return None  # degenerate regime, reject per protocol

    return res


def predict_next_period(res, x_next: np.ndarray) -> float:
    """One-step-ahead forecast: regime-probability-weighted combination of the
    two regimes' conditional means, using the FILTERED (not smoothed) final-
    period regime probability as the real-time state estimate, propagated one
    step forward via the estimated transition matrix. `regime_transition[i, j]`
    is P(state i at t+1 | state j at t), so next_prob = transition @ filtered."""
    k_exog = len(MS_FEATURES)
    n_regimes = 2
    param_names = res.model.param_names
    param_map = dict(zip(param_names, res.params))

    trans = np.asarray(res.regime_transition)
    trans = trans[:, :, 0] if trans.ndim == 3 else trans

    filtered_last = np.asarray(res.filtered_marginal_probabilities)[-1]
    next_regime_prob = trans @ filtered_last

    means = []
    for regime in range(n_regimes):
        c = param_map.get(f"const[{regime}]")
        betas = [param_map.get(f"x{j + 1}[{regime}]") for j in range(k_exog)]
        if c is None or any(b is None for b in betas):
            return float("nan")
        means.append(c + np.dot(betas, x_next))

    return float(np.dot(next_regime_prob, means))


def run_backtest(df: pd.DataFrame, initial_train_end="2015-12-31"):
    df = df.sort_values("decision_date").reset_index(drop=True)
    origins = df.index[df["decision_date"] > pd.Timestamp(initial_train_end)]

    records = []
    n_rejected = 0
    for origin_idx in origins:
        origin_date = df.loc[origin_idx, "decision_date"]
        y_true = df.loc[origin_idx, TARGET]
        if pd.isna(y_true):
            continue

        train_end_idx = origin_idx - HORIZON
        if train_end_idx < 40:
            continue
        train_df = df.iloc[:train_end_idx]

        valid = train_df[MS_FEATURES + [TARGET]].notna().all(axis=1)
        train_valid = train_df.loc[valid]
        if len(train_valid) < 40:
            continue

        y_train = train_valid[TARGET].values
        X_train = train_valid[MS_FEATURES].values
        hist_pred = historical_mean_benchmark(pd.Series(y_train))

        x_row = df.loc[[origin_idx], MS_FEATURES]
        ms_pred = hist_pred
        if x_row.notna().all(axis=1).iloc[0]:
            res = fit_markov_switching(X_train, y_train)
            if res is None:
                n_rejected += 1
            else:
                pred = predict_next_period(res, x_row.values[0])
                if not np.isnan(pred):
                    ms_pred = pred

        records.append({
            "decision_date": origin_date,
            "y_true": float(y_true),
            "pred_B0_historical_mean": hist_pred,
            "pred_M5_markov_switching": ms_pred,
            "n_train_obs": int(len(train_valid)),
        })

    return pd.DataFrame.from_records(records), n_rejected


def main():
    df = pd.read_parquet(PROCESSED / "features_monthly_core.parquet")
    print("Running Markov-switching (M5) backtest for h=3...")
    result, n_rejected = run_backtest(df)
    print(f"  -> {len(result)} origins produced; {n_rejected} rejected (non-convergence or degenerate regime)")

    # Use the primary pipeline's B0 benchmark predictions (fit on the full 24-feature
    # NA-filtering, matching Table 2/3) rather than this module's own B0, which is
    # computed over a slightly different training subset due to the MS model's
    # smaller (2-feature) NA-filter -- ensures an apples-to-apples RMSE/R^2 comparison.
    main_preds = pd.read_parquet(OUT_PRED / "backtest_predictions.parquet")
    main_b0 = main_preds[main_preds["target"] == TARGET][["decision_date", "pred_B0_historical_mean"]]
    result = result.drop(columns=["pred_B0_historical_mean"]).merge(main_b0, on="decision_date", how="left")

    OUT_PRED.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PRED / "backtest_markov_switching.csv", index=False)

    from soxstudy.evaluation.metrics import rmse, oos_r2, clark_west_statistic, block_bootstrap_loss_diff_ci

    sub = result.dropna(subset=["y_true", "pred_B0_historical_mean", "pred_M5_markov_switching"])
    y = sub["y_true"].values
    p_bench = sub["pred_B0_historical_mean"].values
    p_ms = sub["pred_M5_markov_switching"].values

    summary = {
        "n": int(len(sub)),
        "n_rejected_fits": int(n_rejected),
        "rmse_benchmark": rmse(y, p_bench),
        "rmse_markov_switching": rmse(y, p_ms),
        "oos_r2_vs_benchmark": oos_r2(y, p_ms, p_bench),
    }
    cw = clark_west_statistic(y, p_bench, p_ms)
    summary["clark_west_p_one_sided"] = cw["p_value_one_sided"]
    boot = block_bootstrap_loss_diff_ci(y, p_ms, p_bench)
    summary["bootstrap_ci_low"] = boot["ci_low"]
    summary["bootstrap_ci_high"] = boot["ci_high"]
    summary["note"] = (
        "SECONDARY ROBUSTNESS MODULE. Compact 2-feature (VIX, industrial-production "
        "growth) 2-regime Markov-switching regression, filtered probabilities only, "
        "estimated fresh at each outer origin. Tests whether data-driven regime "
        "breaks recover forecast value the calendar-defined regime split (Table 3) "
        "only illustrates informally."
    )

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(OUT_TABLES / "table10_markov_switching_robustness.csv", index=False)

    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    with open(OUT_LOGS / "markov_switching_status.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
