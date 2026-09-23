"""Forecast models for the two secondary outcomes named in the study design
but not previously modeled: next-month realized volatility (HAR-RV) and the
three-month downside-risk indicator (logistic regression). Both use the same
nested expanding-window, embargoed, origin-only-refit discipline as the
primary return-forecasting models (backtest.py).

HAR-RV (Corsi, 2009) regresses future realized volatility on lagged
volatility components at three horizons (last month, 3-month average,
12-month average) -- the standard parsimonious volatility-forecasting
benchmark, chosen over a full GARCH because monthly (not daily) est
imation frequency here makes GARCH's own within-sample dynamics a poor fit.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from soxstudy.models.backtest import MARKET_ONLY_FEATURES, FUNDAMENTALS_EXTRA_FEATURES

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"
OUT_PRED = ROOT / "outputs" / "predictions"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_LOGS = ROOT / "outputs" / "logs"

HAR_FEATURES = ["rv_lag1", "rv_lag_avg3", "rv_lag_avg12"]
DOWNSIDE_FEATURES = MARKET_ONLY_FEATURES + FUNDAMENTALS_EXTRA_FEATURES


def historical_mean_benchmark(y_train: pd.Series) -> float:
    return float(y_train.dropna().mean()) if y_train.notna().any() else 0.0


def run_har_rv_backtest(df: pd.DataFrame, initial_train_end="2015-12-31"):
    """HAR-RV forecast of next-month realized volatility, refit at every
    origin on training data only (no future RV observation ever enters
    fitting), horizon h=1 so the embargo is a single month."""
    df = df.sort_values("decision_date").reset_index(drop=True)
    origins = df.index[df["decision_date"] > pd.Timestamp(initial_train_end)]

    records = []
    for origin_idx in origins:
        origin_date = df.loc[origin_idx, "decision_date"]
        y_true = df.loc[origin_idx, "sox_rv_next"]
        if pd.isna(y_true):
            continue

        train_end_idx = origin_idx - 1  # h=1 embargo
        if train_end_idx < 24:
            continue
        train_df = df.iloc[:train_end_idx]

        valid = train_df[HAR_FEATURES + ["sox_rv_next"]].notna().all(axis=1)
        train_valid = train_df.loc[valid]
        if len(train_valid) < 24:
            continue

        y_train = train_valid["sox_rv_next"]
        X_train = train_valid[HAR_FEATURES]
        hist_pred = historical_mean_benchmark(y_train)

        x_row = df.loc[[origin_idx], HAR_FEATURES]
        har_pred = hist_pred
        if x_row.notna().all(axis=1).iloc[0]:
            model = LinearRegression()
            model.fit(X_train.values, y_train.values)
            har_pred = float(model.predict(x_row.values)[0])
            har_pred = max(har_pred, 0.0)  # volatility cannot be negative

        records.append({
            "decision_date": origin_date,
            "y_true": float(y_true),
            "pred_B0_historical_mean": hist_pred,
            "pred_HAR_RV": har_pred,
            "n_train_obs": int(len(train_valid)),
        })

    return pd.DataFrame.from_records(records)


def run_downside_logit_backtest(df: pd.DataFrame, initial_train_end="2015-12-31", horizon=3):
    """Logistic regression forecast of the 3-month downside-risk indicator,
    same embargoed expanding-window discipline as the return models."""
    df = df.sort_values("decision_date").reset_index(drop=True)
    origins = df.index[df["decision_date"] > pd.Timestamp(initial_train_end)]

    records = []
    for origin_idx in origins:
        origin_date = df.loc[origin_idx, "decision_date"]
        y_true = df.loc[origin_idx, "sox_downside_h3"]
        if pd.isna(y_true):
            continue

        train_end_idx = origin_idx - horizon
        if train_end_idx < 24:
            continue
        train_df = df.iloc[:train_end_idx]

        valid = train_df[DOWNSIDE_FEATURES + ["sox_downside_h3"]].notna().all(axis=1)
        train_valid = train_df.loc[valid]
        if len(train_valid) < 30 or train_valid["sox_downside_h3"].nunique() < 2:
            continue

        y_train = train_valid["sox_downside_h3"]
        X_train = train_valid[DOWNSIDE_FEATURES]
        base_rate = float(y_train.mean())

        x_row = df.loc[[origin_idx], DOWNSIDE_FEATURES]
        prob_pred = base_rate
        if x_row.notna().all(axis=1).iloc[0]:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X_train.values)
            model = LogisticRegression(C=1.0, max_iter=2000)
            model.fit(Xs, y_train.values)
            xs_row = scaler.transform(x_row.values)
            prob_pred = float(model.predict_proba(xs_row)[0, 1])

        records.append({
            "decision_date": origin_date,
            "y_true": float(y_true),
            "pred_base_rate": base_rate,
            "pred_logit": prob_pred,
            "n_train_obs": int(len(train_valid)),
        })

    return pd.DataFrame.from_records(records)


def evaluate_volatility(result: pd.DataFrame) -> dict:
    from soxstudy.evaluation.metrics import rmse, oos_r2

    sub = result.dropna(subset=["y_true", "pred_B0_historical_mean", "pred_HAR_RV"])
    y, p_bench, p_har = sub["y_true"].values, sub["pred_B0_historical_mean"].values, sub["pred_HAR_RV"].values

    # QLIKE loss (Patton, 2011), the standard volatility-forecast loss that is
    # robust to noisy volatility proxies, computed on variance scale (RV^2).
    def qlike(y_var, pred_var):
        pred_var = np.clip(pred_var, 1e-8, None)
        return float(np.mean(y_var / pred_var - np.log(y_var / pred_var) - 1))

    y_var, bench_var, har_var = y ** 2, p_bench ** 2, p_har ** 2

    return {
        "n": int(len(sub)),
        "rmse_benchmark": rmse(y, p_bench),
        "rmse_har_rv": rmse(y, p_har),
        "oos_r2_vs_benchmark": oos_r2(y, p_har, p_bench),
        "qlike_benchmark": qlike(y_var, bench_var),
        "qlike_har_rv": qlike(y_var, har_var),
    }


def evaluate_downside(result: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

    sub = result.dropna(subset=["y_true", "pred_base_rate", "pred_logit"])
    y = sub["y_true"].values
    p_base = sub["pred_base_rate"].values
    p_logit = np.clip(sub["pred_logit"].values, 1e-6, 1 - 1e-6)

    metrics = {"n": int(len(sub)), "base_rate_mean": float(p_base.mean())}
    if len(np.unique(y)) < 2:
        metrics["note"] = "degenerate outcome in test window; AUC undefined"
        return metrics

    metrics["roc_auc_logit"] = float(roc_auc_score(y, p_logit))
    metrics["pr_auc_logit"] = float(average_precision_score(y, p_logit))
    metrics["brier_logit"] = float(brier_score_loss(y, p_logit))
    metrics["brier_base_rate"] = float(brier_score_loss(y, p_base))
    metrics["log_loss_logit"] = float(log_loss(y, p_logit, labels=[0, 1]))
    metrics["log_loss_base_rate"] = float(log_loss(y, p_base, labels=[0, 1]))
    return metrics


def main():
    df = pd.read_parquet(PROCESSED / "features_monthly_core.parquet")

    print("Running HAR-RV backtest for next-month realized volatility...")
    rv_result = run_har_rv_backtest(df)
    print(f"  -> {len(rv_result)} origins produced")
    OUT_PRED.mkdir(parents=True, exist_ok=True)
    rv_result.to_csv(OUT_PRED / "backtest_har_rv.csv", index=False)
    rv_metrics = evaluate_volatility(rv_result)

    print("Running logistic regression backtest for 3-month downside risk...")
    downside_result = run_downside_logit_backtest(df)
    print(f"  -> {len(downside_result)} origins produced")
    downside_result.to_csv(OUT_PRED / "backtest_downside_logit.csv", index=False)
    downside_metrics = evaluate_downside(downside_result)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([rv_metrics]).to_csv(OUT_TABLES / "table11_volatility_forecast_performance.csv", index=False)
    pd.DataFrame([downside_metrics]).to_csv(OUT_TABLES / "table12_downside_risk_classification.csv", index=False)

    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    summary = {"realized_volatility": rv_metrics, "downside_risk": downside_metrics}
    with open(OUT_LOGS / "volatility_downside_status.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
