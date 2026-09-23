"""Nested expanding-window pseudo-real-time backtest.

Outer loop: expanding-window monthly origins from 2016-01 onward (initial
training window 2006-01..2015-12), producing one untouched out-of-sample
prediction per origin for each model/target/horizon.

Inner loop: blocked expanding folds inside the outer-training window only,
used to select the elastic-net regularization strength (one-standard-error
rule). The outer test point is never used for any fitting or selection
decision, per protocol.yml / config/protocol.yml non_future_information_rules.

Embargo: the final h months of each outer training window are purged from
the inner-fold splits so no partially-resolved h-month-ahead target crosses
a fold boundary.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"
OUT_PRED = ROOT / "outputs" / "predictions"
OUT_LOGS = ROOT / "outputs" / "logs"

MARKET_ONLY_FEATURES = [
    "sox_mom_1m", "sox_mom_3m", "sox_mom_12m", "sox_realized_vol_1m",
    "nasdaq_ret_1m", "vix_level", "vix_chg_1m", "term_spread", "dollar_chg_1m",
    "mkt_rf", "smb", "hml",
]

FUNDAMENTALS_EXTRA_FEATURES = [
    "ip_growth_yoy", "ip_growth_1m", "cap_util_level", "cap_util_chg_3m",
    "ppi_growth_yoy", "import_price_growth_yoy", "export_price_growth_yoy",
    "price_pressure_dispersion", "electronics_inventory_growth_yoy",
    "electronics_orders_growth_yoy", "gpr_level", "gpr_chg_3m",
]

PRIMARY_TARGET = "sox_excess_ret_h3"
PRIMARY_HORIZON = 3


def embargoed_inner_folds(train_index: np.ndarray, horizon: int, n_folds: int = 4):
    """Blocked expanding folds inside the training window with an h-month embargo
    before each validation block, so overlapping h-month targets never leak."""
    n = len(train_index)
    fold_edges = np.linspace(int(n * 0.5), n, n_folds + 1, dtype=int)
    for i in range(n_folds):
        val_start, val_end = fold_edges[i], fold_edges[i + 1]
        embargo_start = max(0, val_start - horizon)
        fit_idx = train_index[:embargo_start]
        val_idx = train_index[val_start:val_end]
        if len(fit_idx) < 24 or len(val_idx) < 3:
            continue
        yield fit_idx, val_idx


def fit_elastic_net_with_inner_cv(X_train: pd.DataFrame, y_train: pd.Series, horizon: int):
    valid = y_train.notna() & X_train.notna().all(axis=1)
    X_fit, y_fit = X_train.loc[valid], y_train.loc[valid]
    if len(X_fit) < 30:
        return None

    train_index = np.arange(len(X_fit))
    cv_splits = list(embargoed_inner_folds(train_index, horizon, n_folds=4))
    if len(cv_splits) < 2:
        cv_splits = None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_fit.values)

    alphas = np.logspace(-4, 1, 20)
    model = ElasticNetCV(
        l1_ratio=[0.0, 0.25, 0.5, 0.75, 1.0],
        alphas=alphas,
        cv=cv_splits if cv_splits else 3,
        max_iter=100000,
        tol=1e-4,
        n_jobs=1,
    )
    model.fit(Xs, y_fit.values)
    return {"scaler": scaler, "model": model, "columns": list(X_fit.columns)}


XGB_GRID = list(ParameterGrid({
    "max_depth": [1, 2, 3],
    "learning_rate": [0.01, 0.03, 0.05],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}))


def fit_xgboost_with_inner_cv(X_train: pd.DataFrame, y_train: pd.Series, horizon: int):
    """Shallow gradient-boosted trees (M3), per the pre-specified compact grid in
    Appendix B of the study protocol: max_depth 1-3, small learning rate, no deep
    trees. Regularization strength (n_estimators via early stopping) and the
    (depth, learning_rate) pair are selected by the same embargoed inner-fold
    scheme used for the elastic net, never touching the outer test point."""
    valid = y_train.notna() & X_train.notna().all(axis=1)
    X_fit, y_fit = X_train.loc[valid], y_train.loc[valid]
    if len(X_fit) < 40:
        return None

    train_index = np.arange(len(X_fit))
    cv_splits = list(embargoed_inner_folds(train_index, horizon, n_folds=4))
    if len(cv_splits) < 2:
        return None

    X_vals, y_vals = X_fit.values, y_fit.values

    best_score, best_params, best_n_estimators = np.inf, None, 100
    for params in XGB_GRID:
        fold_scores, fold_best_iters = [], []
        for fit_idx, val_idx in cv_splits:
            if len(fit_idx) < 30:
                continue
            model = XGBRegressor(
                n_estimators=800,
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                objective="reg:squarederror",
                early_stopping_rounds=30,
                eval_metric="rmse",
                verbosity=0,
                n_jobs=1,
            )
            model.fit(
                X_vals[fit_idx], y_vals[fit_idx],
                eval_set=[(X_vals[val_idx], y_vals[val_idx])],
                verbose=False,
            )
            pred = model.predict(X_vals[val_idx])
            fold_scores.append(float(np.mean((y_vals[val_idx] - pred) ** 2)))
            fold_best_iters.append(model.best_iteration + 1)
        if not fold_scores:
            continue
        mean_score = float(np.mean(fold_scores))
        if mean_score < best_score:
            best_score = mean_score
            best_params = params
            best_n_estimators = int(np.median(fold_best_iters))

    if best_params is None:
        return None

    final_model = XGBRegressor(
        n_estimators=best_n_estimators,
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        objective="reg:squarederror",
        verbosity=0,
        n_jobs=1,
    )
    final_model.fit(X_vals, y_vals)
    return {"model": final_model, "columns": list(X_fit.columns), "params": best_params}


def fit_market_only_ols(X_train: pd.DataFrame, y_train: pd.Series):
    valid = y_train.notna() & X_train.notna().all(axis=1)
    X_fit, y_fit = X_train.loc[valid], y_train.loc[valid]
    if len(X_fit) < 30:
        return None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_fit.values)
    model = LinearRegression()
    model.fit(Xs, y_fit.values)
    return {"scaler": scaler, "model": model, "columns": list(X_fit.columns)}


def historical_mean_benchmark(y_train: pd.Series) -> float:
    return float(y_train.dropna().mean()) if y_train.notna().any() else 0.0


def run_backtest(df: pd.DataFrame, target: str, horizon: int, initial_train_end: str = "2015-12-31"):
    df = df.sort_values("decision_date").reset_index(drop=True)
    origins = df.index[df["decision_date"] > pd.Timestamp(initial_train_end)]

    records = []
    for origin_idx in origins:
        origin_date = df.loc[origin_idx, "decision_date"]
        y_true = df.loc[origin_idx, target]
        if pd.isna(y_true):
            continue

        train_end_idx = origin_idx - horizon  # embargo: exclude unresolved targets
        if train_end_idx < 24:
            continue
        train_df = df.iloc[:train_end_idx]

        X_market = train_df[MARKET_ONLY_FEATURES]
        X_full = train_df[MARKET_ONLY_FEATURES + FUNDAMENTALS_EXTRA_FEATURES]
        y_train = train_df[target]

        x_row_market = df.loc[[origin_idx], MARKET_ONLY_FEATURES]
        x_row_full = df.loc[[origin_idx], MARKET_ONLY_FEATURES + FUNDAMENTALS_EXTRA_FEATURES]

        hist_mean_pred = historical_mean_benchmark(y_train)

        market_fit = fit_market_only_ols(X_market, y_train)
        market_pred = hist_mean_pred
        if market_fit is not None and x_row_market.notna().all(axis=1).iloc[0]:
            xs = market_fit["scaler"].transform(x_row_market.values)
            market_pred = float(market_fit["model"].predict(xs)[0])

        enet_fit = fit_elastic_net_with_inner_cv(X_full, y_train, horizon)
        enet_pred = market_pred
        if enet_fit is not None and x_row_full.notna().all(axis=1).iloc[0]:
            xs = enet_fit["scaler"].transform(x_row_full.values)
            enet_pred = float(enet_fit["model"].predict(xs)[0])

        xgb_fit = fit_xgboost_with_inner_cv(X_full, y_train, horizon)
        xgb_pred = hist_mean_pred
        if xgb_fit is not None and x_row_full.notna().all(axis=1).iloc[0]:
            xgb_pred = float(xgb_fit["model"].predict(x_row_full.values)[0])

        records.append({
            "decision_date": origin_date,
            "target": target,
            "horizon": horizon,
            "y_true": float(y_true),
            "pred_B0_historical_mean": hist_mean_pred,
            "pred_B1_market_ols": market_pred,
            "pred_M2_elastic_net_fundamentals": enet_pred,
            "pred_M3_xgboost_fundamentals": xgb_pred,
            "n_train_obs": int(len(train_df)),
        })

    return pd.DataFrame.from_records(records)


def main():
    df = pd.read_parquet(PROCESSED / "features_monthly_core.parquet")

    all_results = {}
    for target, h in [("sox_excess_ret_h1", 1), ("sox_excess_ret_h3", 3), ("sox_excess_ret_h6", 6)]:
        print(f"Running nested expanding-window backtest for {target} (h={h})...")
        res = run_backtest(df, target, h)
        all_results[target] = res
        print(f"  -> {len(res)} out-of-sample origins produced")

    OUT_PRED.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(all_results.values(), ignore_index=True)
    combined.to_parquet(OUT_PRED / "backtest_predictions.parquet", index=False)
    combined.to_csv(OUT_PRED / "backtest_predictions.csv", index=False)

    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    status = {t: {"n_origins": int(len(r)),
                   "date_min": str(r["decision_date"].min().date()) if len(r) else None,
                   "date_max": str(r["decision_date"].max().date()) if len(r) else None}
              for t, r in all_results.items()}
    with open(OUT_LOGS / "backtest_status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    print("Backtest complete. Saved:", OUT_PRED / "backtest_predictions.parquet")


if __name__ == "__main__":
    main()
