"""Secondary robustness model: adds the UN Comtrade semiconductor trade-flow
block (HS 8542 integrated-circuit imports as a demand proxy; HS 8486
semiconductor-manufacturing-equipment imports as a capacity-investment proxy)
to the primary market+macro feature set. Reported as a REMOVABLE ROBUSTNESS
MODULE (M2d), not folded into the primary M2 model or the primary hypothesis
test, consistent with the treatment of the SEC XBRL corporate block (M2c).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from soxstudy.models.backtest import (
    MARKET_ONLY_FEATURES, FUNDAMENTALS_EXTRA_FEATURES,
    fit_elastic_net_with_inner_cv, historical_mean_benchmark,
)
from soxstudy.evaluation.metrics import (
    rmse, oos_r2, clark_west_statistic, block_bootstrap_loss_diff_ci
)

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"
OUT_PRED = ROOT / "outputs" / "predictions"
OUT_TABLES = ROOT / "outputs" / "tables"

TRADE_FEATURES = ["demand_growth_yoy", "demand_growth_1m", "equip_growth_yoy"]
TARGET = "sox_excess_ret_h3"
HORIZON = 3


def run_trade_robustness_backtest(df: pd.DataFrame, initial_train_end="2015-12-31"):
    df = df.sort_values("decision_date").reset_index(drop=True)
    for c in TRADE_FEATURES:
        df[c] = df[c].ffill(limit=3)

    origins = df.index[df["decision_date"] > pd.Timestamp(initial_train_end)]
    all_features = MARKET_ONLY_FEATURES + FUNDAMENTALS_EXTRA_FEATURES + TRADE_FEATURES

    records = []
    for origin_idx in origins:
        origin_date = df.loc[origin_idx, "decision_date"]
        y_true = df.loc[origin_idx, TARGET]
        if pd.isna(y_true):
            continue

        train_end_idx = origin_idx - HORIZON
        if train_end_idx < 24:
            continue
        train_df = df.iloc[:train_end_idx]

        X_train = train_df[all_features]
        y_train = train_df[TARGET]
        x_row = df.loc[[origin_idx], all_features]

        hist_pred = historical_mean_benchmark(y_train)

        fit = fit_elastic_net_with_inner_cv(X_train, y_train, HORIZON)
        pred = hist_pred
        if fit is not None and x_row.notna().all(axis=1).iloc[0]:
            xs = fit["scaler"].transform(x_row.values)
            pred = float(fit["model"].predict(xs)[0])

        records.append({
            "decision_date": origin_date,
            "y_true": float(y_true),
            "pred_B0_historical_mean": hist_pred,
            "pred_M2d_elastic_net_plus_trade": pred,
            "n_train_obs": int(len(train_df)),
        })

    return pd.DataFrame.from_records(records)


def main():
    df = pd.read_parquet(PROCESSED / "features_monthly_extended.parquet")
    if not all(c in df.columns for c in TRADE_FEATURES):
        print("Trade-flow features not present in extended panel; skipping robustness run.")
        return

    result = run_trade_robustness_backtest(df)
    OUT_PRED.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PRED / "backtest_trade_robustness.csv", index=False)

    sub = result.dropna(subset=["y_true", "pred_B0_historical_mean", "pred_M2d_elastic_net_plus_trade"])
    y = sub["y_true"].values
    p_bench = sub["pred_B0_historical_mean"].values
    p_model = sub["pred_M2d_elastic_net_plus_trade"].values

    summary = {
        "n": int(len(sub)),
        "rmse_benchmark": rmse(y, p_bench),
        "rmse_with_trade_block": rmse(y, p_model),
        "oos_r2_vs_benchmark": oos_r2(y, p_model, p_bench),
    }
    cw = clark_west_statistic(y, p_bench, p_model)
    summary["clark_west_p_one_sided"] = cw["p_value_one_sided"]
    boot = block_bootstrap_loss_diff_ci(y, p_model, p_bench)
    summary["bootstrap_ci_low"] = boot["ci_low"]
    summary["bootstrap_ci_high"] = boot["ci_high"]
    summary["note"] = (
        "SECONDARY ROBUSTNESS MODULE ONLY. UN Comtrade HS 8542 (integrated "
        "circuits) import flows proxy end-demand and HS 8486 (semiconductor "
        "manufacturing equipment) import flows proxy capacity investment, both "
        "entered with a conservative 2-month release lag. Reported separately "
        "from the primary M2 model and primary hypothesis test, not merged "
        "into them."
    )

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(OUT_TABLES / "table13_trade_robustness_module.csv", index=False)

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
