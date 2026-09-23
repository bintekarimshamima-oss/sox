"""Robustness and regime-stability checks for the primary h=3 comparison:
pre/post-COVID split, pre/post major regime windows, and exclusion of the
five largest-absolute-return months (crisis-influence check).
"""
import json
from pathlib import Path

import pandas as pd

from soxstudy.evaluation.metrics import (
    rmse, oos_r2, clark_west_statistic, block_bootstrap_loss_diff_ci
)

ROOT = Path(__file__).resolve().parents[3]
PRED_DIR = ROOT / "outputs" / "predictions"
OUT_TABLES = ROOT / "outputs" / "tables"

REGIME_WINDOWS = {
    "pre_covid_2016_2019": ("2016-01-01", "2019-12-31"),
    "covid_shortage_2020_2021": ("2020-01-01", "2021-12-31"),
    "inventory_correction_2022_2023": ("2022-01-01", "2023-12-31"),
    "ai_acceleration_2024_2026": ("2024-01-01", "2026-12-31"),
}


def eval_subset(sub: pd.DataFrame) -> dict:
    has_xgb = "pred_M3_xgboost_fundamentals" in sub.columns
    required = ["y_true", "pred_B0_historical_mean", "pred_M2_elastic_net_fundamentals"]
    sub = sub.dropna(subset=required)
    if len(sub) < 8:
        return {"n": len(sub), "note": "too few observations"}
    y = sub["y_true"].values
    p_bench = sub["pred_B0_historical_mean"].values
    p_enet = sub["pred_M2_elastic_net_fundamentals"].values
    cw = clark_west_statistic(y, p_bench, p_enet)
    boot = block_bootstrap_loss_diff_ci(y, p_enet, p_bench, block_size=min(6, max(2, len(sub) // 4)))
    result = {
        "n": int(len(sub)),
        "rmse_benchmark": rmse(y, p_bench),
        "rmse_enet": rmse(y, p_enet),
        "oos_r2": oos_r2(y, p_enet, p_bench),
        "clark_west_p_one_sided": cw["p_value_one_sided"],
        "bootstrap_ci_low": boot["ci_low"],
        "bootstrap_ci_high": boot["ci_high"],
    }
    if has_xgb and sub["pred_M3_xgboost_fundamentals"].notna().all():
        p_xgb = sub["pred_M3_xgboost_fundamentals"].values
        cw_xgb = clark_west_statistic(y, p_bench, p_xgb)
        boot_xgb = block_bootstrap_loss_diff_ci(y, p_xgb, p_bench, block_size=min(6, max(2, len(sub) // 4)))
        result["rmse_xgb"] = rmse(y, p_xgb)
        result["oos_r2_xgb"] = oos_r2(y, p_xgb, p_bench)
        result["clark_west_p_one_sided_xgb"] = cw_xgb["p_value_one_sided"]
        result["bootstrap_ci_low_xgb"] = boot_xgb["ci_low"]
        result["bootstrap_ci_high_xgb"] = boot_xgb["ci_high"]
    return result


def main():
    preds = pd.read_parquet(PRED_DIR / "backtest_predictions.parquet")
    h3 = preds[preds["target"] == "sox_excess_ret_h3"].copy()
    h3["decision_date"] = pd.to_datetime(h3["decision_date"])

    rows = []
    for regime, (start, end) in REGIME_WINDOWS.items():
        sub = h3[(h3["decision_date"] >= start) & (h3["decision_date"] <= end)]
        result = eval_subset(sub)
        result["regime"] = regime
        rows.append(result)

    full = eval_subset(h3)
    full["regime"] = "full_sample"
    rows.append(full)

    h3_sorted = h3.reindex(h3["y_true"].abs().sort_values(ascending=False).index)
    excl_top5 = h3.drop(h3_sorted.index[:5])
    crisis_check = eval_subset(excl_top5)
    crisis_check["regime"] = "excl_top5_abs_return_months"
    rows.append(crisis_check)

    robustness_df = pd.DataFrame(rows)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    robustness_df.to_csv(OUT_TABLES / "table8_regime_and_robustness_checks.csv", index=False)

    print(robustness_df.to_string(index=False))
    print("\nSaved:", OUT_TABLES / "table8_regime_and_robustness_checks.csv")


if __name__ == "__main__":
    main()
