"""Point-forecast evaluation: RMSE/MAE, out-of-sample R^2 vs a benchmark, and
the Clark-West (2007) test for comparing nested model forecasts, plus a
stationary block bootstrap confidence interval for the loss differential.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
PRED_DIR = ROOT / "outputs" / "predictions"
OUT_METRICS = ROOT / "outputs" / "metrics"
OUT_TABLES = ROOT / "outputs" / "tables"


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def oos_r2(y_true, y_pred_model, y_pred_benchmark):
    y_true = np.asarray(y_true)
    sse_model = np.sum((y_true - np.asarray(y_pred_model)) ** 2)
    sse_bench = np.sum((y_true - np.asarray(y_pred_benchmark)) ** 2)
    return float(1 - sse_model / sse_bench)


def clark_west_statistic(y_true, pred_restricted, pred_unrestricted):
    """Clark & West (2007) MSPE-adjusted test for comparing nested models.
    H0: restricted (benchmark) model forecasts as well as the unrestricted
    (larger) model out of sample. One-sided: reject in favor of the larger
    model when CW statistic is large and positive.
    """
    y_true = np.asarray(y_true, dtype=float)
    f_restricted = np.asarray(pred_restricted, dtype=float)
    f_unrestricted = np.asarray(pred_unrestricted, dtype=float)

    e_r = y_true - f_restricted
    e_u = y_true - f_unrestricted
    f_hat = e_r ** 2 - (e_u ** 2 - (f_restricted - f_unrestricted) ** 2)

    n = len(f_hat)
    mean_f = np.mean(f_hat)
    se_f = np.std(f_hat, ddof=1) / np.sqrt(n)
    if se_f == 0 or np.isnan(se_f):
        return {"cw_stat": float("nan"), "p_value_one_sided": float("nan"), "n": n}
    t_stat = mean_f / se_f
    p_one_sided = 1 - stats.norm.cdf(t_stat)
    return {"cw_stat": float(t_stat), "p_value_one_sided": float(p_one_sided), "n": int(n)}


def block_bootstrap_loss_diff_ci(y_true, pred_a, pred_b, block_size=6, n_boot=2000, seed=42):
    """Stationary/moving block bootstrap CI for E[loss_a - loss_b] (squared error).
    Negative values favor model A (lower loss). CI excluding zero => significant.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    loss_a = (y_true - np.asarray(pred_a, dtype=float)) ** 2
    loss_b = (y_true - np.asarray(pred_b, dtype=float)) ** 2
    diff = loss_a - loss_b
    n = len(diff)
    if n < block_size * 2:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "mean_diff": float(np.mean(diff))}

    n_blocks = int(np.ceil(n / block_size))
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([diff[s:s + block_size] for s in starts])[:n]
        boot_means[b] = sample.mean()

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return {"ci_low": float(ci_low), "ci_high": float(ci_high), "mean_diff": float(np.mean(diff))}


def evaluate_target(df: pd.DataFrame, target: str) -> dict:
    sub = df[df["target"] == target].dropna(
        subset=["y_true", "pred_B0_historical_mean", "pred_B1_market_ols", "pred_M2_elastic_net_fundamentals"]
    )
    if len(sub) < 10:
        return {"target": target, "n": len(sub), "note": "insufficient observations"}

    has_xgb = "pred_M3_xgboost_fundamentals" in sub.columns and sub["pred_M3_xgboost_fundamentals"].notna().any()
    subset_cols = ["y_true", "pred_B0_historical_mean", "pred_B1_market_ols", "pred_M2_elastic_net_fundamentals"]
    if has_xgb:
        subset_cols.append("pred_M3_xgboost_fundamentals")

    y = sub["y_true"].values
    p_hist = sub["pred_B0_historical_mean"].values
    p_mkt = sub["pred_B1_market_ols"].values
    p_enet = sub["pred_M2_elastic_net_fundamentals"].values

    result = {
        "target": target,
        "n": int(len(sub)),
        "rmse_B0_historical_mean": rmse(y, p_hist),
        "rmse_B1_market_ols": rmse(y, p_mkt),
        "rmse_M2_elastic_net_fundamentals": rmse(y, p_enet),
        "mae_B0_historical_mean": mae(y, p_hist),
        "mae_B1_market_ols": mae(y, p_mkt),
        "mae_M2_elastic_net_fundamentals": mae(y, p_enet),
    }

    strongest_benchmark = p_mkt if result["rmse_B1_market_ols"] <= result["rmse_B0_historical_mean"] else p_hist
    strongest_benchmark_name = "B1_market_ols" if result["rmse_B1_market_ols"] <= result["rmse_B0_historical_mean"] else "B0_historical_mean"
    result["strongest_benchmark"] = strongest_benchmark_name

    result["oos_r2_vs_strongest_benchmark"] = oos_r2(y, p_enet, strongest_benchmark)

    cw = clark_west_statistic(y, strongest_benchmark, p_enet)
    result["clark_west_stat"] = cw["cw_stat"]
    result["clark_west_p_one_sided"] = cw["p_value_one_sided"]

    boot = block_bootstrap_loss_diff_ci(y, p_enet, strongest_benchmark)
    result["bootstrap_loss_diff_mean"] = boot["mean_diff"]
    result["bootstrap_ci_low"] = boot["ci_low"]
    result["bootstrap_ci_high"] = boot["ci_high"]
    result["bootstrap_ci_excludes_zero"] = bool(boot["ci_high"] < 0 or boot["ci_low"] > 0)

    if has_xgb:
        p_xgb = sub["pred_M3_xgboost_fundamentals"].values
        result["rmse_M3_xgboost_fundamentals"] = rmse(y, p_xgb)
        result["mae_M3_xgboost_fundamentals"] = mae(y, p_xgb)
        result["oos_r2_M3_vs_strongest_benchmark"] = oos_r2(y, p_xgb, strongest_benchmark)
        cw_xgb = clark_west_statistic(y, strongest_benchmark, p_xgb)
        result["clark_west_stat_M3"] = cw_xgb["cw_stat"]
        result["clark_west_p_one_sided_M3"] = cw_xgb["p_value_one_sided"]
        boot_xgb = block_bootstrap_loss_diff_ci(y, p_xgb, strongest_benchmark)
        result["bootstrap_ci_low_M3"] = boot_xgb["ci_low"]
        result["bootstrap_ci_high_M3"] = boot_xgb["ci_high"]
        result["bootstrap_ci_excludes_zero_M3"] = bool(boot_xgb["ci_high"] < 0 or boot_xgb["ci_low"] > 0)

    return result


def main():
    preds = pd.read_parquet(PRED_DIR / "backtest_predictions.parquet")

    results = []
    for target in ["sox_excess_ret_h1", "sox_excess_ret_h3", "sox_excess_ret_h6"]:
        results.append(evaluate_target(preds, target))

    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(results)
    metrics_df.to_parquet(OUT_METRICS / "primary_metrics.parquet", index=False)
    metrics_df.to_csv(OUT_TABLES / "table5_primary_backtest_performance.csv", index=False)

    primary = next(r for r in results if r["target"] == "sox_excess_ret_h3")
    decision = {
        "primary_target": "sox_excess_ret_h3",
        "positive_oos_r2": primary.get("oos_r2_vs_strongest_benchmark", float("nan")) > 0,
        "oos_r2": primary.get("oos_r2_vs_strongest_benchmark"),
        "clark_west_p_one_sided": primary.get("clark_west_p_one_sided"),
        "significant_at_05": (primary.get("clark_west_p_one_sided") or 1.0) < 0.05,
        "bootstrap_ci_excludes_zero": primary.get("bootstrap_ci_excludes_zero"),
        "primary_hypothesis_supported": (
            primary.get("oos_r2_vs_strongest_benchmark", -1) > 0
            and (primary.get("clark_west_p_one_sided") or 1.0) < 0.05
            and primary.get("bootstrap_ci_excludes_zero", False)
        ),
    }
    with open(ROOT / "outputs" / "logs" / "primary_decision.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, default=str)

    print(json.dumps(results, indent=2, default=str))
    print("\nPRIMARY DECISION:", json.dumps(decision, indent=2, default=str))


if __name__ == "__main__":
    main()
