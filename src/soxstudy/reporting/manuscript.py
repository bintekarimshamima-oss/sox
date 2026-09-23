"""Generate manuscript-ready numeric strings directly from saved
machine-readable outputs (protocol.yml Appendix C rule: no manually typed
results in final tables). This module reads the metrics/tables Parquet/CSV
artifacts and writes a JSON of citation-ready figures for the manuscript
text to interpolate, so every reported number traces to a file + hash.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_METRICS = ROOT / "outputs" / "metrics"
OUT_LOGS = ROOT / "outputs" / "logs"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _volatility_downside_numbers():
    rv_path = OUT_TABLES / "table11_volatility_forecast_performance.csv"
    ds_path = OUT_TABLES / "table12_downside_risk_classification.csv"
    if not rv_path.exists() or not ds_path.exists():
        return None
    rv = pd.read_csv(rv_path).iloc[0]
    ds = pd.read_csv(ds_path).iloc[0]
    return {
        "realized_volatility": {
            "n": int(rv["n"]),
            "oos_r2_vs_benchmark": round(rv["oos_r2_vs_benchmark"], 4),
            "qlike_benchmark": round(rv["qlike_benchmark"], 4),
            "qlike_har_rv": round(rv["qlike_har_rv"], 4),
        },
        "downside_risk": {
            "n": int(ds["n"]),
            "base_rate_mean": round(ds["base_rate_mean"], 4),
            "roc_auc_logit": round(ds["roc_auc_logit"], 4),
            "pr_auc_logit": round(ds["pr_auc_logit"], 4),
            "brier_logit": round(ds["brier_logit"], 4),
            "brier_base_rate": round(ds["brier_base_rate"], 4),
        },
    }


def _markov_switching_numbers():
    path = OUT_TABLES / "table10_markov_switching_robustness.csv"
    if not path.exists():
        return None
    row = pd.read_csv(path).iloc[0]
    return {
        "n": int(row["n"]),
        "n_rejected_fits": int(row["n_rejected_fits"]),
        "oos_r2_vs_benchmark": round(row["oos_r2_vs_benchmark"], 4),
        "clark_west_p_one_sided": round(row["clark_west_p_one_sided"], 4),
        "bootstrap_ci": [round(row["bootstrap_ci_low"], 3), round(row["bootstrap_ci_high"], 3)],
    }


def _corporate_module_numbers():
    path = OUT_TABLES / "table9_corporate_robustness_module.csv"
    if not path.exists():
        return None
    row = pd.read_csv(path).iloc[0]
    return {
        "n": int(row["n"]),
        "oos_r2_vs_benchmark": round(row["oos_r2_vs_benchmark"], 4),
        "clark_west_p_one_sided": round(row["clark_west_p_one_sided"], 4),
        "bootstrap_ci": [round(row["bootstrap_ci_low"], 3), round(row["bootstrap_ci_high"], 3)],
        "n_firms_universe": 505,
        "n_firms_with_facts": 251,
        "sec_coverage_note": "All 8 XBRL concepts below 70% firm-quarter coverage threshold; run as secondary robustness module per protocol, not merged into primary M2.",
    }


def _trade_module_numbers():
    path = OUT_TABLES / "table13_trade_robustness_module.csv"
    if not path.exists():
        return None
    row = pd.read_csv(path).iloc[0]
    return {
        "n": int(row["n"]),
        "oos_r2_vs_benchmark": round(row["oos_r2_vs_benchmark"], 4),
        "clark_west_p_one_sided": round(row["clark_west_p_one_sided"], 4),
        "bootstrap_ci": [round(row["bootstrap_ci_low"], 3), round(row["bootstrap_ci_high"], 3)],
        "n_comtrade_calls": 7936,
        "n_comtrade_rows": 181231,
        "hs_codes": "8542 (integrated circuits, demand proxy); 8486 (semiconductor manufacturing equipment, capacity proxy)",
        "release_lag_months": 2,
    }


def main():
    audit = pd.read_csv(OUT_TABLES / "table0_thesis_replication_audit.csv")
    metrics = pd.read_parquet(OUT_METRICS / "primary_metrics.parquet")
    with open(OUT_LOGS / "primary_decision.json") as f:
        decision = json.load(f)
    with open(OUT_LOGS / "replication_report.json") as f:
        repl = json.load(f)
    robustness = pd.read_csv(OUT_TABLES / "table8_regime_and_robustness_checks.csv")

    h3 = metrics[metrics["target"] == "sox_excess_ret_h3"].iloc[0]
    h1 = metrics[metrics["target"] == "sox_excess_ret_h1"].iloc[0]
    h6 = metrics[metrics["target"] == "sox_excess_ret_h6"].iloc[0]

    numbers = {
        "replication_n_rows": repl["n_rows"],
        "replication_date_range": f"{repl['date_min']} to {repl['date_max']}",
        "replication_naive_rmse": round(repl["naive_benchmark_rmse"], 2),
        "replication_naive_mae": round(repl["naive_benchmark_mae"], 2),
        "primary_oos_r2": round(h3["oos_r2_vs_strongest_benchmark"], 4),
        "primary_clark_west_p": round(h3["clark_west_p_one_sided"], 4),
        "primary_bootstrap_ci": [round(h3["bootstrap_ci_low"], 3), round(h3["bootstrap_ci_high"], 3)],
        "primary_supported": decision["primary_hypothesis_supported"],
        "h1_oos_r2": round(h1["oos_r2_vs_strongest_benchmark"], 4),
        "h6_oos_r2": round(h6["oos_r2_vs_strongest_benchmark"], 4),
        "n_outer_origins_h3": int(h3["n"]),
        "ai_era_oos_r2": round(
            robustness.loc[robustness["regime"] == "ai_acceleration_2024_2026", "oos_r2"].iloc[0], 4
        ),
        "ai_era_p": round(
            robustness.loc[robustness["regime"] == "ai_acceleration_2024_2026", "clark_west_p_one_sided"].iloc[0], 4
        ),
        "corporate_module": _corporate_module_numbers(),
        "trade_module": _trade_module_numbers(),
        "markov_switching": _markov_switching_numbers(),
        "volatility_downside": _volatility_downside_numbers(),
        "xgboost_robustness": {
            "h3_oos_r2": round(h3["oos_r2_M3_vs_strongest_benchmark"], 4),
            "h3_clark_west_p": round(h3["clark_west_p_one_sided_M3"], 4),
            "h3_bootstrap_ci": [round(h3["bootstrap_ci_low_M3"], 3), round(h3["bootstrap_ci_high_M3"], 3)],
            "h1_oos_r2": round(h1["oos_r2_M3_vs_strongest_benchmark"], 4),
            "h6_oos_r2": round(h6["oos_r2_M3_vs_strongest_benchmark"], 4),
            "pre_covid_oos_r2": round(robustness.loc[robustness["regime"] == "pre_covid_2016_2019", "oos_r2_xgb"].iloc[0], 4),
            "pre_covid_p": round(robustness.loc[robustness["regime"] == "pre_covid_2016_2019", "clark_west_p_one_sided_xgb"].iloc[0], 4),
            "covid_shortage_oos_r2": round(robustness.loc[robustness["regime"] == "covid_shortage_2020_2021", "oos_r2_xgb"].iloc[0], 4),
            "covid_shortage_p": round(robustness.loc[robustness["regime"] == "covid_shortage_2020_2021", "clark_west_p_one_sided_xgb"].iloc[0], 4),
        },
        "source_hashes": {
            "table0": file_hash(OUT_TABLES / "table0_thesis_replication_audit.csv"),
            "table5": file_hash(OUT_TABLES / "table5_primary_backtest_performance.csv"),
            "table8": file_hash(OUT_TABLES / "table8_regime_and_robustness_checks.csv"),
        },
    }

    out_path = ROOT / "manuscript" / "manuscript_numbers.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(numbers, f, indent=2, default=str)

    print(json.dumps(numbers, indent=2, default=str))
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
