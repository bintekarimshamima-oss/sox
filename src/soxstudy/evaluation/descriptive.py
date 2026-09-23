"""Descriptive and measurement analysis on the core monthly panel: coverage,
correlations, rolling stability, and a structural-break screen (CUSUM-style)
ahead of prediction. This is the measurement section the protocol calls a
publishable contribution in its own right (section 6.4 of the proposal).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGS = ROOT / "outputs" / "figures"


def rolling_corr_stability(df: pd.DataFrame, feature: str, target: str, window: int = 60) -> pd.Series:
    return df[feature].rolling(window).corr(df[target])


def cusum_screen(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) < 20:
        return {"max_abs_cusum": None, "break_flag": False}
    resid = s - s.mean()
    cusum = resid.cumsum() / (s.std() * np.sqrt(len(s)))
    max_abs = float(cusum.abs().max())
    # Approximate 5% critical boundary for standardized CUSUM (Brownian bridge, ~1.36)
    return {"max_abs_cusum": max_abs, "break_flag": bool(max_abs > 1.36)}


def main():
    df = pd.read_parquet(PROCESSED / "features_monthly_core.parquet")
    df = df.sort_values("decision_date").reset_index(drop=True)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    feature_cols = [
        "sox_mom_1m", "sox_mom_3m", "sox_mom_12m", "sox_realized_vol_1m",
        "nasdaq_ret_1m", "vix_level", "vix_chg_1m", "term_spread", "dollar_chg_1m",
        "ip_growth_yoy", "ip_growth_1m", "cap_util_level", "cap_util_chg_3m",
        "ppi_growth_yoy", "import_price_growth_yoy", "export_price_growth_yoy",
        "price_pressure_dispersion", "electronics_inventory_growth_yoy",
        "electronics_orders_growth_yoy", "gpr_level", "gpr_chg_3m",
    ]

    coverage = df[feature_cols].notna().mean().rename("coverage_ratio").reset_index()
    coverage.columns = ["feature", "coverage_ratio"]
    coverage.to_csv(OUT_TABLES / "table1_feature_coverage.csv", index=False)

    corr_matrix = df[feature_cols].corr()
    corr_matrix.to_csv(OUT_TABLES / "table2_feature_correlation_matrix.csv")

    lead_lag_rows = []
    for feat in feature_cols:
        for h, target_col in [(1, "sox_excess_ret_h1"), (3, "sox_excess_ret_h3"), (6, "sox_excess_ret_h6")]:
            pair = df[[feat, target_col]].dropna()
            if len(pair) < 24:
                continue
            corr = pair[feat].corr(pair[target_col])
            lead_lag_rows.append({"feature": feat, "horizon": h, "correlation_with_forward_excess_return": corr, "n": len(pair)})
    lead_lag_df = pd.DataFrame(lead_lag_rows)
    lead_lag_df.to_csv(OUT_TABLES / "table3_lead_lag_correlations.csv", index=False)

    vif_rows = []
    X = df[feature_cols].dropna()
    if len(X) > len(feature_cols) + 5:
        from numpy.linalg import LinAlgError
        Xs = (X - X.mean()) / X.std()
        corr_x = Xs.corr().values
        try:
            inv_corr = np.linalg.inv(corr_x)
            vifs = np.diag(inv_corr)
            for feat, vif in zip(feature_cols, vifs):
                vif_rows.append({"feature": feat, "vif": float(vif)})
        except LinAlgError:
            pass
    pd.DataFrame(vif_rows).to_csv(OUT_TABLES / "table4_variance_inflation_factors.csv", index=False)

    break_rows = []
    for feat in feature_cols + ["sox_excess_ret_h3"]:
        result = cusum_screen(df[feat])
        result["feature"] = feat
        break_rows.append(result)
    pd.DataFrame(break_rows).to_csv(OUT_TABLES / "table_structural_break_screen.csv", index=False)

    report = {
        "n_features": len(feature_cols),
        "mean_coverage": float(coverage["coverage_ratio"].mean()),
        "min_coverage_feature": coverage.loc[coverage["coverage_ratio"].idxmin(), "feature"],
        "min_coverage_value": float(coverage["coverage_ratio"].min()),
        "features_with_break_flag": [r["feature"] for r in break_rows if r["break_flag"]],
    }
    with open(ROOT / "outputs" / "logs" / "descriptive_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print("\nSaved tables 1-4 and structural break screen to", OUT_TABLES)


if __name__ == "__main__":
    main()
