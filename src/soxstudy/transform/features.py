"""Construct stationary predictor features and forecast outcomes on the core
monthly panel (FRED macro/market + French factors + GPR). Trade-flow (Comtrade)
and corporate (SEC XBRL) blocks are merged in separately once those ingestion
jobs finish (see merge_extended_blocks.py), following the same
origin-only-refit discipline.

All transforms here are causal (use only current and past values). Factor
loadings (PCA) are refit inside each outer training window at model-fitting
time, NOT here — this module only prepares stationary candidate features.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
RAW_FRED = ROOT / "data" / "raw" / "fred"


def pct_change_safe(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.pct_change(periods=periods)


def yoy_growth(s: pd.Series) -> pd.Series:
    return s.pct_change(periods=12)


def compute_daily_realized_volatility() -> pd.DataFrame:
    """True monthly realized volatility from actual daily SOX log returns
    (annualized sqrt(21 * sum(daily_return^2) within the month)), replacing
    the earlier placeholder that mistakenly computed a 2-observation rolling
    std of MONTHLY returns and mislabeled it as realized volatility."""
    daily = pd.read_csv(RAW_FRED / "NASDAQSOX.csv")
    daily = daily.rename(columns={"observation_date": "date", "NASDAQSOX": "price"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily["price"] = pd.to_numeric(daily["price"], errors="coerce")
    daily = daily.dropna(subset=["price"]).sort_values("date")
    daily["log_ret"] = np.log(daily["price"]).diff()
    daily["ref_month"] = daily["date"].dt.to_period("M")

    monthly_rv = daily.groupby("ref_month").agg(
        sox_realized_vol=("log_ret", lambda x: np.sqrt(np.sum(x.dropna() ** 2))),
        n_trading_days=("log_ret", lambda x: x.notna().sum()),
    ).reset_index()
    monthly_rv["decision_date"] = monthly_rv["ref_month"].dt.to_timestamp("M")
    return monthly_rv[["decision_date", "sox_realized_vol", "n_trading_days"]]


def build_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy().sort_values("decision_date").reset_index(drop=True)
    log_price = np.log(df["NASDAQSOX"])

    for h in [1, 3, 6]:
        fwd_log_ret = log_price.shift(-h) - log_price
        rf_col = df["rf"].fillna(0.0)
        compounded_rf = rf_col.rolling(window=h, min_periods=h).sum().shift(-h + 1)
        df[f"sox_excess_ret_h{h}"] = 100 * fwd_log_ret - 100 * compounded_rf

    rv_daily = compute_daily_realized_volatility()
    df = df.merge(rv_daily, on="decision_date", how="left")
    df["sox_rv_next"] = df["sox_realized_vol"].shift(-1)

    expanding_q20 = df["sox_excess_ret_h3"].expanding(min_periods=24).quantile(0.20)
    df["sox_downside_h3"] = (df["sox_excess_ret_h3"] < expanding_q20).astype(float)
    df.loc[df["sox_excess_ret_h3"].isna(), "sox_downside_h3"] = np.nan

    return df


def build_predictor_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["sox_mom_1m"] = np.log(df["NASDAQSOX"]).diff(1)
    df["sox_mom_3m"] = np.log(df["NASDAQSOX"]).diff(3)
    df["sox_mom_12m"] = np.log(df["NASDAQSOX"]).diff(12)
    df["sox_realized_vol_1m"] = df["sox_mom_1m"].rolling(6).std()

    df["nasdaq_ret_1m"] = np.log(df["NASDAQCOM"]).diff(1)
    df["vix_level"] = df["VIXCLS"]
    df["vix_chg_1m"] = df["VIXCLS"].diff(1)
    df["term_spread"] = df["DGS10"] - df["DGS3MO"]
    df["dollar_chg_1m"] = np.log(df["DTWEXBGS"]).diff(1)

    df["ip_growth_yoy"] = yoy_growth(df["IPG3344S"])
    df["ip_growth_1m"] = pct_change_safe(df["IPG3344S"])
    df["cap_util_level"] = df["CAPUTLG3344S"]
    df["cap_util_chg_3m"] = df["CAPUTLG3344S"].diff(3)

    df["ppi_growth_yoy"] = yoy_growth(df["PCU33443344"])
    df["import_price_growth_yoy"] = yoy_growth(df["IZ3344"])
    df["export_price_growth_yoy"] = yoy_growth(df["IY3344"])
    df["price_pressure_dispersion"] = (
        df[["import_price_growth_yoy", "export_price_growth_yoy", "ppi_growth_yoy"]].std(axis=1)
    )

    df["electronics_inventory_growth_yoy"] = yoy_growth(df["A34SIS"])
    df["electronics_orders_growth_yoy"] = yoy_growth(df["A34HNO"])

    df["gpr_level"] = df["gpr"]
    df["gpr_chg_3m"] = df["gpr"].diff(3)

    df["mkt_rf"] = df["mkt_rf"]
    df["smb"] = df["smb"]
    df["hml"] = df["hml"]

    # HAR-RV style lagged realized-volatility features (Corsi, 2009): daily
    # (current month), weekly-equivalent (3-month average), monthly-equivalent
    # (12-month average) components, all using only current/past information.
    df["rv_lag1"] = df["sox_realized_vol"]
    df["rv_lag_avg3"] = df["sox_realized_vol"].rolling(3).mean()
    df["rv_lag_avg12"] = df["sox_realized_vol"].rolling(12).mean()

    return df


FEATURE_COLUMNS = [
    "sox_mom_1m", "sox_mom_3m", "sox_mom_12m", "sox_realized_vol_1m",
    "nasdaq_ret_1m", "vix_level", "vix_chg_1m", "term_spread", "dollar_chg_1m",
    "ip_growth_yoy", "ip_growth_1m", "cap_util_level", "cap_util_chg_3m",
    "ppi_growth_yoy", "import_price_growth_yoy", "export_price_growth_yoy",
    "price_pressure_dispersion", "electronics_inventory_growth_yoy",
    "electronics_orders_growth_yoy", "gpr_level", "gpr_chg_3m",
    "mkt_rf", "smb", "hml", "rv_lag1", "rv_lag_avg3", "rv_lag_avg12",
]

OUTCOME_COLUMNS = [
    "sox_excess_ret_h1", "sox_excess_ret_h3", "sox_excess_ret_h6",
    "sox_rv_next", "sox_downside_h3", "sox_realized_vol", "n_trading_days",
]


def main():
    panel = pd.read_parquet(INTERIM / "monthly_panel_core.parquet")
    panel = build_outcomes(panel)
    panel = build_predictor_features(panel)

    keep = ["decision_date"] + FEATURE_COLUMNS + OUTCOME_COLUMNS + ["NASDAQSOX"]
    features_monthly = panel[keep].copy()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    features_monthly.to_parquet(PROCESSED / "features_monthly_core.parquet", index=False)
    features_monthly.to_csv(PROCESSED / "features_monthly_core.csv", index=False)

    report = {
        "n_rows": int(len(features_monthly)),
        "date_min": str(features_monthly["decision_date"].min().date()),
        "date_max": str(features_monthly["decision_date"].max().date()),
        "feature_columns": FEATURE_COLUMNS,
        "outcome_columns": OUTCOME_COLUMNS,
        "missing_pct": features_monthly.isna().mean().round(4).to_dict(),
    }
    with open(ROOT / "outputs" / "logs" / "feature_construction_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Feature panel: {report['n_rows']} rows, {report['date_min']}..{report['date_max']}")
    print(f"{len(FEATURE_COLUMNS)} predictor features, {len(OUTCOME_COLUMNS)} outcomes.")


if __name__ == "__main__":
    main()
