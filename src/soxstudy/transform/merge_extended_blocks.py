"""Merge the UN Comtrade trade-flow block and SEC XBRL corporate-fundamentals
block into the core monthly panel, once those ingestion jobs have produced
data/interim/comtrade_trade_flows.parquet and data/interim/sec_xbrl_facts.parquet.

Both blocks are release-aware:
  - Comtrade: primary forecast uses a conservative 2-month lag from the trade
    reference month (config/protocol.yml release_lags.comtrade_primary_lag_months).
  - SEC XBRL: each fact enters on the first decision date after its `filed`
    date (not the fiscal period end), per the filing-date rule in the proposal.

This script is idempotent and safe to run before the extended sources exist:
it checks for the interim files and, if absent, logs a clear skip message
rather than failing, so the core pipeline is never blocked on it.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

COMTRADE_PATH = INTERIM / "comtrade_trade_flows.parquet"
SEC_PATH = INTERIM / "sec_xbrl_facts.parquet"

COMTRADE_LAG_MONTHS = 2


def build_comtrade_features() -> pd.DataFrame | None:
    if not COMTRADE_PATH.exists():
        print(f"SKIP: {COMTRADE_PATH} not found yet (Comtrade ingestion still running or not started).")
        return None

    trade = pd.read_parquet(COMTRADE_PATH)
    trade["period"] = trade["period"].astype(str)
    trade["ref_month"] = pd.to_datetime(trade["period"], format="%Y%m").dt.to_period("M")

    ic = trade[trade["hs_family"].astype(str).str.contains("8542")]
    ic_m = ic[ic["flow"] == "M"]

    demand = ic_m.groupby("ref_month").agg(
        trade_value_usd_sum=("trade_value_usd", "sum"),
        reporter_count=("reporter_code", "nunique"),
    ).reset_index()
    demand["demand_growth_yoy"] = demand["trade_value_usd_sum"].pct_change(12)
    demand["demand_growth_1m"] = demand["trade_value_usd_sum"].pct_change(1)

    equip = trade[trade["hs_family"].astype(str).str.contains("8486")]
    equip_m = equip[equip["flow"] == "M"]
    capacity_inv = equip_m.groupby("ref_month").agg(
        equip_trade_value_usd_sum=("trade_value_usd", "sum"),
    ).reset_index()
    capacity_inv["equip_growth_yoy"] = capacity_inv["equip_trade_value_usd_sum"].pct_change(12)

    merged = demand.merge(capacity_inv, on="ref_month", how="outer").sort_values("ref_month")
    merged["decision_date"] = (merged["ref_month"] + COMTRADE_LAG_MONTHS).dt.to_timestamp("M")
    return merged[["decision_date", "demand_growth_yoy", "demand_growth_1m", "equip_growth_yoy", "reporter_count"]]


def build_sec_features() -> pd.DataFrame | None:
    if not SEC_PATH.exists():
        print(f"SKIP: {SEC_PATH} not found yet (SEC XBRL ingestion still running or not started).")
        return None

    facts = pd.read_parquet(SEC_PATH)
    facts["filed"] = pd.to_datetime(facts["filed"])
    facts["decision_date"] = facts["filed"].dt.to_period("M").dt.to_timestamp("M")

    pivot = facts.pivot_table(
        index=["cik", "decision_date"], columns="concept", values="val", aggfunc="last"
    ).reset_index()

    revenue_col = "RevenueFromContractWithCustomerExcludingAssessedTax"
    if revenue_col not in pivot.columns:
        revenue_col = "SalesRevenueNet"

    if revenue_col in pivot.columns and "InventoryNet" in pivot.columns:
        pivot["inventory_intensity"] = pivot["InventoryNet"] / pivot[revenue_col].replace(0, np.nan)
    if revenue_col in pivot.columns and "ResearchAndDevelopmentExpense" in pivot.columns:
        pivot["rd_intensity"] = pivot["ResearchAndDevelopmentExpense"] / pivot[revenue_col].replace(0, np.nan)
    if revenue_col in pivot.columns and "CostOfRevenue" in pivot.columns:
        pivot["gross_margin"] = (pivot[revenue_col] - pivot["CostOfRevenue"]) / pivot[revenue_col].replace(0, np.nan)

    agg = pivot.groupby("decision_date").agg(
        median_inventory_intensity=("inventory_intensity", "median") if "inventory_intensity" in pivot.columns else ("cik", "count"),
        median_rd_intensity=("rd_intensity", "median") if "rd_intensity" in pivot.columns else ("cik", "count"),
        median_gross_margin=("gross_margin", "median") if "gross_margin" in pivot.columns else ("cik", "count"),
        n_firms_reporting=("cik", "nunique"),
    ).reset_index()

    return agg


def main():
    core = pd.read_parquet(PROCESSED / "features_monthly_core.parquet")

    comtrade_feat = build_comtrade_features()
    sec_feat = build_sec_features()

    extended = core.copy()
    merged_blocks = []
    if comtrade_feat is not None:
        extended = extended.merge(comtrade_feat, on="decision_date", how="left")
        merged_blocks.append("comtrade")
    if sec_feat is not None:
        extended = extended.merge(sec_feat, on="decision_date", how="left")
        merged_blocks.append("sec_xbrl")

    if not merged_blocks:
        print("No extended blocks available yet. Core panel unchanged; rerun this script once "
              "data/interim/comtrade_trade_flows.parquet and/or sec_xbrl_facts.parquet exist.")
        return

    extended.to_parquet(PROCESSED / "features_monthly_extended.parquet", index=False)
    extended.to_csv(PROCESSED / "features_monthly_extended.csv", index=False)

    report = {"merged_blocks": merged_blocks, "n_rows": int(len(extended)), "n_cols": int(len(extended.columns))}
    with open(ROOT / "outputs" / "logs" / "extended_merge_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
