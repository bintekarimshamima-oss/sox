"""Build the release-aware monthly panel from FRED, French, and GPR raw snapshots.

Three clocks are kept distinct throughout:
  - reference_month: the economic period a value describes
  - available_date: the first date the value could plausibly have been known
  - decision_date: the month-end forecast origin

FRED/BLS/Census series here are used with a conservative one-month
publication lag (recorded ALFRED vintages are not available), i.e. a value
for reference month M becomes available at end of month M+1. Market series
(SOX, VIX, rates, dollar, Nasdaq) are available same-day (end of day).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

MARKET_SERIES = ["NASDAQSOX", "VIXCLS", "DGS3MO", "DGS10", "DTWEXBGS", "NASDAQCOM"]
MONTHLY_MACRO_SERIES = [
    "IPG3344S", "CAPUTLG3344S", "PCU33443344", "IZ3344", "IY3344",
    "A34SIS", "A34HNO",
]


def read_fred_csv(series_id: str) -> pd.DataFrame:
    path = RAW / "fred" / f"{series_id}.csv"
    df = pd.read_csv(path)
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date", series_id: "value"})
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def month_end_from_daily(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    df = df.copy()
    df["ref_month"] = df["date"].dt.to_period("M")
    out = df.groupby("ref_month").agg(**{value_name: ("value", "last")}).reset_index()
    out["decision_date"] = out["ref_month"].dt.to_timestamp("M")
    return out[["decision_date", value_name]]


def monthly_with_lag(df: pd.DataFrame, value_name: str, lag_months: int = 1) -> pd.DataFrame:
    """Series is a monthly reference value; apply a conservative publication lag
    so it enters the panel only at decision_date = reference_month_end + lag."""
    df = df.copy()
    df["ref_month"] = df["date"].dt.to_period("M")
    df = df.drop_duplicates(subset="ref_month", keep="last")
    df["decision_date"] = (df["ref_month"] + lag_months).dt.to_timestamp("M")
    df = df.rename(columns={"value": value_name})
    return df[["decision_date", value_name]]


def build_market_panel() -> pd.DataFrame:
    frames = []
    for sid in MARKET_SERIES:
        raw = read_fred_csv(sid)
        frames.append(month_end_from_daily(raw, sid))
    panel = frames[0]
    for f in frames[1:]:
        panel = panel.merge(f, on="decision_date", how="outer")
    return panel.sort_values("decision_date").reset_index(drop=True)


def build_macro_panel() -> pd.DataFrame:
    frames = []
    for sid in MONTHLY_MACRO_SERIES:
        raw = read_fred_csv(sid)
        frames.append(monthly_with_lag(raw, sid, lag_months=1))
    panel = frames[0]
    for f in frames[1:]:
        panel = panel.merge(f, on="decision_date", how="outer")
    return panel.sort_values("decision_date").reset_index(drop=True)


def build_french_panel() -> pd.DataFrame:
    path = RAW / "french" / "F-F_Research_Data_5_Factors_2x3.csv"
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    # Find header row (starts with ",Mkt-RF")
    header_idx = next(i for i, l in enumerate(lines) if l.startswith(",Mkt-RF"))
    data_lines = []
    for l in lines[header_idx + 1:]:
        parts = l.split(",")
        first = parts[0].strip()
        if len(first) == 6 and first.isdigit():  # YYYYMM monthly rows only
            data_lines.append(l)
        elif first == "" or (len(first) == 4 and first.isdigit()):
            continue  # blank or annual trailer
        else:
            break
    from io import StringIO
    csv_text = ",Mkt-RF,SMB,HML,RMW,CMA,RF\n" + "\n".join(data_lines)
    df = pd.read_csv(StringIO(csv_text))
    df.columns = ["yyyymm", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    df["ref_month"] = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m").dt.to_period("M")
    # French factors published with ~1 month lag relative to reference month
    df["decision_date"] = (df["ref_month"] + 1).dt.to_timestamp("M")
    for c in ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return df[["decision_date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]]


def build_gpr_panel() -> pd.DataFrame:
    path = RAW / "gpr" / "data_gpr_export.xls"
    df = pd.read_excel(path, sheet_name="Sheet1")
    df = df[["month", "GPR", "GPRT", "GPRA"]].copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["month", "GPR"])
    df["ref_month"] = df["month"].dt.to_period("M")
    # Conservative one-month lag per protocol (real-time archive unavailable)
    df["decision_date"] = (df["ref_month"] + 1).dt.to_timestamp("M")
    df = df.rename(columns={"GPR": "gpr", "GPRT": "gpr_threats", "GPRA": "gpr_acts"})
    return df[["decision_date", "gpr", "gpr_threats", "gpr_acts"]]


def build_sox_replication_panel() -> pd.DataFrame:
    path = RAW / "SOX_data_original_replication_input.xlsx"
    df = pd.read_excel(path)
    df["ref_month"] = df["Date"].dt.to_period("M")
    monthly = df.groupby("ref_month").agg(sox_close_thesis=("SOX_Price", "last")).reset_index()
    monthly["decision_date"] = monthly["ref_month"].dt.to_timestamp("M")
    return monthly[["decision_date", "sox_close_thesis"]]


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    market = build_market_panel()
    macro = build_macro_panel()
    french = build_french_panel()
    gpr = build_gpr_panel()
    sox_thesis = build_sox_replication_panel()

    panel = market.merge(macro, on="decision_date", how="outer")
    panel = panel.merge(french, on="decision_date", how="outer")
    panel = panel.merge(gpr, on="decision_date", how="outer")
    panel = panel.merge(sox_thesis, on="decision_date", how="outer")
    panel = panel.sort_values("decision_date").reset_index(drop=True)

    study_start = pd.Timestamp("2006-01-31")
    freeze_date = pd.Timestamp("2026-09-03")
    panel = panel[(panel["decision_date"] >= study_start) & (panel["decision_date"] <= freeze_date)]
    panel = panel.reset_index(drop=True)

    panel.to_parquet(INTERIM / "monthly_panel_core.parquet", index=False)
    panel.to_csv(INTERIM / "monthly_panel_core.csv", index=False)

    report = {
        "n_months": int(len(panel)),
        "date_min": str(panel["decision_date"].min().date()),
        "date_max": str(panel["decision_date"].max().date()),
        "columns": panel.columns.tolist(),
        "missing_pct": panel.isna().mean().round(4).to_dict(),
    }
    with open(ROOT / "outputs" / "logs" / "harmonization_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Built monthly panel: {report['n_months']} months, {report['date_min']} to {report['date_max']}")
    print("Columns:", report["columns"])
    print("Missingness:", json.dumps(report["missing_pct"], indent=2))


if __name__ == "__main__":
    main()
