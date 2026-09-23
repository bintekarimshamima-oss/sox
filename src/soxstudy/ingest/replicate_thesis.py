import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data" / "raw" / "SOX_data_original_replication_input.xlsx"
OUT_DIR = ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_excel(INPUT)
    df = df.sort_values("Date").reset_index(drop=True)

    n = len(df)
    date_min, date_max = df["Date"].min(), df["Date"].max()
    price = df["SOX_Price"]

    log_ret = np.log(price / price.shift(1)).dropna()

    desc = {
        "n_rows": int(n),
        "date_min": str(date_min.date()),
        "date_max": str(date_max.date()),
        "price_mean": float(price.mean()),
        "price_sd": float(price.std()),
        "price_min": float(price.min()),
        "price_max": float(price.max()),
        "log_ret_mean_pct": float(log_ret.mean() * 100),
        "log_ret_sd_pct": float(log_ret.std() * 100),
        "log_ret_excess_kurtosis": float(log_ret.kurtosis()),
        "n_missing": int(df["SOX_Price"].isna().sum()),
        "n_duplicate_dates": int(df["Date"].duplicated().sum()),
        "n_nonpositive_prices": int((df["SOX_Price"] <= 0).sum()),
        "dates_sorted": bool(df["Date"].is_monotonic_increasing),
    }

    # Reproduce the thesis's 80:20 chronological split
    split_idx = int(round(n * 0.8))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    desc["train_n"] = int(len(train))
    desc["test_n"] = int(len(test))
    desc["train_end_date"] = str(train["Date"].max().date())
    desc["test_start_date"] = str(test["Date"].min().date())
    desc["train_max_price"] = float(train["SOX_Price"].max())
    desc["test_max_price"] = float(test["SOX_Price"].max())
    desc["test_vs_train_max_pct_increase"] = float(
        (desc["test_max_price"] / desc["train_max_price"] - 1) * 100
    )

    # Naive (previous-close) benchmark on the test split, as the thesis's mandatory baseline
    test_prices = test["SOX_Price"].reset_index(drop=True)
    naive_pred = test_prices.shift(1)
    valid = naive_pred.notna()
    errors = test_prices[valid] - naive_pred[valid]
    desc["naive_benchmark_rmse"] = float(np.sqrt((errors ** 2).mean()))
    desc["naive_benchmark_mae"] = float(errors.abs().mean())

    audit_table = pd.DataFrame([
        {"audit_item": "Structure", "observed_evidence": "One worksheet; two columns: Date and SOX_Price"},
        {"audit_item": "Rows and period", "observed_evidence": f"{desc['n_rows']} trading days; {desc['date_min']} to {desc['date_max']}"},
        {"audit_item": "Integrity", "observed_evidence": f"Missing={desc['n_missing']}, duplicate dates={desc['n_duplicate_dates']}, non-positive={desc['n_nonpositive_prices']}, sorted={desc['dates_sorted']}"},
        {"audit_item": "Price distribution", "observed_evidence": f"Mean {desc['price_mean']:.2f}; SD {desc['price_sd']:.2f}; range {desc['price_min']:.2f}-{desc['price_max']:.2f}"},
        {"audit_item": "Daily log returns", "observed_evidence": f"Mean {desc['log_ret_mean_pct']:.3f}%; SD {desc['log_ret_sd_pct']:.3f}%; excess kurtosis {desc['log_ret_excess_kurtosis']:.2f}"},
        {"audit_item": "Fixed split (80:20)", "observed_evidence": f"Train: {desc['train_n']} through {desc['train_end_date']}; test: {desc['test_n']} from {desc['test_start_date']}"},
        {"audit_item": "Range shift", "observed_evidence": f"Test maximum is {desc['test_vs_train_max_pct_increase']:.1f}% above training maximum"},
        {"audit_item": "Naive benchmark (test)", "observed_evidence": f"RMSE={desc['naive_benchmark_rmse']:.2f}; MAE={desc['naive_benchmark_mae']:.2f}"},
    ])

    audit_table.to_csv(OUT_DIR / "table0_thesis_replication_audit.csv", index=False)
    with open(ROOT / "outputs" / "logs" / "replication_report.json", "w", encoding="utf-8") as f:
        json.dump(desc, f, indent=2)

    print(audit_table.to_string(index=False))
    print("\nSaved:", OUT_DIR / "table0_thesis_replication_audit.csv")


if __name__ == "__main__":
    main()
