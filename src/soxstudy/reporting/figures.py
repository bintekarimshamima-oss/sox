import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"
PRED_DIR = ROOT / "outputs" / "predictions"
OUT_FIGS = ROOT / "outputs" / "figures"

REGIME_BANDS = [
    ("2008-09-01", "2009-06-30", "GFC"),
    ("2020-02-01", "2020-06-30", "COVID shock"),
    ("2020-07-01", "2021-12-31", "Chip shortage"),
    ("2022-01-01", "2023-06-30", "Inventory correction"),
    ("2023-07-01", "2026-08-31", "AI acceleration"),
]


def add_regime_bands(ax):
    for start, end, label in REGIME_BANDS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.08, color="tab:red")


def fig1_sox_series():
    df = pd.read_parquet(PROCESSED / "features_monthly_core.parquet")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df["decision_date"], df["NASDAQSOX"], color="tab:blue", linewidth=1.2)
    add_regime_bands(ax)
    ax.set_title("PHLX Semiconductor Index (SOX), month-end level, 2006-2026")
    ax.set_ylabel("SOX index level")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(OUT_FIGS / "fig1_sox_series_with_regimes.png", dpi=150)
    plt.close(fig)


def fig2_factor_panel():
    df = pd.read_parquet(PROCESSED / "features_monthly_core.parquet")
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(df["decision_date"], df["ip_growth_yoy"] * 100, color="tab:green")
    axes[0].set_ylabel("IP growth YoY (%)")
    axes[0].set_title("Industrial production, capacity utilization, and GPR over time")
    axes[1].plot(df["decision_date"], df["cap_util_level"], color="tab:orange")
    axes[1].set_ylabel("Capacity utilization (%)")
    axes[2].plot(df["decision_date"], df["gpr_level"], color="tab:red")
    axes[2].set_ylabel("GPR index")
    axes[2].set_xlabel("Date")
    for ax in axes:
        add_regime_bands(ax)
    fig.tight_layout()
    fig.savefig(OUT_FIGS / "fig2_macro_factor_panel.png", dpi=150)
    plt.close(fig)


def fig3_cumulative_loss_diff():
    preds = pd.read_parquet(PRED_DIR / "backtest_predictions.parquet")
    h3 = preds[preds["target"] == "sox_excess_ret_h3"].copy()
    h3["decision_date"] = pd.to_datetime(h3["decision_date"])
    h3 = h3.sort_values("decision_date")
    loss_bench = (h3["y_true"] - h3["pred_B0_historical_mean"]) ** 2
    loss_enet = (h3["y_true"] - h3["pred_M2_elastic_net_fundamentals"]) ** 2
    cum_diff = (loss_enet - loss_bench).cumsum()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(h3["decision_date"], cum_diff, color="tab:purple")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Cumulative OOS loss difference: fundamentals model vs. historical-mean benchmark\n(h=3 excess return; below zero favors fundamentals)")
    ax.set_ylabel("Cumulative squared-error difference")
    ax.set_xlabel("Decision date")
    fig.tight_layout()
    fig.savefig(OUT_FIGS / "fig6_cumulative_oos_loss_difference.png", dpi=150)
    plt.close(fig)


def fig4_rolling_correlation():
    df = pd.read_parquet(PROCESSED / "features_monthly_core.parquet")
    df = df.sort_values("decision_date")
    roll_corr = df["ip_growth_yoy"].rolling(36).corr(df["sox_excess_ret_h3"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["decision_date"], roll_corr, color="teal")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    add_regime_bands(ax)
    ax.set_title("Rolling 36-month correlation: industrial-production growth vs. forward 3-month SOX excess return")
    ax.set_ylabel("Rolling correlation")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(OUT_FIGS / "fig4_rolling_correlation_stability.png", dpi=150)
    plt.close(fig)


def main():
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    fig1_sox_series()
    fig2_factor_panel()
    fig3_cumulative_loss_diff()
    fig4_rolling_correlation()
    print("Figures saved to", OUT_FIGS)


if __name__ == "__main__":
    main()
