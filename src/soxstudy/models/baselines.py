import numpy as np
import pandas as pd


def historical_mean_forecast(y_train: np.ndarray) -> float:
    return float(np.nanmean(y_train))


def zero_return_forecast() -> float:
    return 0.0


def ar1_forecast(y_train: pd.Series) -> float:
    """One-step-ahead AR(1)-implied unconditional-mean forecast fit on train only."""
    y = y_train.dropna().values
    if len(y) < 10:
        return float(np.nanmean(y_train)) if len(y_train.dropna()) else 0.0
    y_t = y[1:]
    y_tm1 = y[:-1]
    x = np.column_stack([np.ones(len(y_tm1)), y_tm1])
    try:
        beta, *_ = np.linalg.lstsq(x, y_t, rcond=None)
        c, phi = beta
        forecast = c + phi * y[-1]
        return float(forecast)
    except np.linalg.LinAlgError:
        return float(np.nanmean(y_train))
