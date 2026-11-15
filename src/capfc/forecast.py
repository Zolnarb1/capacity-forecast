"""Day-ahead forecasting of server metrics (capacity planning).

For each series: train on everything except the last day, forecast the last day
(288 points, 5-minute steps), compare with what really happened.

Models:
  naive            tomorrow = the last observed value, flat
  seasonal naive   tomorrow = today, same time of day
  holt-winters     exponential smoothing with additive daily seasonality (statsmodels)
  gradient boosting  features known a day in advance: value 1 and 2 days earlier,
                   time of day. Direct forecast, no recursion.

Metric: MASE (mean absolute scaled error). MAE divided by the MAE of seasonal naive
on the training data, so series with different units (CPU %, bytes) can be averaged.
Below 1 = better than "same as yesterday".
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

DAY = 288


def _features(s: pd.Series) -> pd.DataFrame:
    minutes = s.index.hour * 60 + s.index.minute
    return pd.DataFrame({
        "lag_1d": s.shift(DAY), "lag_2d": s.shift(2 * DAY),
        "tod_sin": np.sin(2 * np.pi * minutes / 1440), "tod_cos": np.cos(2 * np.pi * minutes / 1440),
    }, index=s.index)


def forecast_last_day(s: pd.Series) -> dict[str, np.ndarray]:
    s = s.asfreq("5min").interpolate(limit_direction="both")
    train, test = s.iloc[:-DAY], s.iloc[-DAY:]
    out = {"naive": np.repeat(train.iloc[-1], DAY),
           "seasonal naive": train.iloc[-DAY:].to_numpy()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hw = ExponentialSmoothing(train, trend=None, seasonal="add", seasonal_periods=DAY,
                                  initialization_method="estimated").fit()
        out["holt-winters"] = hw.forecast(DAY).to_numpy()
    f = _features(s)
    fit = f.iloc[2 * DAY:-DAY]
    gbm = HistGradientBoostingRegressor(max_iter=200, random_state=0)
    gbm.fit(fit, train.iloc[2 * DAY:])
    out["gradient boosting"] = gbm.predict(f.iloc[-DAY:])
    out["_actual"] = test.to_numpy()
    out["_scale"] = float(np.mean(np.abs(train.iloc[DAY:].to_numpy() - train.iloc[:-DAY].to_numpy()))) or 1.0
    out["_index"] = test.index
    return out


def mase(pred: np.ndarray, actual: np.ndarray, scale: float) -> float:
    return float(np.mean(np.abs(pred - actual)) / scale)
