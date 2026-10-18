import numpy as np


def compute_seasonal_error(values, freq="D", seasonality=None, eps=1e-8):
    from gluonts.ev.ts_stats import seasonal_error
    from gluonts.time_feature.seasonality import get_seasonality

    arr = np.ma.masked_invalid(np.asarray(values, dtype=np.float32))
    if arr.ndim != 1:
        raise ValueError(f"seasonal_error expects 1D values, got shape={arr.shape}")
    if len(arr) == 0:
        raise ValueError("seasonal_error requires non-empty values")

    if seasonality is None:
        seasonality = get_seasonality(str(freq))

    seasonality = int(seasonality)
    if seasonality <= 0:
        raise ValueError(f"seasonality must be positive, got {seasonality}")
    if seasonality > len(arr):
        seasonality = 1

    err = seasonal_error(arr, seasonality=seasonality, time_axis=-1)
    if np.ma.is_masked(err):
        err = err.filled(np.nan)
    err = float(np.asarray(err).reshape(-1)[0])
    return err if np.isfinite(err) else float(eps)
