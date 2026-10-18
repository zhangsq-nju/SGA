import numpy as np


class _SimpleTestData:
    def __init__(self, inputs, labels):
        self.input = inputs
        self.label = labels


def _period_start(start, freq):
    import pandas as pd

    try:
        return pd.Period(start, freq=freq)
    except Exception as exc:
        raise ValueError(f"Invalid MASE start/freq: start={start}, freq={freq}") from exc


def _calculate_mase_errors(response_data, seasonality=None, batch_size=5000):
    from gluonts.ev.metrics import MASE
    from gluonts.model.evaluation import evaluate_forecasts
    from gluonts.model.forecast import QuantileForecast

    inputs = []
    labels = []
    forecasts = []
    id_by_item_id = {}

    for idx, item in enumerate(response_data):
        series_id = item["id"]
        if "context" not in item:
            raise ValueError(
                "MASE requires 'context' in response_data.jsonl. "
                "Please rerun inference_once to regenerate response_data.jsonl."
            )

        context = np.asarray(item["context"], dtype=np.float32)
        y_true = np.asarray(item["y_true"], dtype=np.float32)
        y_pred = np.asarray(item["y_pred"], dtype=np.float32)

        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch for id={series_id}: y_true{y_true.shape} vs y_pred{y_pred.shape}")
        if context.ndim != 1 or y_true.ndim != 1:
            raise ValueError(f"MASE expects 1D context/y_true for id={series_id}")
        if len(context) == 0:
            raise ValueError(f"MASE requires non-empty context for id={series_id}")

        freq = item.get("freq", "D")
        start = _period_start("2000-01-01", freq)
        item_id = str(series_id)
        if item_id in id_by_item_id:
            item_id = f"{item_id}#{idx}"
        id_by_item_id[item_id] = series_id

        inputs.append(
            {
                "target": context,
                "start": start,
                "item_id": item_id,
            }
        )
        labels.append(
            {
                "target": y_true,
                "start": start + len(context),
                "item_id": item_id,
            }
        )
        forecasts.append(
            QuantileForecast(
                forecast_arrays=y_pred.reshape(1, -1),
                start_date=start + len(context),
                forecast_keys=["0.5"],
                item_id=item_id,
            )
        )

    mase_df = evaluate_forecasts(
        forecasts,
        test_data=_SimpleTestData(inputs, labels),
        metrics=[MASE()],
        axis=1,
        batch_size=batch_size,
        seasonality=seasonality,
    )

    metric_col = "MASE[0.5]"
    results = {}
    for index, row in mase_df.iterrows():
        item_id = index[0] if isinstance(index, tuple) else index
        err = float(row[metric_col])
        if np.isfinite(err):
            results[id_by_item_id[str(item_id)]] = err

    return results


def calculate_series_error(response_data, error_metric="mae", seasonality=None):
    error_metric = error_metric.lower()
    if error_metric not in {"mae", "rmse", "mase"}:
        raise ValueError(f"Unsupported error_metric: {error_metric}")

    if error_metric == "mase":
        return _calculate_mase_errors(response_data, seasonality=seasonality)

    results = {}
    for item in response_data:
        series_id = item["id"]
        y_true = np.asarray(item["y_true"], dtype=np.float64)
        y_pred = np.asarray(item["y_pred"], dtype=np.float64)

        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch for id={series_id}: y_true{y_true.shape} vs y_pred{y_pred.shape}")

        if error_metric == "mae":
            err = np.abs(y_true - y_pred).mean()
        elif error_metric == "rmse":
            err = np.sqrt(np.square(y_true - y_pred).mean())
        else:
            raise ValueError(f"Unsupported error_metric: {error_metric}")

        if np.isnan(err).any():
            # print(f"Warning: Computed error is NaN for id={series_id}. y_true={y_true}, y_pred={y_pred}")
            continue

        results[series_id] = float(err)

    return results
