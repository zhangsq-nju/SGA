from datasets import load_dataset
from gluonts.transform import LastValueImputation
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.seasonal_error import compute_seasonal_error

ETT_SOURCE_BASE_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
ETT_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def _safe_float_list(values):
    # 检查是否有非数值值，如果有则报错
    for v in values:
        if not isinstance(v, (int, float)):
            raise ValueError(f"Non-numeric value found: {v} in series {values}")
    return values


def impute_contexts_last_value(data):
    imputer = LastValueImputation()
    return data | {
        "contexts": [
            imputer(np.asarray(context, dtype=np.float64)).astype(float).tolist()
            for context in data["contexts"]
        ]
    }


def _period_compatible_freq(freq):
    if freq is None:
        return None

    freq = str(freq)
    upper = freq.upper()
    if upper in {"ME", "MS", "M"}:
        return "M"
    if upper.startswith(("QE", "QS")) or upper == "Q":
        return "Q"
    if upper.startswith(("YE", "YS")) or upper == "Y":
        return "Y"
    return freq


def _infer_dataset_freq(data):
    if len(data) == 0 or "timestamp" not in data[0]:
        return None

    timestamps = data[0]["timestamp"]
    if timestamps is None or len(timestamps) == 0:
        return None

    return _period_compatible_freq(pd.DatetimeIndex(timestamps).to_period()[0].freqstr)


def _window_series(series, prediction_length, context_length):
    split_idx = len(series) - prediction_length
    if split_idx <= 0:
        return None, None

    start_idx = max(0, split_idx - context_length)
    return series[start_idx:split_idx], series[split_idx : split_idx + prediction_length]


def _is_numeric_series(values):
    if not isinstance(values, list) or len(values) == 0:
        return False
    return all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in values)


def _is_ett_dataset(cfg):
    return (
        cfg.get("dataset_repo", None) == "autogluon/chronos_datasets_extra"
        and cfg.get("dataset_name", None) in {"ETTh", "ETTm"}
    )


def _ett_freq(dataset_name):
    if dataset_name == "ETTh":
        return "h"
    if dataset_name == "ETTm":
        return "15min"
    raise ValueError(f"Unsupported ETT dataset_name={dataset_name}")


def _ett_csv_source(cfg, csv_name):
    source_dir = cfg.get("ett_source_dir", None)
    if source_dir is not None:
        return Path(source_dir).expanduser() / csv_name
    source_base_url = cfg.get("ett_source_base_url", ETT_SOURCE_BASE_URL)
    return f"{source_base_url.rstrip('/')}/{csv_name}"


def _load_ett_records(cfg):
    dataset_name = cfg.dataset_name
    records = []
    for region in (1, 2):
        csv_name = f"{dataset_name}{region}.csv"
        df = pd.read_csv(_ett_csv_source(cfg, csv_name))
        date_col = "date" if "date" in df.columns else "timestamp" if "timestamp" in df.columns else None
        record = {"id": f"{dataset_name}{region}"}
        if date_col is not None:
            record["timestamp"] = pd.to_datetime(df[date_col]).tolist()

        missing_cols = [col for col in ETT_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{csv_name} is missing ETT columns: {missing_cols}")
        for col in ETT_COLUMNS:
            record[col] = df[col].astype(float).tolist()
        records.append(record)
    return records


def _target_columns(item, cfg):
    if "target_cols" in cfg and cfg.target_cols is not None:
        return list(cfg.target_cols)
    if cfg.target_col != "all":
        return [cfg.target_col]
    if not _is_ett_dataset(cfg):
        raise ValueError("target_col=all is only supported for ETT datasets ETTh/ETTm.")

    return [key for key in ETT_COLUMNS if key in item and _is_numeric_series(item[key])]


def _build_data_from_records(records, cfg, freq):
    ids = []
    contexts = []
    targets = []
    seasonal_errors = []

    prediction_length = int(cfg.prediction_length)
    context_length = int(cfg.context_length)
    max_items = cfg.max_items
    subset = cfg.get("subset", None)

    if subset is not None and not any("subset" in item for item in records):
        raise ValueError("Subset filtering is enabled but 'subset' column is missing in the dataset.")

    for idx, item in enumerate(records):
        if max_items is not None and len(ids) >= int(max_items):
            break

        if subset is not None and item["subset"] != subset:
            continue

        for target_col in _target_columns(item, cfg):
            if max_items is not None and len(ids) >= int(max_items):
                break
            if target_col not in item:
                continue

            series = _safe_float_list(item[target_col])
            context, target = _window_series(series, prediction_length, context_length)
            if context is None:
                continue

            base_id = item.get("id", item.get("item_id", idx))
            ids.append(f"{base_id}:{target_col}" if cfg.target_col == "all" else base_id)
            contexts.append(context)
            targets.append(target)
            seasonal_errors.append(compute_seasonal_error(context, freq=freq))

    if not ids:
        raise ValueError("No valid series found. Please check dataset config and sequence lengths.")

    return {"ids": ids, "contexts": contexts, "targets": targets, "freq": freq, "seasonal_errors": seasonal_errors}


def load_data(cfg):
    print(
        f"Loading dataset: repo={cfg.dataset_repo}, name={cfg.dataset_name}, split={cfg.split}, target_col={cfg.target_col}"
    )

    if _is_ett_dataset(cfg):
        data = _load_ett_records(cfg)
        freq = _period_compatible_freq(cfg.get("freq", None)) or _ett_freq(cfg.dataset_name)
    else:
        data = load_dataset(
            cfg.dataset_repo,
            cfg.dataset_name,
            split=cfg.split,
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        )
        freq = _period_compatible_freq(cfg.get("freq", None)) or _infer_dataset_freq(data) or "D"
    out = _build_data_from_records(data, cfg, freq)

    return out
