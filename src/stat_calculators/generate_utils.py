import numpy as np
import torch


CHRONOS2_QUANTILE_HEADS = [
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
]
DECILE_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def init_prob_info():
    return {"neg entropy": []}


def to_1d_forecast(x):
    arr = as_numpy(x).astype(np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D univariate forecast, got shape {arr.shape}")
    return arr.astype(float).tolist()


def to_median_forecast(quantile_forecast):
    arr = as_numpy(quantile_forecast).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[-1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D median forecast, got shape {arr.shape}")
    return arr.astype(float).tolist()


def model_param_device(pipeline):
    try:
        return next(pipeline.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def model_param_dtype(pipeline):
    try:
        return next(pipeline.parameters()).dtype
    except StopIteration:
        return torch.float32


def ensure_sample_series_shape(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 3 and arr.shape[1] == 1:
        return arr[:, 0, :]
    return arr


def token_neg_entropy(probs):
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    logps = np.log(probs)
    normalized_logps = logps - np.log(np.exp(logps).sum())
    return float(np.sum(normalized_logps * np.exp(normalized_logps)))


def build_token_entropy(topk_probs):
    return {"neg entropy": [token_neg_entropy(probs) for probs in topk_probs]}


def derive_quantile_levels(sample_u_low, sample_u_high):
    heads = np.asarray(CHRONOS2_QUANTILE_HEADS, dtype=np.float64)
    low = float(sample_u_low)
    high = float(sample_u_high)
    if not (0.0 < low < high < 1.0):
        raise ValueError("sample_u_low/high must satisfy 0 < low < high < 1")
    if low < heads[0] or high > heads[-1]:
        raise ValueError(
            f"sample_u_low/high must be covered by Chronos-2 quantile heads "
            f"[{heads[0]}, {heads[-1]}]"
        )

    eps = 1e-12
    selected = heads[(heads >= low - eps) & (heads <= high + eps)]
    left = heads[heads <= low + eps]
    right = heads[heads >= high - eps]
    if left.size:
        selected = np.concatenate([[left[-1]], selected])
    if right.size:
        selected = np.concatenate([selected, [right[0]]])

    levels = sorted({round(float(v), 12) for v in selected})
    if len(levels) < 2:
        raise ValueError("Derived fewer than two quantile levels; widen sample_u range")
    return levels


def validate_decile_sample_range(sample_u_low, sample_u_high, model_name):
    min_q = DECILE_QUANTILE_LEVELS[0]
    max_q = DECILE_QUANTILE_LEVELS[-1]
    if not (min_q <= float(sample_u_low) < float(sample_u_high) <= max_q):
        raise ValueError(
            f"{model_name} uses fixed decile outputs [0.1, ..., 0.9], so "
            f"sample_u_low/high must satisfy {min_q} <= low < high <= {max_q}"
        )


def resolve_multi_step_length(multiple_cfg, default_length, max_length, model_name):
    raw_length = multiple_cfg.get("multi_step_length", None)
    if raw_length is None:
        raw_length = default_length
    length = int(raw_length)
    if length < 1 or length > int(max_length):
        raise ValueError(
            f"multiple_cfg.multi_step_length for {model_name} must satisfy "
            f"1 <= multi_step_length <= {int(max_length)}"
        )
    return length


def to_quantile_array(quantiles, model_prediction_length, num_quantiles):
    arr = as_numpy(quantiles).astype(np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D quantile forecasts, got shape {arr.shape}")
    if arr.shape == (model_prediction_length, num_quantiles):
        return arr
    if arr.shape == (num_quantiles, model_prediction_length):
        return arr.T
    raise ValueError(
        "Unexpected quantile forecast shape "
        f"{arr.shape}; expected ({model_prediction_length}, {num_quantiles}) "
        f"or ({num_quantiles}, {model_prediction_length})"
    )


def to_quantile_batch(quantiles, batch_size, prediction_length, num_quantiles):
    arr = as_numpy(quantiles).astype(np.float32)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D quantile forecasts, got shape {arr.shape}")
    if arr.shape == (batch_size, prediction_length, num_quantiles):
        return arr
    if arr.shape == (batch_size, num_quantiles, prediction_length):
        return np.transpose(arr, (0, 2, 1))
    raise ValueError(
        "Unexpected quantile forecast shape "
        f"{arr.shape}; expected ({batch_size}, {prediction_length}, {num_quantiles}) "
        f"or ({batch_size}, {num_quantiles}, {prediction_length})"
    )


def to_decile_quantile_batch(output, batch_size, prediction_length):
    return to_quantile_batch(
        output,
        batch_size=batch_size,
        prediction_length=prediction_length,
        num_quantiles=len(DECILE_QUANTILE_LEVELS),
    )


def ensure_sample_batch_shape(arr, batch_size, num_samples, prediction_length, model_name):
    arr = as_numpy(arr).astype(np.float32)
    if arr.ndim == 2 and batch_size == 1:
        arr = arr.reshape(1, *ensure_sample_series_shape(arr).shape)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected {model_name} batch output shape: {arr.shape}")
    if arr.shape[0] == batch_size and arr.shape[1] == num_samples:
        out = arr
    elif arr.shape[0] == batch_size and arr.shape[2] == num_samples:
        out = np.transpose(arr, (0, 2, 1))
    elif arr.shape[0] == num_samples and arr.shape[1] == batch_size:
        out = np.transpose(arr, (1, 0, 2))
    else:
        raise ValueError(
            f"Unexpected {model_name} batch output shape: {arr.shape}; "
            f"expected batch/sample axes ({batch_size}, {num_samples}, ...)"
        )
    return out[:, :num_samples, :prediction_length].astype(float)


def make_strictly_increasing(values):
    values = np.asarray(values, dtype=np.float64).copy()
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            values[i] = np.nextafter(values[i - 1], np.inf)
    return values


def linear_interval(sampled_u, quantile_levels=DECILE_QUANTILE_LEVELS):
    levels = np.asarray(quantile_levels, dtype=np.float64)
    upper = int(np.searchsorted(levels, sampled_u, side="right"))
    upper = min(max(upper, 1), len(levels) - 1)
    lower = upper - 1
    return lower, upper


def linear_quantile_sample(quantile_series, sampled_u, quantile_levels=DECILE_QUANTILE_LEVELS):
    lower, upper = linear_interval(sampled_u, quantile_levels)
    levels = np.asarray(quantile_levels, dtype=np.float64)
    alpha = (float(sampled_u) - levels[lower]) / (levels[upper] - levels[lower])
    quantile_series = np.asarray(quantile_series, dtype=np.float64)
    return quantile_series[:, lower] + alpha * (quantile_series[:, upper] - quantile_series[:, lower])


def linear_neg_entropy(step_quantiles, sample_u_low, sample_u_high, quantile_levels=DECILE_QUANTILE_LEVELS):
    if float(sample_u_high) <= float(sample_u_low):
        raise ValueError("sample_u_high must be greater than sample_u_low")
    levels = np.asarray(quantile_levels, dtype=np.float64)
    quantiles = make_strictly_increasing(step_quantiles)
    derivatives = np.diff(quantiles) / np.diff(levels)
    weights, valid_derivatives = [], []
    for index, derivative in enumerate(derivatives):
        left = max(float(sample_u_low), float(levels[index]))
        right = min(float(sample_u_high), float(levels[index + 1]))
        if right <= left or derivative <= 0.0 or not np.isfinite(derivative):
            continue
        weights.append(right - left)
        valid_derivatives.append(derivative)
    if not valid_derivatives:
        return 0.0
    return float(-np.sum(np.asarray(weights) * np.log(np.clip(valid_derivatives, 1e-12, None))))


def sample_from_quantiles(quantile_levels, first_step_quantiles, rng, sample_u_low, sample_u_high):
    q = make_strictly_increasing(first_step_quantiles)
    p = np.asarray(quantile_levels, dtype=np.float64)
    sampled_u = float(rng.uniform(sample_u_low, sample_u_high))
    sampled_value = float(np.interp(sampled_u, p, q))

    lower, upper = linear_interval(sampled_u, quantile_levels)
    derivative = (q[upper] - q[lower]) / (p[upper] - p[lower])
    if derivative <= 0.0 or not np.isfinite(derivative):
        density = 0.0
    else:
        density = float(1.0 / np.clip(derivative, 1e-12, None))
    return sampled_value, sampled_u, density
