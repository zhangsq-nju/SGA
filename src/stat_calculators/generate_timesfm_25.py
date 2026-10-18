import numpy as np
from dataclasses import replace

from src.stat_calculators.generate_utils import (
    DECILE_QUANTILE_LEVELS,
    init_prob_info,
    linear_neg_entropy,
    linear_quantile_sample,
    resolve_multi_step_length,
    validate_decile_sample_range,
)


TIMESFM25_MEDIAN_COLUMN = 5  # full forecast columns are: mean, q10, ..., q90.


def _compile_for_sampling(pipeline, model_prediction_length):
    forecast_config = getattr(pipeline, "forecast_config", None)
    if forecast_config is None:
        return

    desired_horizon = int(model_prediction_length)
    if getattr(pipeline, "_sga_requested_max_horizon", None) == desired_horizon:
        return

    pipeline.compile(
        replace(
            forecast_config,
            max_horizon=desired_horizon,
        )
    )
    pipeline._sga_requested_max_horizon = desired_horizon


def _to_full_forecast(full_forecast, prediction_length):
    arr = np.asarray(full_forecast, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D TimesFM forecast, got shape {arr.shape}")
    if arr.shape[0] < prediction_length or arr.shape[1] != 10:
        raise ValueError(
            f"Unexpected TimesFM forecast shape {arr.shape}; expected at least "
            f"({prediction_length}, 10)"
        )
    return arr[:prediction_length]


def _to_full_forecast_batch(full_forecast, batch_size, prediction_length):
    arr = np.asarray(full_forecast, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0, :, :]
    if arr.ndim == 2 and batch_size == 1:
        arr = arr.reshape(1, *arr.shape)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D TimesFM forecast batch, got shape {arr.shape}")
    if arr.shape[0] < batch_size or arr.shape[1] < prediction_length or arr.shape[2] != 10:
        raise ValueError(
            f"Unexpected TimesFM forecast batch shape {arr.shape}; expected "
            f"(>= {batch_size}, >= {prediction_length}, 10)"
        )
    return arr[:batch_size, :prediction_length, :]


def _predict_full_forecast(pipeline, context, prediction_length):
    _, full_forecast = pipeline.forecast(
        horizon=prediction_length,
        inputs=[np.asarray(context, dtype=np.float32)],
    )
    return _to_full_forecast(full_forecast, prediction_length)


def _predict_full_forecast_batch(pipeline, contexts, prediction_length):
    _, full_forecast = pipeline.forecast(
        horizon=prediction_length,
        inputs=[np.asarray(context, dtype=np.float32) for context in contexts],
    )
    return _to_full_forecast_batch(full_forecast, len(contexts), prediction_length)


def inference_once(cfg, pipeline, context):
    forecast = _predict_full_forecast(
        pipeline=pipeline,
        context=context.detach().cpu().numpy().astype(np.float32),
        prediction_length=cfg.prediction_length,
    )
    return forecast[:, TIMESFM25_MEDIAN_COLUMN].astype(float).tolist()


def inference_once_batch(cfg, pipeline, contexts):
    forecasts = _predict_full_forecast_batch(
        pipeline=pipeline,
        contexts=[context.detach().cpu().numpy().astype(np.float32) for context in contexts],
        prediction_length=cfg.prediction_length,
    )
    return forecasts[:, :, TIMESFM25_MEDIAN_COLUMN].astype(float).tolist()


def _multi_step_autoregressive_sample(
    pipeline,
    context,
    prediction_length,
    multi_step_length,
    rng,
    sample_u_low,
    sample_u_high,
    first_full_forecast=None,
):
    sampled_values = []
    prob_info = init_prob_info()
    rolling_context = context.detach().cpu().numpy().astype(np.float32)
    is_first_block = True

    while len(sampled_values) < prediction_length:
        if is_first_block and first_full_forecast is not None:
            full_forecast = first_full_forecast
        else:
            full_forecast = _predict_full_forecast(
                pipeline=pipeline,
                context=rolling_context,
                prediction_length=multi_step_length,
            )
        is_first_block = False

        use_len = min(multi_step_length, prediction_length - len(sampled_values))
        quantile_series = full_forecast[:use_len, 1:]
        sampled_u = float(rng.uniform(sample_u_low, sample_u_high))
        patch_values = linear_quantile_sample(quantile_series, sampled_u)

        for value, step_quantiles in zip(patch_values, quantile_series):
            stats = linear_neg_entropy(step_quantiles=step_quantiles, sample_u_low=sample_u_low, sample_u_high=sample_u_high)
            sampled_values.append(float(value))
            prob_info["neg entropy"].append(stats)

        rolling_context = np.concatenate([rolling_context, np.asarray(patch_values, dtype=np.float32)])

    return sampled_values, prob_info


def _sample_quantile_series(quantile_series, rng, sample_u_low, sample_u_high):
    sampled_u = float(rng.uniform(sample_u_low, sample_u_high))
    patch_values = linear_quantile_sample(quantile_series, sampled_u)
    stats_list = [
        linear_neg_entropy(step_quantiles=step_quantiles, sample_u_low=sample_u_low, sample_u_high=sample_u_high)
        for step_quantiles in quantile_series
    ]
    return [float(v) for v in patch_values], stats_list


def inference_multiple(cfg, pipeline, context):
    sample_u_low = float(cfg.multiple_cfg.get("sample_u_low", 0.1))
    sample_u_high = float(cfg.multiple_cfg.get("sample_u_high", 0.9))
    validate_decile_sample_range(sample_u_low, sample_u_high, "TimesFM-2.5")
    multi_step_length = resolve_multi_step_length(
        cfg.multiple_cfg,
        default_length=cfg.multiple_cfg.get("timesfm_patch_length", 128),
        max_length=128,
        model_name="TimesFM-2.5",
    )
    _compile_for_sampling(
        pipeline,
        multi_step_length,
    )

    seed = int(cfg.multiple_cfg.get("seed", 455555))
    num_samples = int(cfg.multiple_cfg.num_samples)
    first_full_forecast = None
    if num_samples > 0 and int(cfg.prediction_length) > 0:
        first_full_forecast = _predict_full_forecast(
            pipeline=pipeline,
            context=context.detach().cpu().numpy().astype(np.float32),
            prediction_length=multi_step_length,
        )

    responses = []
    for sidx in range(num_samples):
        rng = np.random.default_rng(seed + sidx)
        sample, prob_info = _multi_step_autoregressive_sample(
            pipeline=pipeline,
            context=context,
            prediction_length=cfg.prediction_length,
            multi_step_length=multi_step_length,
            rng=rng,
            sample_u_low=sample_u_low,
            sample_u_high=sample_u_high,
            first_full_forecast=first_full_forecast,
        )
        responses.append({"predicted series": sample, "prob_info": prob_info})
    return responses


def inference_multiple_batch(cfg, pipeline, contexts):
    sample_u_low = float(cfg.multiple_cfg.get("sample_u_low", 0.1))
    sample_u_high = float(cfg.multiple_cfg.get("sample_u_high", 0.9))
    validate_decile_sample_range(sample_u_low, sample_u_high, "TimesFM-2.5")
    multi_step_length = resolve_multi_step_length(
        cfg.multiple_cfg,
        default_length=cfg.multiple_cfg.get("timesfm_patch_length", 128),
        max_length=128,
        model_name="TimesFM-2.5",
    )
    _compile_for_sampling(pipeline, multi_step_length)

    seed = int(cfg.multiple_cfg.get("seed", 455555))
    batch_start_index = int(cfg.get("_batch_start_index", 0))
    num_samples = int(cfg.multiple_cfg.num_samples)
    prediction_length = int(cfg.prediction_length)
    context_arrays = [context.detach().cpu().numpy().astype(np.float32) for context in contexts]

    out = [[{"predicted series": [], "prob_info": init_prob_info()} for _ in range(num_samples)] for _ in contexts]
    rolling_contexts = [
        [context.copy() for _ in range(num_samples)]
        for context in context_arrays
    ]
    rngs = [
        [np.random.default_rng(seed + (batch_start_index + bidx) * num_samples + sidx) for sidx in range(num_samples)]
        for bidx in range(len(contexts))
    ]

    generated = 0
    while generated < prediction_length:
        use_len = min(multi_step_length, prediction_length - generated)
        if generated == 0:
            forecast_batch = _predict_full_forecast_batch(
                pipeline,
                context_arrays,
                multi_step_length,
            )
            flat_indices = [(bidx, sidx) for bidx in range(len(contexts)) for sidx in range(num_samples)]
        else:
            flat_contexts = []
            flat_indices = []
            for bidx in range(len(contexts)):
                for sidx in range(num_samples):
                    flat_contexts.append(rolling_contexts[bidx][sidx])
                    flat_indices.append((bidx, sidx))
            flat_forecasts = _predict_full_forecast_batch(
                pipeline,
                flat_contexts,
                multi_step_length,
            )

        for flat_pos, (bidx, sidx) in enumerate(flat_indices):
            full_forecast = forecast_batch[bidx] if generated == 0 else flat_forecasts[flat_pos]
            patch_values, stats_list = _sample_quantile_series(
                full_forecast[:use_len, 1:],
                rngs[bidx][sidx],
                sample_u_low,
                sample_u_high,
            )
            response = out[bidx][sidx]
            response["predicted series"].extend(patch_values)
            for stats in stats_list:
                response["prob_info"]["neg entropy"].append(stats)
            rolling_contexts[bidx][sidx] = np.concatenate(
                [
                    rolling_contexts[bidx][sidx],
                    np.asarray(patch_values, dtype=np.float32),
                ]
            )

        generated += use_len

    return out
