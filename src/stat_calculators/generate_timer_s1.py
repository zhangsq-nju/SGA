import numpy as np
import torch

from src.stat_calculators.generate_utils import (
    DECILE_QUANTILE_LEVELS,
    as_numpy,
    init_prob_info,
    linear_neg_entropy,
    linear_quantile_sample,
    model_param_device,
    model_param_dtype,
    resolve_multi_step_length,
    to_decile_quantile_batch,
    validate_decile_sample_range,
)


TIMER_S1_MEDIAN_INDEX = 4


def _to_quantile_array(output, prediction_length):
    arr = as_numpy(output).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D Timer-S1 quantile output, got shape {arr.shape}")

    num_quantiles = len(DECILE_QUANTILE_LEVELS)
    if arr.shape == (num_quantiles, prediction_length):
        return arr.T
    if arr.shape == (prediction_length, num_quantiles):
        return arr
    raise ValueError(
        "Unexpected Timer-S1 output shape "
        f"{arr.shape}; expected ({num_quantiles}, {prediction_length}) "
        f"or ({prediction_length}, {num_quantiles})"
    )


def _predict_quantiles(pipeline, context, prediction_length, revin=True):
    seqs = context.unsqueeze(0).to(device=model_param_device(pipeline), dtype=model_param_dtype(pipeline))
    with torch.inference_mode():
        output = pipeline.generate(seqs, max_new_tokens=prediction_length, revin=revin)
    return _to_quantile_array(output, prediction_length)


def _predict_quantiles_batch(pipeline, contexts, prediction_length, revin=True):
    if not contexts:
        return np.empty((0, int(prediction_length), len(DECILE_QUANTILE_LEVELS)), dtype=np.float32)
    out = [None] * len(contexts)
    groups = {}
    for idx, context in enumerate(contexts):
        groups.setdefault(int(context.numel()), []).append(idx)

    for indices in groups.values():
        seqs = torch.stack([contexts[i] for i in indices], dim=0).to(
            device=model_param_device(pipeline),
            dtype=model_param_dtype(pipeline),
        )
        with torch.inference_mode():
            output = pipeline.generate(seqs, max_new_tokens=prediction_length, revin=revin)
        batch_quantiles = to_decile_quantile_batch(
            output,
            batch_size=len(indices),
            prediction_length=prediction_length,
        )
        for pos, original_idx in enumerate(indices):
            out[original_idx] = batch_quantiles[pos]
    return np.stack(out, axis=0)


def inference_once(cfg, pipeline, context):
    quantiles = _predict_quantiles(
        pipeline=pipeline,
        context=context,
        prediction_length=cfg.prediction_length,
        revin=bool(cfg.get("revin", True)),
    )
    return quantiles[:, TIMER_S1_MEDIAN_INDEX].astype(float).tolist()


def inference_once_batch(cfg, pipeline, contexts):
    quantiles = _predict_quantiles_batch(
        pipeline=pipeline,
        contexts=contexts,
        prediction_length=cfg.prediction_length,
        revin=bool(cfg.get("revin", True)),
    )
    return quantiles[:, :, TIMER_S1_MEDIAN_INDEX].astype(float).tolist()


def _multi_step_autoregressive_sample(
    pipeline,
    context,
    prediction_length,
    multi_step_length,
    rng,
    sample_u_low,
    sample_u_high,
    revin=True,
    first_quantile_arr=None,
):
    sampled_values = []
    prob_info = init_prob_info()
    rolling_context = context.clone()
    is_first_block = True

    while len(sampled_values) < prediction_length:
        if is_first_block and first_quantile_arr is not None:
            quantile_arr = first_quantile_arr
        else:
            quantile_arr = _predict_quantiles(
                pipeline=pipeline,
                context=rolling_context,
                prediction_length=multi_step_length,
                revin=revin,
            )
        is_first_block = False

        use_len = min(multi_step_length, prediction_length - len(sampled_values))
        quantile_series = quantile_arr[:use_len]
        sampled_u = float(rng.uniform(sample_u_low, sample_u_high))
        patch_values = linear_quantile_sample(quantile_series, sampled_u)

        for value, step_quantiles in zip(patch_values, quantile_series):
            stats = linear_neg_entropy(step_quantiles=step_quantiles, sample_u_low=sample_u_low, sample_u_high=sample_u_high)
            sampled_values.append(float(value))
            prob_info["neg entropy"].append(stats)

        next_values = torch.tensor(patch_values, dtype=rolling_context.dtype)
        rolling_context = torch.cat([rolling_context.cpu(), next_values], dim=0)

    return sampled_values, prob_info


def inference_multiple(cfg, pipeline, context):
    sample_u_low = float(cfg.multiple_cfg.get("sample_u_low", 0.1))
    sample_u_high = float(cfg.multiple_cfg.get("sample_u_high", 0.9))
    validate_decile_sample_range(sample_u_low, sample_u_high, "Timer-S1")
    multi_step_length = resolve_multi_step_length(
        cfg.multiple_cfg,
        default_length=cfg.multiple_cfg.get("timer_s1_patch_length", 16),
        max_length=16,
        model_name="Timer-S1",
    )

    seed = int(cfg.multiple_cfg.get("seed", 455555))
    num_samples = int(cfg.multiple_cfg.num_samples)
    revin = bool(cfg.get("revin", True))
    first_quantile_arr = None
    if num_samples > 0 and int(cfg.prediction_length) > 0:
        first_quantile_arr = _predict_quantiles(
            pipeline=pipeline,
            context=context,
            prediction_length=multi_step_length,
            revin=revin,
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
            revin=revin,
            first_quantile_arr=first_quantile_arr,
        )
        responses.append({"predicted series": sample, "prob_info": prob_info})
    return responses


def _sample_quantile_series(quantile_series, rng, sample_u_low, sample_u_high):
    sampled_u = float(rng.uniform(sample_u_low, sample_u_high))
    patch_values = linear_quantile_sample(quantile_series, sampled_u)
    stats_list = [
        linear_neg_entropy(step_quantiles=step_quantiles, sample_u_low=sample_u_low, sample_u_high=sample_u_high)
        for step_quantiles in quantile_series
    ]
    return [float(v) for v in patch_values], stats_list


def inference_multiple_batch(cfg, pipeline, contexts):
    sample_u_low = float(cfg.multiple_cfg.get("sample_u_low", 0.1))
    sample_u_high = float(cfg.multiple_cfg.get("sample_u_high", 0.9))
    validate_decile_sample_range(sample_u_low, sample_u_high, "Timer-S1")
    multi_step_length = resolve_multi_step_length(
        cfg.multiple_cfg,
        default_length=cfg.multiple_cfg.get("timer_s1_patch_length", 16),
        max_length=16,
        model_name="Timer-S1",
    )
    seed = int(cfg.multiple_cfg.get("seed", 455555))
    batch_start_index = int(cfg.get("_batch_start_index", 0))
    num_samples = int(cfg.multiple_cfg.num_samples)
    prediction_length = int(cfg.prediction_length)
    revin = bool(cfg.get("revin", True))

    out = [[{"predicted series": [], "prob_info": init_prob_info()} for _ in range(num_samples)] for _ in contexts]
    rolling_contexts = [[context.clone() for _ in range(num_samples)] for context in contexts]
    rngs = [
        [np.random.default_rng(seed + (batch_start_index + bidx) * num_samples + sidx) for sidx in range(num_samples)]
        for bidx in range(len(contexts))
    ]

    generated = 0
    while generated < prediction_length:
        use_len = min(multi_step_length, prediction_length - generated)
        if generated == 0:
            forecast_batch = _predict_quantiles_batch(pipeline, contexts, multi_step_length, revin=revin)
            flat_indices = [(bidx, sidx) for bidx in range(len(contexts)) for sidx in range(num_samples)]
        else:
            flat_contexts = []
            flat_indices = []
            for bidx in range(len(contexts)):
                for sidx in range(num_samples):
                    flat_contexts.append(rolling_contexts[bidx][sidx])
                    flat_indices.append((bidx, sidx))
            flat_forecasts = _predict_quantiles_batch(pipeline, flat_contexts, multi_step_length, revin=revin)

        for flat_pos, (bidx, sidx) in enumerate(flat_indices):
            quantile_arr = forecast_batch[bidx] if generated == 0 else flat_forecasts[flat_pos]
            patch_values, stats_list = _sample_quantile_series(
                quantile_arr[:use_len],
                rngs[bidx][sidx],
                sample_u_low,
                sample_u_high,
            )
            response = out[bidx][sidx]
            response["predicted series"].extend(patch_values)
            for stats in stats_list:
                response["prob_info"]["neg entropy"].append(stats)
            rolling_contexts[bidx][sidx] = torch.cat(
                [
                    rolling_contexts[bidx][sidx].cpu(),
                    torch.tensor(patch_values, dtype=rolling_contexts[bidx][sidx].dtype),
                ],
                dim=0,
            )
        generated += use_len

    return out
