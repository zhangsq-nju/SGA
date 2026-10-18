import numpy as np
import torch

from src.stat_calculators.generate_utils import (
    derive_quantile_levels,
    init_prob_info,
    linear_neg_entropy,
    linear_quantile_sample,
    resolve_multi_step_length,
    to_1d_forecast,
    to_quantile_array,
    to_quantile_batch,
)


def inference_once(cfg, pipeline, context):
    _, point_forecast = pipeline.predict_quantiles(
        inputs=[context],
        prediction_length=cfg.prediction_length,
        quantile_levels=[0.5],
        batch_size=cfg.get("batch_size", 32),
    )
    return to_1d_forecast(point_forecast[0])


def inference_once_batch(cfg, pipeline, contexts):
    _, point_forecasts = pipeline.predict_quantiles(
        inputs=contexts,
        prediction_length=cfg.prediction_length,
        quantile_levels=[0.5],
        batch_size=cfg.get("batch_size", 32),
    )
    return [to_1d_forecast(point_forecasts[i]) for i in range(len(contexts))]


def _predict_quantile_arr(pipeline, context, prediction_length, quantile_levels, batch_size):
    quantiles, _ = pipeline.predict_quantiles(
        inputs=[context],
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
        batch_size=batch_size,
    )
    return to_quantile_array(
        quantiles[0],
        model_prediction_length=prediction_length,
        num_quantiles=len(quantile_levels),
    )


def _predict_quantile_batch(pipeline, contexts, prediction_length, quantile_levels, batch_size):
    quantiles, _ = pipeline.predict_quantiles(
        inputs=contexts,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
        batch_size=batch_size,
    )
    return to_quantile_batch(
        quantiles,
        batch_size=len(contexts),
        prediction_length=prediction_length,
        num_quantiles=len(quantile_levels),
    )


def _multi_step_autoregressive_sample(
    pipeline,
    context,
    prediction_length,
    multi_step_length,
    quantile_levels,
    batch_size,
    rng,
    sample_u_low,
    sample_u_high,
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
            quantile_arr = _predict_quantile_arr(
                pipeline=pipeline,
                context=rolling_context,
                prediction_length=multi_step_length,
                quantile_levels=quantile_levels,
                batch_size=batch_size,
            )
        is_first_block = False

        use_len = min(multi_step_length, prediction_length - len(sampled_values))
        quantile_series = quantile_arr[:use_len]
        sampled_u = float(rng.uniform(sample_u_low, sample_u_high))
        patch_values = linear_quantile_sample(
            quantile_series,
            sampled_u,
            quantile_levels=quantile_levels,
        )

        for value, step_quantiles in zip(patch_values, quantile_series):
            stats = linear_neg_entropy(step_quantiles=step_quantiles, sample_u_low=sample_u_low, sample_u_high=sample_u_high, quantile_levels=quantile_levels)
            sampled_values.append(float(value))
            prob_info["neg entropy"].append(stats)

        next_values = torch.tensor(patch_values, dtype=rolling_context.dtype)
        rolling_context = torch.cat([rolling_context.cpu(), next_values], dim=0)

    return sampled_values, prob_info


def inference_multiple(cfg, pipeline, context):
    sample_u_low = float(cfg.multiple_cfg.get("sample_u_low", 0.1))
    sample_u_high = float(cfg.multiple_cfg.get("sample_u_high", 0.9))
    quantile_levels = derive_quantile_levels(sample_u_low, sample_u_high)
    multi_step_length = resolve_multi_step_length(
        cfg.multiple_cfg,
        default_length=16,
        max_length=16,
        model_name="Chronos-2",
    )

    batch_size = int(cfg.get("batch_size", 32))
    seed = int(cfg.multiple_cfg.get("seed", 455555))
    num_samples = int(cfg.multiple_cfg.num_samples)
    first_quantile_arr = None
    if num_samples > 0 and int(cfg.prediction_length) > 0:
        first_quantile_arr = _predict_quantile_arr(
            pipeline=pipeline,
            context=context,
            prediction_length=multi_step_length,
            quantile_levels=quantile_levels,
            batch_size=batch_size,
        )

    responses = []
    for sidx in range(num_samples):
        rng = np.random.default_rng(seed + sidx)
        sample, prob_info = _multi_step_autoregressive_sample(
            pipeline=pipeline,
            context=context,
            prediction_length=cfg.prediction_length,
            multi_step_length=multi_step_length,
            quantile_levels=quantile_levels,
            batch_size=batch_size,
            rng=rng,
            sample_u_low=sample_u_low,
            sample_u_high=sample_u_high,
            first_quantile_arr=first_quantile_arr,
        )
        responses.append({"predicted series": sample, "prob_info": prob_info})
    return responses


def _sample_quantile_series(quantile_series, quantile_levels, rng, sample_u_low, sample_u_high):
    sampled_u = float(rng.uniform(sample_u_low, sample_u_high))
    patch_values = linear_quantile_sample(
        quantile_series,
        sampled_u,
        quantile_levels=quantile_levels,
    )
    stats_list = [
        linear_neg_entropy(step_quantiles=step_quantiles, sample_u_low=sample_u_low, sample_u_high=sample_u_high, quantile_levels=quantile_levels)
        for step_quantiles in quantile_series
    ]
    return [float(v) for v in patch_values], stats_list


def inference_multiple_batch(cfg, pipeline, contexts):
    sample_u_low = float(cfg.multiple_cfg.get("sample_u_low", 0.1))
    sample_u_high = float(cfg.multiple_cfg.get("sample_u_high", 0.9))
    quantile_levels = derive_quantile_levels(sample_u_low, sample_u_high)
    multi_step_length = resolve_multi_step_length(
        cfg.multiple_cfg,
        default_length=16,
        max_length=16,
        model_name="Chronos-2",
    )
    batch_size = int(cfg.get("batch_size", 32))
    seed = int(cfg.multiple_cfg.get("seed", 455555))
    batch_start_index = int(cfg.get("_batch_start_index", 0))
    num_samples = int(cfg.multiple_cfg.num_samples)
    prediction_length = int(cfg.prediction_length)

    out = [[{"predicted series": [], "prob_info": init_prob_info()} for _ in range(num_samples)] for _ in contexts]
    rolling_contexts = [
        [context.clone() for _ in range(num_samples)]
        for context in contexts
    ]
    rngs = [
        [np.random.default_rng(seed + (batch_start_index + bidx) * num_samples + sidx) for sidx in range(num_samples)]
        for bidx in range(len(contexts))
    ]

    generated = 0
    while generated < prediction_length:
        use_len = min(multi_step_length, prediction_length - generated)
        if generated == 0:
            forecast_batch = _predict_quantile_batch(
                pipeline,
                contexts,
                multi_step_length,
                quantile_levels,
                batch_size,
            )
            flat_indices = [(bidx, sidx) for bidx in range(len(contexts)) for sidx in range(num_samples)]
        else:
            flat_contexts = []
            flat_indices = []
            for bidx in range(len(contexts)):
                for sidx in range(num_samples):
                    flat_contexts.append(rolling_contexts[bidx][sidx])
                    flat_indices.append((bidx, sidx))
            flat_forecasts = _predict_quantile_batch(
                pipeline,
                flat_contexts,
                multi_step_length,
                quantile_levels,
                batch_size,
            )

        for flat_pos, (bidx, sidx) in enumerate(flat_indices):
            quantile_arr = forecast_batch[bidx] if generated == 0 else flat_forecasts[flat_pos]
            patch_values, stats_list = _sample_quantile_series(
                quantile_arr[:use_len],
                quantile_levels,
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
