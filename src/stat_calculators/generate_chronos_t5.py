import numpy as np
import torch

from src.stat_calculators.generate_utils import (
    build_token_entropy,
    ensure_sample_series_shape,
    to_median_forecast,
    to_quantile_batch,
)
from src.utils.randomness import seed_from_cfg


def inference_once(cfg, pipeline, context):
    seed_from_cfg(cfg)
    quantile_forecast, _ = pipeline.predict_quantiles(
        context,
        prediction_length=cfg.prediction_length,
        quantile_levels=[0.5],
        num_samples=int(cfg.once_cfg.get("num_samples", 20)),
        temperature=cfg.once_cfg.temperature,
        top_k=cfg.once_cfg.top_k,
        top_p=cfg.once_cfg.top_p,
    )
    return to_median_forecast(quantile_forecast)


def _to_context_batch_tensor(contexts):
    max_len = max(int(context.numel()) for context in contexts)
    batch = torch.full((len(contexts), max_len), float("nan"), dtype=torch.float32)
    for idx, context in enumerate(contexts):
        context = context.detach().cpu().to(dtype=torch.float32)
        batch[idx, -int(context.numel()) :] = context
    return batch


def inference_once_batch(cfg, pipeline, contexts):
    seed_from_cfg(cfg, cfg.get("_batch_start_index", 0))
    context_batch = _to_context_batch_tensor(contexts)
    quantile_forecast, _ = pipeline.predict_quantiles(
        context_batch,
        prediction_length=cfg.prediction_length,
        quantile_levels=[0.5],
        num_samples=int(cfg.once_cfg.get("num_samples", 20)),
        temperature=cfg.once_cfg.temperature,
        top_k=cfg.once_cfg.top_k,
        top_p=cfg.once_cfg.top_p,
    )
    median = to_quantile_batch(
        quantile_forecast,
        batch_size=len(contexts),
        prediction_length=cfg.prediction_length,
        num_quantiles=1,
    )[:, :, 0]
    return median.astype(float).tolist()


def _model_device(pipeline):
    return next(pipeline.model.model.parameters()).device


def _run_generate_with_scores(
    pipeline,
    token_ids,
    attention_mask,
    prediction_length,
    num_samples,
    temperature,
    top_k,
):
    device = _model_device(pipeline)
    return pipeline.model.model.generate(
        input_ids=token_ids.to(device),
        attention_mask=attention_mask.to(device),
        max_new_tokens=prediction_length,
        num_return_sequences=num_samples,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        output_scores=True,
        return_dict_in_generate=True,
    )


def inference_multiple(cfg, pipeline, context):
    seed_from_cfg(cfg)
    context = context.unsqueeze(0)
    token_ids, attention_mask, scale = pipeline.tokenizer.context_input_transform(context.cpu())
    generation_output = _run_generate_with_scores(
        pipeline=pipeline,
        token_ids=token_ids,
        attention_mask=attention_mask,
        prediction_length=cfg.prediction_length,
        num_samples=cfg.multiple_cfg.num_samples,
        temperature=cfg.multiple_cfg.temperature,
        top_k=cfg.multiple_cfg.top_k,
    )

    sampled_tokens = generation_output.sequences[:, -cfg.prediction_length :]
    logits_tensor = torch.stack(generation_output.scores, dim=1)
    probs_tensor = torch.softmax(logits_tensor, dim=-1)

    k = min(cfg.multiple_cfg.top_token_count, probs_tensor.shape[-1])
    top_probs = torch.topk(probs_tensor, k=k, dim=-1).values

    sampled_tokens_cpu = sampled_tokens.to("cpu")
    forecast_samples = pipeline.tokenizer.output_transform(sampled_tokens_cpu, scale)

    predicted_samples = ensure_sample_series_shape(
        forecast_samples.squeeze().detach().cpu().numpy().astype(np.float32)
    ).tolist()

    responses = []
    for sidx in range(cfg.multiple_cfg.num_samples):
        responses.append(
            {
                "predicted series": predicted_samples[sidx],
                "prob_info": build_token_entropy(top_probs[sidx].detach().cpu().numpy().astype(np.float32).tolist()),
            }
        )
    return responses


def inference_multiple_batch(cfg, pipeline, contexts):
    seed_from_cfg(cfg, cfg.get("_batch_start_index", 0))
    batch_size = len(contexts)
    num_samples = int(cfg.multiple_cfg.num_samples)
    prediction_length = int(cfg.prediction_length)
    context_batch = _to_context_batch_tensor(contexts)
    token_ids, attention_mask, scale = pipeline.tokenizer.context_input_transform(context_batch.cpu())
    generation_output = _run_generate_with_scores(
        pipeline=pipeline,
        token_ids=token_ids,
        attention_mask=attention_mask,
        prediction_length=prediction_length,
        num_samples=num_samples,
        temperature=cfg.multiple_cfg.temperature,
        top_k=cfg.multiple_cfg.top_k,
    )

    sampled_tokens = generation_output.sequences[:, -prediction_length:]
    logits_tensor = torch.stack(generation_output.scores, dim=1)
    probs_tensor = torch.softmax(logits_tensor, dim=-1)

    k = min(cfg.multiple_cfg.top_token_count, probs_tensor.shape[-1])
    top_probs = torch.topk(probs_tensor, k=k, dim=-1).values

    sampled_tokens_cpu = sampled_tokens.to("cpu")
    sampled_tokens_3d = sampled_tokens_cpu.reshape(batch_size, num_samples, prediction_length)
    forecast_samples = pipeline.tokenizer.output_transform(sampled_tokens_3d, scale)
    forecast_arr = ensure_sample_series_shape(
        forecast_samples.detach().cpu().numpy().astype(np.float32)
    )
    if forecast_arr.shape != (batch_size, num_samples, prediction_length):
        raise ValueError(
            "Unexpected Chronos-T5 batch forecast shape after output_transform: "
            f"{forecast_arr.shape}; expected {(batch_size, num_samples, prediction_length)}"
        )
    top_probs_arr = top_probs.detach().cpu().numpy().astype(np.float32).reshape(
        batch_size,
        num_samples,
        prediction_length,
        k,
    )

    responses_batch = []
    for bidx in range(batch_size):
        responses = []
        for sidx in range(num_samples):
            responses.append(
                {
                    "predicted series": forecast_arr[bidx, sidx].astype(float).tolist(),
                    "prob_info": build_token_entropy(top_probs_arr[bidx, sidx].tolist()),
                }
            )
        responses_batch.append(responses)
    return responses_batch
