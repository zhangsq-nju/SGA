import numpy as np
import torch
from scipy.stats import gaussian_kde

from src.stat_calculators.generate_utils import as_numpy, ensure_sample_batch_shape, model_param_device
from src.utils.randomness import seed_from_cfg


def _ensure_sample_shape(arr, num_samples, prediction_length):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 2:
        if arr.shape[0] != num_samples and arr.shape[1] == num_samples:
            arr = arr.T
    elif arr.ndim == 3:
        arr = arr[0]
    else:
        raise ValueError(f"Unexpected Sundial output shape: {arr.shape}")

    if arr.ndim != 2:
        raise ValueError(f"Expected sample-major Sundial output, got shape {arr.shape}")
    if arr.shape[0] != num_samples and arr.shape[1] == num_samples:
        arr = arr.T
    return arr[:num_samples, :prediction_length]


def _step_kde_densities(samples):
    samples = np.asarray(samples, dtype=np.float64)
    densities = np.ones_like(samples, dtype=np.float64)
    for step in range(samples.shape[1]):
        values = samples[:, step]
        if len(values) < 2 or not np.isfinite(values).all() or np.allclose(values, values[0]):
            continue
        try:
            densities[:, step] = gaussian_kde(values)(values)
        except Exception:
            densities[:, step] = 1.0
    return densities


def _build_prob_info(all_densities):
    densities = np.clip(np.asarray(all_densities, dtype=np.float64), 1e-12, None)
    probabilities = densities / densities.sum(axis=0, keepdims=True)
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    negative_entropy = np.sum(probabilities * np.log(probabilities), axis=0)
    return {"neg entropy": negative_entropy.astype(float).tolist()}


def _align_context(context, input_token_len):
    if input_token_len <= 1:
        return context
    if len(context) < input_token_len:
        pad_value = context[0].item() if hasattr(context[0], "item") else context[0]
        pad = torch.full(
            (input_token_len - len(context),),
            float(pad_value),
            dtype=context.dtype,
            device=context.device,
        )
        return torch.cat([pad, context], dim=0)
    aligned_len = (len(context) // input_token_len) * input_token_len
    return context[-aligned_len:]


def _check_prediction_length(cfg, pipeline):
    max_output_len = max(getattr(pipeline.config, "output_token_lens", [cfg.prediction_length]))
    if cfg.prediction_length > max_output_len:
        raise ValueError(
            f"Sundial prediction_length={cfg.prediction_length} exceeds single-patch "
            f"output length {max_output_len}. Current integration uses direct forward "
            "because transformers generate is incompatible with the local version."
        )


def _forward_samples(cfg, pipeline, context, num_samples):
    seed_from_cfg(cfg)
    _check_prediction_length(cfg, pipeline)
    context = _align_context(
        context,
        int(getattr(pipeline.config, "input_token_len", 1)),
    ).to(model_param_device(pipeline))

    with torch.no_grad():
        outputs = pipeline(
            input_ids=context.unsqueeze(0),
            max_output_length=cfg.prediction_length,
            num_samples=num_samples,
            revin=True,
            use_cache=False,
            return_dict=True,
        )
    return _ensure_sample_shape(
        as_numpy(outputs.logits),
        num_samples=num_samples,
        prediction_length=cfg.prediction_length,
    )


def _forward_samples_batch(cfg, pipeline, contexts, num_samples):
    seed_from_cfg(cfg, cfg.get("_batch_start_index", 0))
    _check_prediction_length(cfg, pipeline)
    if not contexts:
        return np.empty((0, int(num_samples), int(cfg.prediction_length)), dtype=float)

    aligned = [
        _align_context(context, int(getattr(pipeline.config, "input_token_len", 1))).to(model_param_device(pipeline))
        for context in contexts
    ]
    out = [None] * len(contexts)
    groups = {}
    for idx, context in enumerate(aligned):
        groups.setdefault(int(context.numel()), []).append(idx)

    for indices in groups.values():
        input_ids = torch.stack([aligned[i] for i in indices], dim=0)
        flat_batch_size = len(indices) * int(num_samples)
        flat_input_ids = input_ids.repeat_interleave(int(num_samples), dim=0)
        means = flat_input_ids.mean(1, keepdim=True).detach()
        stdev = flat_input_ids.std(dim=1, keepdim=True, unbiased=False).detach()
        stdev = torch.where(stdev > 1e-2, stdev, torch.ones_like(stdev))
        normalized_input_ids = (flat_input_ids - means) / stdev
        with torch.no_grad():
            outputs = pipeline(
                input_ids=normalized_input_ids,
                max_output_length=cfg.prediction_length,
                num_samples=1,
                revin=False,
                use_cache=False,
                return_dict=True,
            )
        predictions = outputs.logits
        if predictions.ndim == 3 and predictions.shape[1] == 1:
            predictions = predictions[:, 0, :]
        elif predictions.ndim == 1 and flat_batch_size == 1:
            predictions = predictions.reshape(1, -1)
        if predictions.ndim != 2 or predictions.shape[0] < flat_batch_size:
            raise ValueError(
                f"Unexpected Sundial expanded-batch output shape: {tuple(predictions.shape)}; "
                f"expected at least ({flat_batch_size}, {cfg.prediction_length})"
            )
        predictions = (
            predictions[:flat_batch_size, : cfg.prediction_length] * stdev[:flat_batch_size]
            + means[:flat_batch_size]
        )
        batch_samples = as_numpy(predictions).astype(np.float32).reshape(
            len(indices),
            int(num_samples),
            int(cfg.prediction_length),
        )
        for pos, original_idx in enumerate(indices):
            out[original_idx] = batch_samples[pos]

    return np.stack(out, axis=0)


def inference_once(cfg, pipeline, context):
    num_samples = int(cfg.once_cfg.get("num_samples", 20))
    samples = _forward_samples(cfg, pipeline, context, num_samples=num_samples)
    return np.median(samples, axis=0).astype(float).tolist()


def inference_once_batch(cfg, pipeline, contexts):
    num_samples = int(cfg.once_cfg.get("num_samples", 20))
    samples = _forward_samples_batch(cfg, pipeline, contexts, num_samples=num_samples)
    return np.median(samples, axis=1).astype(float).tolist()


def inference_multiple(cfg, pipeline, context):
    num_samples = int(cfg.multiple_cfg.num_samples)
    predicted_samples = _forward_samples(cfg, pipeline, context, num_samples=num_samples).astype(float)
    step_densities = _step_kde_densities(predicted_samples)

    responses = []
    for sidx in range(num_samples):
        responses.append(
            {
                "predicted series": predicted_samples[sidx].tolist(),
                "step_density_estimates": step_densities[sidx].astype(float).tolist(),
                "prob_info": _build_prob_info(step_densities),
            }
        )
    return responses


def inference_multiple_batch(cfg, pipeline, contexts):
    num_samples = int(cfg.multiple_cfg.num_samples)
    predicted_batch = _forward_samples_batch(cfg, pipeline, contexts, num_samples=num_samples).astype(float)

    responses_batch = []
    for predicted_samples in predicted_batch:
        step_densities = _step_kde_densities(predicted_samples)
        responses = []
        for sidx in range(num_samples):
            responses.append(
                {
                    "predicted series": predicted_samples[sidx].tolist(),
                    "step_density_estimates": step_densities[sidx].astype(float).tolist(),
                    "prob_info": _build_prob_info(step_densities),
                }
            )
        responses_batch.append(responses)
    return responses_batch
