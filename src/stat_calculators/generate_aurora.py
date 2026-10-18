import numpy as np
import torch
from scipy.stats import gaussian_kde

from src.stat_calculators.generate_utils import as_numpy, ensure_sample_batch_shape
from src.utils.randomness import seed_from_cfg


def _ensure_sample_shape(arr, num_samples, prediction_length):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise ValueError(f"Unexpected Aurora output shape: {arr.shape}")

    if arr.shape[0] != num_samples and arr.shape[1] == num_samples:
        arr = arr.T
    if arr.ndim != 2:
        raise ValueError(f"Expected sample-major Aurora output, got shape {arr.shape}")
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


def _forward_samples(cfg, pipeline, context, num_samples):
    seed_from_cfg(cfg)
    inference_token_len = int(cfg.get("inference_token_len", 48))
    if inference_token_len < 1:
        raise ValueError("inference_token_len must be positive")

    context = context.to(next(pipeline.parameters()).device)
    with torch.inference_mode():
        output = pipeline.generate(
            inputs=context.unsqueeze(0),
            max_output_length=cfg.prediction_length,
            num_samples=num_samples,
            inference_token_len=inference_token_len,
            revin=bool(cfg.get("revin", True)),
        )
    return _ensure_sample_shape(
        as_numpy(output),
        num_samples=num_samples,
        prediction_length=cfg.prediction_length,
    )


def _forward_samples_batch(cfg, pipeline, contexts, num_samples):
    seed_from_cfg(cfg, cfg.get("_batch_start_index", 0))
    if not contexts:
        return np.empty((0, int(num_samples), int(cfg.prediction_length)), dtype=float)

    inference_token_len = int(cfg.get("inference_token_len", 48))
    if inference_token_len < 1:
        raise ValueError("inference_token_len must be positive")

    device = next(pipeline.parameters()).device
    out = [None] * len(contexts)
    groups = {}
    for idx, context in enumerate(contexts):
        groups.setdefault(int(context.numel()), []).append(idx)

    for indices in groups.values():
        context_tensor = torch.stack([contexts[i] for i in indices], dim=0).to(device)
        with torch.inference_mode():
            output = pipeline.generate(
                inputs=context_tensor,
                max_output_length=cfg.prediction_length,
                num_samples=num_samples,
                inference_token_len=inference_token_len,
                revin=bool(cfg.get("revin", True)),
            )
        batch_samples = ensure_sample_batch_shape(
            output,
            batch_size=len(indices),
            num_samples=num_samples,
            prediction_length=cfg.prediction_length,
            model_name="Aurora",
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
