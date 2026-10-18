import torch
from chronos import BaseChronosPipeline, ChronosPipeline
from transformers import AutoModelForCausalLM


def infer_backend_from_model_name(model_name):
    model_name = str(model_name).lower()
    if "aurora" in model_name:
        return "aurora"
    if "timer-s1" in model_name or "timer_s1" in model_name:
        return "timer_s1"
    if "timesfm" in model_name:
        return "timesfm25"
    if "sundial" in str(model_name).lower():
        return "sundial"
    if "chronos-2" in model_name or "chronos2" in model_name:
        return "chronos2"
    return "chronos_t5"


def _resolve_device_map(requested_device):
    if requested_device and str(requested_device).lower() != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype_name, device_map):
    if dtype_name != "auto":
        if not hasattr(torch, dtype_name):
            raise ValueError(f"Unsupported dtype: {dtype_name}")
        return getattr(torch, dtype_name)
    if device_map == "mps":
        return torch.float32
    return torch.bfloat16


def _resolve_timer_s1_device_map(requested_device):
    device_map = _resolve_device_map(requested_device)
    if requested_device is None or str(requested_device).lower() == "auto":
        # Timer-S1 is ~16GB bf16 and MPS allocator warmup fails on this machine.
        if device_map == "mps":
            return "cpu"
    return device_map


def _patch_sundial_transformers_cache_api():
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: getattr(self, "max_cache_len", None)
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, *args, **kwargs: self.get_seq_length()


def load_model(cfg):
    model_path = cfg.model_name
    inference_cfg = cfg.inference

    backend = infer_backend_from_model_name(model_path)
    device_map = (
        _resolve_timer_s1_device_map(inference_cfg.get("device_map", None))
        if backend == "timer_s1"
        else _resolve_device_map(inference_cfg.get("device_map", None))
    )

    if backend == "aurora":
        try:
            from aurora import load_model as load_aurora_model
        except ImportError as exc:
            raise ImportError(
                "aurora-model is unavailable. Install Aurora support with "
                "`pip install aurora-model` before using an Aurora model."
            ) from exc
        print(f"Loading Aurora model: {model_path}, device_map={device_map}")
        pipeline = load_aurora_model(
            repo_id=model_path,
            weights_filename=inference_cfg.get("weights_filename", "model.safetensors"),
            cache_dir=inference_cfg.get("cache_dir", None),
            force_download=bool(inference_cfg.get("force_download", False)),
            device=device_map,
        )
        pipeline.eval()
    elif backend == "sundial":
        # Sundial's embedding layer concatenates a float32 mask with the input,
        # so bf16 weights fail with "Float and BFloat16" dtype mismatch.
        dtype = torch.float32
        _patch_sundial_transformers_cache_api()
        print(f"Loading Sundial model: {model_path}, device_map={device_map}, dtype={dtype}")
        pipeline = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
        )
        pipeline.to(device_map)
        pipeline.eval()
    elif backend == "chronos2":
        print(f"Loading Chronos-2 model: {model_path}, device_map={device_map}")
        pipeline = BaseChronosPipeline.from_pretrained(
            model_path,
            device_map=device_map,
        )
    elif backend == "timer_s1":
        dtype = _resolve_dtype(inference_cfg.get("dtype", "auto"), device_map)
        print(f"Loading Timer-S1 model: {model_path}, device_map={device_map}, dtype={dtype}")
        pipeline = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
        )
        pipeline.to(device_map)
        pipeline.eval()
    elif backend == "timesfm25":
        try:
            import timesfm
        except ImportError as exc:
            raise ImportError(
                "timesfm is unavailable. Install TimesFM-2.5 support with "
                "`pip install 'timesfm[torch]'` before using a TimesFM model."
            ) from exc
        torch.set_float32_matmul_precision(inference_cfg.get("matmul_precision", "high"))
        print(f"Loading TimesFM-2.5 model: {model_path}")
        pipeline = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            model_path,
            torch_compile=bool(inference_cfg.get("torch_compile", False)),
        )
        pipeline.compile(
            timesfm.ForecastConfig(
                max_context=int(inference_cfg.context_length) + int(inference_cfg.prediction_length),
                max_horizon=max(
                    int(inference_cfg.prediction_length),
                    int(inference_cfg.multiple_cfg.get("model_prediction_length", 1)),
                ),
                normalize_inputs=bool(inference_cfg.get("normalize_inputs", True)),
                per_core_batch_size=int(inference_cfg.get("per_core_batch_size", 32)),
                use_continuous_quantile_head=bool(
                    inference_cfg.get("use_continuous_quantile_head", True)
                ),
                force_flip_invariance=bool(inference_cfg.get("force_flip_invariance", True)),
                infer_is_positive=bool(inference_cfg.get("infer_is_positive", True)),
                fix_quantile_crossing=bool(inference_cfg.get("fix_quantile_crossing", True)),
            )
        )
    else:
        dtype = _resolve_dtype(inference_cfg.get("dtype", "auto"), device_map)
        print(f"Loading Chronos-T5 model: {model_path}, device_map={device_map}, dtype={dtype}")
        pipeline = ChronosPipeline.from_pretrained(
            model_path,
            device_map=device_map,
            dtype=dtype,
        )

    return {"pipeline": pipeline, "backend": backend}
