import json
import time

import torch
from omegaconf import open_dict
from tqdm import tqdm

from src.stat_calculators import generate_aurora
from src.stat_calculators import generate_chronos_2
from src.stat_calculators import generate_chronos_t5
from src.stat_calculators import generate_sundial
from src.stat_calculators import generate_timer_s1
from src.stat_calculators import generate_timesfm_25
from src.utils.randomness import seed_from_cfg


_ONCE_GENERATORS = {
    "aurora": generate_aurora.inference_once,
    "chronos2": generate_chronos_2.inference_once,
    "chronos_t5": generate_chronos_t5.inference_once,
    "sundial": generate_sundial.inference_once,
    "timer_s1": generate_timer_s1.inference_once,
    "timesfm25": generate_timesfm_25.inference_once,
}

_ONCE_BATCH_GENERATORS = {
    "aurora": getattr(generate_aurora, "inference_once_batch", None),
    "chronos2": getattr(generate_chronos_2, "inference_once_batch", None),
    "chronos_t5": getattr(generate_chronos_t5, "inference_once_batch", None),
    "sundial": getattr(generate_sundial, "inference_once_batch", None),
    "timer_s1": getattr(generate_timer_s1, "inference_once_batch", None),
    "timesfm25": getattr(generate_timesfm_25, "inference_once_batch", None),
}


def model_inference_once(cfg, model, data, log, output_dir):
    pipeline = model["pipeline"]
    backend = model.get("backend", "chronos_t5")
    contexts, targets, ids = data["contexts"], data["targets"], data["ids"]

    log.debug(f"Starting point forecast inference, backend={backend}")
    start_time = time.time()

    infer_once = _ONCE_GENERATORS.get(backend, generate_chronos_t5.inference_once)
    infer_once_batch = _ONCE_BATCH_GENERATORS.get(backend)
    response_data = []
    batch_size = int(cfg.get("batch_size", 32))
    for start in tqdm(range(0, len(contexts), batch_size), desc="inference_once"):
        end = min(start + batch_size, len(contexts))
        batch_contexts = [torch.tensor(context, dtype=torch.float32) for context in contexts[start:end]]
        if infer_once_batch is not None:
            old_batch_start = cfg.get("_batch_start_index", None)
            with open_dict(cfg):
                cfg["_batch_start_index"] = start
            try:
                y_preds = infer_once_batch(cfg, pipeline, batch_contexts)
            finally:
                with open_dict(cfg):
                    if old_batch_start is None:
                        del cfg["_batch_start_index"]
                    else:
                        cfg["_batch_start_index"] = old_batch_start
        else:
            y_preds = []
            for offset, context in enumerate(batch_contexts):
                seed_from_cfg(cfg, start + offset)
                y_preds.append(infer_once(cfg, pipeline, context))
        for offset, y_pred in enumerate(y_preds):
            idx = start + offset
            response_data.append(
                {
                    "id": ids[idx],
                    "context": contexts[idx],
                    "y_true": targets[idx],
                    "y_pred": y_pred,
                }
            )

    with open(f"{output_dir}/response_data.jsonl", "w", encoding="utf-8") as f:
        for item in response_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    inference_time = time.time() - start_time
    log.debug(f"Total Inference Time (once): {inference_time:.2f}s")

    return response_data
