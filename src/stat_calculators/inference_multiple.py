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


_MULTIPLE_GENERATORS = {
    "aurora": generate_aurora.inference_multiple,
    "chronos2": generate_chronos_2.inference_multiple,
    "chronos_t5": generate_chronos_t5.inference_multiple,
    "sundial": generate_sundial.inference_multiple,
    "timer_s1": generate_timer_s1.inference_multiple,
    "timesfm25": generate_timesfm_25.inference_multiple,
}

_MULTIPLE_BATCH_GENERATORS = {
    "aurora": getattr(generate_aurora, "inference_multiple_batch", None),
    "chronos2": getattr(generate_chronos_2, "inference_multiple_batch", None),
    "chronos_t5": getattr(generate_chronos_t5, "inference_multiple_batch", None),
    "sundial": getattr(generate_sundial, "inference_multiple_batch", None),
    "timer_s1": getattr(generate_timer_s1, "inference_multiple_batch", None),
    "timesfm25": getattr(generate_timesfm_25, "inference_multiple_batch", None),
}


def model_inference_multiple(cfg, model, data, log, output_dir):
    pipeline = model["pipeline"]
    backend = model.get("backend", "chronos_t5")
    contexts, targets, ids = data["contexts"], data["targets"], data["ids"]

    log.debug(f"Starting forecast trajectory sampling, backend={backend}")
    start_time = time.time()

    infer_multiple = _MULTIPLE_GENERATORS.get(backend, generate_chronos_t5.inference_multiple)
    infer_multiple_batch = _MULTIPLE_BATCH_GENERATORS.get(backend)
    responses_data = []
    batch_size = int(cfg.get("batch_size", 32))
    for start in tqdm(range(0, len(contexts), batch_size), desc="inference_multiple"):
        end = min(start + batch_size, len(contexts))
        batch_contexts = [torch.tensor(context, dtype=torch.float32) for context in contexts[start:end]]
        if infer_multiple_batch is not None:
            old_batch_start = cfg.get("_batch_start_index", None)
            with open_dict(cfg):
                cfg["_batch_start_index"] = start
            try:
                responses_batch = infer_multiple_batch(cfg, pipeline, batch_contexts)
            finally:
                with open_dict(cfg):
                    if old_batch_start is None:
                        del cfg["_batch_start_index"]
                    else:
                        cfg["_batch_start_index"] = old_batch_start
        else:
            responses_batch = []
            for offset, context in enumerate(batch_contexts):
                seed_from_cfg(cfg, start + offset)
                responses_batch.append(infer_multiple(cfg, pipeline, context))
        for offset, responses in enumerate(responses_batch):
            idx = start + offset
            responses_data.append(
                {
                    "id": ids[idx],
                    "y_true": targets[idx],
                    "responses": responses,
                }
            )

    with open(f"{output_dir}/responses_data.jsonl", "w", encoding="utf-8") as f:
        for item in responses_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    inference_time = time.time() - start_time
    log.debug(f"Total Inference Time (multiple): {inference_time:.2f}s")

    return responses_data
