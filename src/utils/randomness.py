import random

import numpy as np
import torch


def resolve_seed(cfg, default=455555):
    if cfg is None:
        return int(default)
    if cfg.get("seed", None) is not None:
        return int(cfg.get("seed"))
    multiple_cfg = cfg.get("multiple_cfg", None)
    if multiple_cfg is not None and multiple_cfg.get("seed", None) is not None:
        return int(multiple_cfg.get("seed"))
    once_cfg = cfg.get("once_cfg", None)
    if once_cfg is not None and once_cfg.get("seed", None) is not None:
        return int(once_cfg.get("seed"))
    return int(default)


def set_global_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        try:
            torch.mps.manual_seed(seed)
        except RuntimeError:
            pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_from_cfg(cfg, offset=0):
    seed = resolve_seed(cfg) + int(offset)
    set_global_seed(seed)
    return seed
