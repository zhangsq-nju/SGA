import numpy as np
from scipy import stats
from typing import List

from .ue_metric import UEMetric


class SpearmanRankCorrelation(UEMetric):
    """Spearman correlation between uncertainty and regression error."""

    def __str__(self):
        return "Spearman"

    def __call__(self, estimator: List[float], target: List[float]) -> float:
        ue = np.asarray(estimator, dtype=np.float64)
        err = np.asarray(target, dtype=np.float64)
        return float(stats.spearmanr(ue, err).correlation)
