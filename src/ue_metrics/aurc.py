import numpy as np
from typing import List
from sklearn.metrics import auc

from .ue_metric import UEMetric


class AURC(UEMetric):
    """Area under risk-coverage curve for regression errors (smaller is better)."""

    def __init__(
        self,
        error_threshold: float = None,
        excess: bool = False,
        normalized: bool = False,
        risk_type: str = None,
    ):
        super().__init__()
        # `risk_type` is kept only for backward compatibility and is ignored.
        self.error_threshold = error_threshold
        self.excess = excess
        self.normalized = normalized

    def __str__(self):
        if self.normalized:
            return "NEAURC"
        if self.excess:
            return "EAURC"
        return "AURC"

    @staticmethod
    def _compute_aurc_from_sorted_losses(sorted_losses: np.ndarray) -> float:
        n_samples = len(sorted_losses)
        if n_samples <= 1:
            return 0.0
        coverages = np.arange(1, n_samples + 1, dtype=np.float64) / n_samples
        risks = np.cumsum(sorted_losses, dtype=np.float64) / np.arange(1, n_samples + 1)
        return float(auc(coverages, risks))

    @staticmethod
    def _compute_random_aurc(losses: np.ndarray) -> float:
        n_samples = len(losses)
        if n_samples <= 1:
            return 0.0
        coverages = np.arange(1, n_samples + 1, dtype=np.float64) / n_samples
        risks = np.full(n_samples, float(np.mean(losses)), dtype=np.float64)
        return float(auc(coverages, risks))

    def __call__(self, estimator: List[float], target: List[float]) -> float:
        uncertainties = np.asarray(estimator, dtype=np.float64)
        errors = np.asarray(target, dtype=np.float64)

        if uncertainties.shape != errors.shape:
            raise ValueError(
                f"Uncertainty/error shape mismatch: {uncertainties.shape} vs {errors.shape}"
            )

        losses = errors
        sorted_indices = np.argsort(uncertainties, kind="stable")
        sorted_losses = losses[sorted_indices]
        aurc_score = self._compute_aurc_from_sorted_losses(sorted_losses)

        if self.excess or self.normalized:
            oracle_sorted_losses = np.sort(losses)
            oracle_aurc = self._compute_aurc_from_sorted_losses(oracle_sorted_losses)

        if self.normalized:
            random_aurc = self._compute_random_aurc(losses)
            denominator = random_aurc - oracle_aurc
            if np.isclose(denominator, 0.0):
                return 0.0
            return max(0.0, (aurc_score - oracle_aurc) / denominator)

        if self.excess:
            return max(0.0, aurc_score - oracle_aurc)

        return aurc_score


class EAURC(AURC):
    """Excess area under the risk-coverage curve."""

    def __init__(self, **kwargs):
        super().__init__(excess=True, **kwargs)


class NEAURC(AURC):
    """Normalized excess area under the risk-coverage curve."""

    def __init__(self, **kwargs):
        super().__init__(normalized=True, **kwargs)
