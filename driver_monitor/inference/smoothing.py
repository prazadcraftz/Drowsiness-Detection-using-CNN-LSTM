"""Bounded raw probability evidence; no model calls or state/alert decisions."""

from collections import deque
from dataclasses import dataclass
import numpy as np

from driver_monitor.config import CLASS_MAPPING, PREDICTION_HISTORY_SIZE

PROBABILITY_SUM_TOLERANCE = 1e-6
SMOOTHING_MODES = ("probability_average", "majority_vote")


def probability_vector(values):
    array = np.asarray(values)
    if array.shape != (3,) or array.dtype.kind not in "fiu":
        raise ValueError("Probabilities must be a real numeric vector with shape (3,)")
    array = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    if abs(float(array.sum()) - 1) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError("Probability vector must sum to 1 within tolerance 1e-6")
    return array


@dataclass(frozen=True)
class SmoothedPrediction:
    probabilities: np.ndarray
    class_id: int
    class_name: str
    confidence: float
    history_fill: int


class PredictionSmoother:
    def __init__(self, history_size=PREDICTION_HISTORY_SIZE, mode="probability_average"):
        if type(history_size) is not int or history_size < 1:
            raise ValueError("prediction_history_size must be an integer >= 1")
        if mode not in SMOOTHING_MODES:
            raise ValueError(f"smoothing_mode must be one of {SMOOTHING_MODES}")
        self.mode = mode
        self._history = deque(maxlen=history_size)

    def __len__(self):
        return len(self._history)

    @property
    def capacity(self):
        return self._history.maxlen

    @property
    def storage_bytes(self):
        return sum(vector.nbytes for vector in self._history)

    def clear(self):
        self._history.clear()

    @property
    def current(self):
        if not self._history:
            return None
        mean = np.mean(np.array(self._history), axis=0)
        candidates = np.arange(3)
        if self.mode == "majority_vote":
            votes = np.bincount([int(np.argmax(vector)) for vector in self._history], minlength=3)
            candidates = np.flatnonzero(votes == votes.max())
        # NumPy's first maximum resolves final exact ties toward the smaller ID.
        winner = int(candidates[np.argmax(mean[candidates])])
        return SmoothedPrediction(mean.copy(), winner, CLASS_MAPPING[winner], float(mean[winner]), len(self))

    def update(self, probabilities):
        self._history.append(probability_vector(probabilities))
        return self.current
