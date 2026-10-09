"""Bounded, chronological replacement for the legacy growing feature histories."""

from collections import deque
import numpy as np


def as_observation(values) -> np.ndarray:
    try:
        if np.iscomplexobj(values):
            raise ValueError("Complex values are not real feature observations")
        array = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("An observation must contain five finite real feature values") from error
    if array.shape != (5,):
        raise ValueError(f"An observation must have shape (5,), received {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("An observation must contain only finite feature values")
    return array


class TemporalBuffer:
    def __init__(self, capacity=16):
        if type(capacity) is not int or capacity not in (16, 32):
            raise ValueError('Temporal buffer capacity must be 16 or 32')
        self._observations = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._observations)

    @property
    def ready(self) -> bool:
        return len(self) == self.capacity

    @property
    def capacity(self) -> int:
        return self._observations.maxlen

    @property
    def feature_storage_bytes(self) -> int:
        """Payload held by the buffer, excluding fixed Python/deque overhead."""
        return sum(observation.nbytes for observation in self._observations)

    def append(self, values):
        self._observations.append(as_observation(values))
        assert len(self) <= self.capacity

    def clear(self):
        self._observations.clear()

    def window(self) -> np.ndarray:
        if not self.ready:
            raise RuntimeError(f"Temporal buffer is warming up: {len(self)}/{self.capacity} observations")
        # Same array construction/transposition as legacy inference.py.
        sequence = np.transpose(np.array(self._observations), (1, 0))
        assert sequence.shape == (5, self.capacity)
        return sequence
