"""One-time loading and exact per-feature legacy min/max arithmetic."""

from pathlib import Path
import numpy as np

from .temporal_buffer import as_observation


class Normalizer:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        try:
            ranges = np.load(self.path, allow_pickle=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"Normalization asset does not exist: {self.path}") from None
        except (ValueError, OSError) as error:
            raise ValueError(f"Unable to load normalization asset {self.path}: {error}") from error
        if not isinstance(ranges, np.ndarray):
            ranges.close()
            raise ValueError(f"Normalization asset {self.path} must be one NumPy array with shape (5, 2)")
        if ranges.shape != (5, 2):
            raise ValueError(f"Normalization asset {self.path} must have shape (5, 2), received {ranges.shape}")
        if (not np.issubdtype(ranges.dtype, np.number) or np.iscomplexobj(ranges)
                or not np.isfinite(ranges).all()):
            raise ValueError(f"Normalization asset {self.path} must contain finite real numbers")
        if not np.all(ranges[:, 1] > ranges[:, 0]):
            raise ValueError(f"Normalization asset {self.path} requires every maximum - minimum > 0")
        self._ranges = ranges.copy()
        self._ranges.flags.writeable = False

    @property
    def ranges(self) -> np.ndarray:
        return self._ranges.copy()

    def normalize(self, values) -> np.ndarray:
        values = as_observation(values)
        # Scalar NumPy arithmetic in canonical feature order, no clip or epsilon.
        return np.array([(values[index] - self._ranges[index][0]) /
                         (self._ranges[index][1] - self._ranges[index][0])
                         for index in range(5)])
