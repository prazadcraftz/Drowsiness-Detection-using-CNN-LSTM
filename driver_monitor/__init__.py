"""Local, raw CNN-LSTM inference with the frozen legacy model contract."""

from .config import RuntimeConfig
from .results import InferenceResult, ObservationResult

__all__ = ["RuntimeConfig", "ObservationResult", "InferenceResult", "DrowsinessInferenceEngine"]


def __getattr__(name):
    if name == "DrowsinessInferenceEngine":
        from .inference.engine import DrowsinessInferenceEngine
        return DrowsinessInferenceEngine
    raise AttributeError(name)
