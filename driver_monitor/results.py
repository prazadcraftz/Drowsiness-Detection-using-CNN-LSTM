"""Per-call snapshots; raw model classes only, without alert or smoothing policy."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ObservationResult:
    face_detected: bool
    raw_features: np.ndarray | None
    normalized_features: np.ndarray | None
    sequence_fill: int
    ready_for_inference: bool
    reset_occurred: bool = False
    timestamp_seconds: float | None = None
    sampled: bool = True
    sample_due: bool = True
    gap_seconds: float | None = None
    gap_reset: bool = False
    effective_observation_interval: float | None = None
    missed_deadlines: int = 0

    @property
    def sequence_capacity(self) -> int:
        # Legacy serialized observation contract remains byte-for-byte stable.
        return getattr(self, '_sequence_capacity', 16)

    @property
    def warming_up(self) -> bool:
        return self.sequence_fill < self.sequence_capacity


@dataclass(frozen=True)
class InferenceResult:
    # None when infer_sequence() receives a complete normalized sequence directly.
    observation: ObservationResult | None
    input_tensor_shape: tuple[int, ...] | None = None
    sequence: np.ndarray | None = None
    input_tensor: np.ndarray | None = None
    logits: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    raw_class_id: int | None = None
    raw_class_name: str | None = None
    confidence: float | None = None
