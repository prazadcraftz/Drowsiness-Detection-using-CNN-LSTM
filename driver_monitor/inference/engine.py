"""Reusable CPU inference preserving the Phase 2 raw temporal model behavior."""

from __future__ import annotations

import numpy as np
import torch

from model import CNN_LSTM
from driver_monitor.config import RuntimeConfig
from driver_monitor.results import InferenceResult, ObservationResult
from driver_monitor.vision.face_detector import FaceDetector
from .normalizer import Normalizer
from .model_profile import get_profile, build_model
from .temporal_buffer import TemporalBuffer, as_observation
from .sampling import TemporalRuntime, TemporalTimestampError

_UNTIMED = object()


class DrowsinessInferenceEngine:
    def __init__(self, config: RuntimeConfig | None = None, *, initialize_detector: bool = True):
        self.config = config if config is not None else RuntimeConfig.from_ini()
        self.profile = get_profile(self.config.model_profile)
        canonical_paths = self.profile.paths()
        if (self.config.model_path, self.config.normalization_path) == canonical_paths:
            self.profile.validate_assets()
        self.normalizer = Normalizer(self.config.normalization_path)
        self.buffer = TemporalBuffer(self.config.sequence_length)
        self.temporal_runtime = TemporalRuntime(self.config.target_model_fps,
                                                self.config.max_face_gap_seconds,
                                                enabled=self.config.enable_temporal_sampling)
        self.detector = None
        self._closed = False
        try:
            self.model = build_model(self.profile, self.config.model_settings(),
                                     legacy_constructor=CNN_LSTM)
            self.model.load_state_dict(torch.load(self.config.model_path,
                                                 map_location="cpu", weights_only=True))
            self.model.eval()
        except Exception as error:
            label = 'CNN-LSTM' if self.profile.name == 'legacy' else 'LSTM-only'
            raise RuntimeError(f"Unable to load {label} checkpoint {self.config.model_path}: {error}") from error
        if initialize_detector:
            self.detector = FaceDetector(self.config.face_model_path)

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("DrowsinessInferenceEngine is closed")

    def process_frame(self, frame: np.ndarray, *, timestamp_seconds=_UNTIMED) -> InferenceResult:
        self._ensure_open()
        if self.detector is None:
            raise RuntimeError("Frame processing requires detector initialization; this engine was created for features/sequences only")
        # A failed timestamp is a terminal runtime/source error for the caller.
        if timestamp_seconds is _UNTIMED and self.temporal_runtime.sampler.last_timestamp is not None:
            raise TemporalTimestampError("Cannot mix untimed observations into an active timed stream")
        landmarks = self.detector.detect(frame)
        decision = self._temporal_decision(timestamp_seconds, landmarks is not None)
        if landmarks is None:
            # Compatibility: no observation, no prediction and no buffer modification.
            return InferenceResult(self._observation(False, None, None, decision))
        if decision is not None and not decision.sampled:
            return InferenceResult(self._observation(True, None, None, decision))
        from driver_monitor.vision.features import extract_features
        return self._append_features(extract_features(frame, landmarks), decision)

    def process_features(self, raw_features, *, timestamp_seconds=_UNTIMED) -> InferenceResult:
        """Append one supplied valid observation, assuming a face was available.

        This path bypasses vision, not normalization or temporal behavior.
        The engine retains no growing prediction/feature history.
        """
        self._ensure_open()
        raw = as_observation(raw_features)
        decision = self._temporal_decision(timestamp_seconds, True)
        if decision is not None and not decision.sampled:
            return InferenceResult(self._observation(True, raw, None, decision))
        return self._append_features(raw, decision)

    def _temporal_decision(self, timestamp, face_detected):
        if timestamp is _UNTIMED:
            if self.temporal_runtime.sampler.last_timestamp is not None:
                raise TemporalTimestampError("Cannot mix untimed observations into an active timed stream")
            return None
        # Unknown-time compatibility observations cannot precede a timed window.
        entering_timed = self.temporal_runtime.sampler.last_timestamp is None and len(self.buffer) > 0
        decision = self.temporal_runtime.observe(timestamp, face_detected=face_detected)
        if entering_timed or decision.gap_reset:
            self.buffer.clear()
        if entering_timed and not decision.gap_reset:
            from dataclasses import replace
            decision = replace(decision, gap_reset=True)
        return decision

    def _observation(self, face_detected, raw, normalized, decision):
        sampled = face_detected if decision is None else decision.sampled
        metadata = {} if decision is None else dict(
            timestamp_seconds=decision.timestamp_seconds, sample_due=decision.sample_due,
            gap_seconds=decision.gap_seconds, gap_reset=decision.gap_reset,
            reset_occurred=decision.gap_reset,
            effective_observation_interval=decision.effective_observation_interval,
            missed_deadlines=decision.missed_deadlines)
        observation = ObservationResult(face_detected, raw, normalized, len(self.buffer),
                                        sampled and self.buffer.ready, sampled=sampled, **metadata)
        if self.buffer.capacity != 16:
            object.__setattr__(observation, '_sequence_capacity', self.buffer.capacity)
        return observation

    def _append_features(self, raw_features, decision):
        raw = as_observation(raw_features)
        normalized = self.normalizer.normalize(raw)
        self.buffer.append(normalized)
        observation = self._observation(True, raw, normalized, decision)
        if not self.buffer.ready:
            return InferenceResult(observation)
        return self.infer_sequence(self.buffer.window(), observation=observation)

    def infer_sequence(self, sequence, *, observation: ObservationResult | None = None) -> InferenceResult:
        """Infer a supplied normalized (5,N) window without changing the buffer."""
        self._ensure_open()
        if np.iscomplexobj(sequence):
            raise ValueError("Model sequence must contain finite real values")
        sequence = np.asarray(sequence, dtype=np.float64)
        expected = (5, self.buffer.capacity)
        if sequence.shape != expected:
            raise ValueError(f"Model sequence must have shape {expected}, received {sequence.shape}")
        if not np.isfinite(sequence).all():
            raise ValueError("Model sequence must contain only finite values")
        with torch.inference_mode():
            input_data = torch.tensor(sequence, dtype=torch.float32).unsqueeze(axis=0)
            assert input_data.shape == (1, *expected)
            logits = self.model(input_data)
            if logits.shape != (1, 3) or not torch.isfinite(logits).all():
                raise RuntimeError("Model must return finite logits with shape (1, 3)")
            probabilities = torch.nn.Softmax(dim=1)(logits)
            raw_class_id = torch.argmax(probabilities, 1).item()
            confidence = probabilities[0, raw_class_id].item()
        return InferenceResult(observation=observation,
                               input_tensor_shape=tuple(input_data.shape),
                               sequence=sequence.copy(),
                               input_tensor=input_data.cpu().numpy().copy(),
                               logits=logits.cpu().numpy().copy(),
                               probabilities=probabilities.cpu().numpy().copy(),
                               raw_class_id=raw_class_id,
                               raw_class_name=self.config.class_mapping[raw_class_id],
                               confidence=confidence)

    def close(self):
        if not self._closed:
            self._closed = True
            if self.detector is not None:
                self.detector.close()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
