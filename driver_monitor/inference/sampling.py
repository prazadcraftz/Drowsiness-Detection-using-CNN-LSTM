"""Bounded timestamp policy; never fabricate observations or pace playback."""

from dataclasses import dataclass
import math
from numbers import Real
from driver_monitor.config import TARGET_MODEL_FPS


class TemporalTimestampError(ValueError):
    """Invalid source time: stop the stream rather than mix temporal histories."""


def positive_seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return float(value)


@dataclass(frozen=True)
class SamplingDecision:
    timestamp_seconds: float
    sample_due: bool
    missed_deadlines: int
    next_sample_time: float | None


class TemporalSampler:
    def __init__(self, target_model_fps=TARGET_MODEL_FPS, *, enabled=True):
        self.target_model_fps = positive_seconds(target_model_fps, "target_model_fps")
        if not math.isfinite(1 / self.target_model_fps):
            raise ValueError("target_model_fps must have a finite representable interval")
        if type(enabled) is not bool:
            raise ValueError("enabled must be a bool")
        self.enabled = enabled
        self.reset()

    def reset(self):
        self._start = None
        self._last_timestamp = None
        self._next_tick = 0

    @property
    def last_timestamp(self):
        return self._last_timestamp

    @property
    def next_sample_time(self):
        return None if self._start is None or not self.enabled else self._start + self._next_tick / self.target_model_fps

    def consider(self, timestamp_seconds) -> SamplingDecision:
        if (isinstance(timestamp_seconds, bool) or not isinstance(timestamp_seconds, Real)
                or not math.isfinite(timestamp_seconds) or timestamp_seconds < 0):
            raise TemporalTimestampError("Source timestamp must be finite, nonnegative seconds; missing timestamps are not accepted in timed mode")
        timestamp = float(timestamp_seconds)
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise TemporalTimestampError("Source timestamps must strictly increase; repeated/backward timestamps are not accepted")
        start = timestamp if self._start is None else self._start
        ticks = (timestamp - start) * self.target_model_fps if self.enabled else 0.0
        if not math.isfinite(ticks):
            raise TemporalTimestampError("Source timestamp span is too large for the configured target cadence")
        # Tolerance covers float arithmetic only: 1e-9 ticks (~5e-11 s at 20 Hz).
        tolerance = max(1e-9, 8 * math.ulp(ticks))
        due = not self.enabled or ticks >= self._next_tick - tolerance
        missed = 0
        if due and self.enabled:
            current_tick = math.floor(ticks + tolerance)
            missed = max(0, current_tick - self._next_tick)
            self._next_tick = current_tick + 1
        self._start = start
        self._last_timestamp = timestamp
        return SamplingDecision(timestamp, due, missed, self.next_sample_time)


@dataclass(frozen=True)
class TemporalDecision:
    timestamp_seconds: float
    sample_due: bool
    sampled: bool
    gap_seconds: float | None
    gap_reset: bool
    effective_observation_interval: float | None
    missed_deadlines: int


class TemporalRuntime:
    """Scalar timing state shared by frames and supplied feature observations."""
    def __init__(self, target_model_fps, max_face_gap_seconds, *, enabled=True):
        self.sampler = TemporalSampler(target_model_fps, enabled=enabled)
        self.max_face_gap_seconds = positive_seconds(max_face_gap_seconds, "max_face_gap_seconds")
        self.last_valid_observation_timestamp = None

    def observe(self, timestamp_seconds, *, face_detected: bool) -> TemporalDecision:
        decision = self.sampler.consider(timestamp_seconds)
        previous = self.last_valid_observation_timestamp
        gap = None if previous is None else decision.timestamp_seconds - previous
        # Tiny tolerance avoids treating decimal equality as an excessive gap.
        tolerance = max(1e-12, 8 * math.ulp(decision.timestamp_seconds))
        reset = gap is not None and gap > self.max_face_gap_seconds + tolerance
        if reset:
            self.last_valid_observation_timestamp = None
        sampled = face_detected and decision.sample_due
        interval = gap if sampled and not reset else None
        if sampled:
            self.last_valid_observation_timestamp = decision.timestamp_seconds
        # Keep the original sampling timeline on reset: no jitter/reanchoring.
        return TemporalDecision(decision.timestamp_seconds, decision.sample_due, sampled,
                                gap, reset, interval, decision.missed_deadlines)
