"""Single bounded worker for local audio; no playback in the inference loop."""

import math
from numbers import Real
import sys
from threading import Condition, Thread

from driver_monitor.inference.state_machine import DriverState

ALERT_COOLDOWN_SECONDS = 5.0  # Application anti-spam default, not medical.


def windows_beep():
    if sys.platform != "win32":
        raise RuntimeError("Local beep backend requires Windows; use --no-audio on other platforms")
    import winsound
    winsound.Beep(1200, 300)  # Bounded playback occurs only on the worker.


class AlertManager:
    def __init__(self, *, enabled=True, cooldown_seconds=ALERT_COOLDOWN_SECONDS, backend=None):
        if (isinstance(cooldown_seconds, bool) or not isinstance(cooldown_seconds, Real)
                or not math.isfinite(cooldown_seconds) or cooldown_seconds < 0):
            raise ValueError("alert_cooldown_seconds must be finite and >= 0")
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self._backend = backend or windows_beep
        self._condition = Condition()
        self._worker = None
        self._pending = self._playing = self._closed = False
        self._last_state = None
        self._last_timestamp = self.last_trigger_timestamp = None
        self.audible_trigger_count = self.suppressed_trigger_count = self.disabled_trigger_count = 0
        self.completed_playback_count = self.failed_playback_count = 0
        self.last_error = None

    def update(self, decision):
        driver = decision.driver
        if driver is None:
            return
        timestamp = driver.timestamp_seconds
        with self._condition:
            if self._closed:
                raise RuntimeError("AlertManager is closed")
            if (isinstance(timestamp, bool) or not isinstance(timestamp, Real) or not math.isfinite(timestamp)
                    or timestamp < 0 or (self._last_timestamp is not None and timestamp <= self._last_timestamp)):
                raise ValueError("Alert timestamp must be finite, nonnegative and strictly increasing")
            entering = driver.state == DriverState.ALERT and self._last_state != DriverState.ALERT
            self._last_state, self._last_timestamp = driver.state, timestamp
            if not entering:
                return
            if not self.enabled:
                self.disabled_trigger_count += 1
                return
            if ((self.last_trigger_timestamp is not None and timestamp - self.last_trigger_timestamp < self.cooldown_seconds)
                    or self._pending or self._playing):
                self.suppressed_trigger_count += 1
                return
            # Only one worker and one pending request, regardless of run length.
            if self._worker is None:
                self._worker = Thread(target=self._work, name="driver-monitor-audio", daemon=True)
                try:
                    self._worker.start()
                except Exception as error:
                    self._worker = None
                    self.failed_playback_count += 1
                    self.last_error = str(error)[:500]
                    return
            self._pending = True
            self.last_trigger_timestamp = timestamp
            self.audible_trigger_count += 1
            self._condition.notify()

    def _work(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending or self._closed)
                if not self._pending:
                    return
                self._pending, self._playing = False, True
            error = None
            try:
                self._backend()
            except Exception as failure:
                error = str(failure)[:500]
            with self._condition:
                self._playing = False
                if error is None:
                    self.completed_playback_count += 1
                else:
                    self.last_error = error
                    self.failed_playback_count += 1
                self._condition.notify_all()

    def snapshot(self, timestamp_seconds=None):
        with self._condition:
            remaining = 0.0
            if timestamp_seconds is not None and self.last_trigger_timestamp is not None:
                remaining = max(0.0, self.cooldown_seconds - (timestamp_seconds - self.last_trigger_timestamp))
            return {"enabled": self.enabled, "currently_playing": self._playing, "pending": self._pending,
                    "closed": self._closed, "last_trigger_timestamp": self.last_trigger_timestamp,
                    "audible_trigger_count": self.audible_trigger_count,
                    "suppressed_trigger_count": self.suppressed_trigger_count,
                    "disabled_trigger_count": self.disabled_trigger_count,
                    "completed_playback_count": self.completed_playback_count,
                    "failed_playback_count": self.failed_playback_count, "last_error": self.last_error,
                    "cooldown_seconds": self.cooldown_seconds, "cooldown_remaining_seconds": remaining}

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            worker = self._worker
        if worker is not None:
            worker.join(timeout=2.0)  # Default backend finishes within 300 ms.
            if worker.is_alive():
                with self._condition:
                    self.last_error = "Audio backend did not finish within shutdown timeout"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
