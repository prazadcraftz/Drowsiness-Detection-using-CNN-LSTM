"""Frame acquisition only; timestamps never control model observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import time

import cv2
import numpy as np


class SourceError(RuntimeError):
    """A source could not open or unexpectedly stopped decoding."""


@dataclass(frozen=True)
class FramePacket:
    frame: np.ndarray | None
    frame_index: int
    timestamp_seconds: float | None
    source_type: str
    eof: bool = False


def parse_source(value: int | str | Path) -> int | Path:
    """Only plain ASCII decimal CLI strings are camera indices.

    Use ./123 or Path('123') for a file whose entire name is numeric.
    """
    if isinstance(value, bool):
        raise ValueError("Source must be a nonnegative camera index or local file path")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("Camera index must be nonnegative")
        return value
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("Source cannot be empty")
        if re.fullmatch(r"[0-9]+", value):
            return int(value)
        if re.fullmatch(r"-[0-9]+", value):
            raise ValueError("Camera index must be nonnegative; prefix a numeric filename with ./")
    if not isinstance(value, (str, Path)):
        raise ValueError("Source must be a nonnegative camera index or local file path")
    return Path(value).expanduser().resolve()


def _positive(value: float) -> float | None:
    return value if math.isfinite(value) and value > 0 else None


class VideoSource:
    def __init__(self, source: int | str | Path):
        parsed = parse_source(source)
        self.camera_index = parsed if isinstance(parsed, int) else None
        self.file_path = parsed if isinstance(parsed, Path) else None
        self.source_type = "webcam" if self.file_path is None else "file"
        self.source_name = f"Camera {parsed}" if self.file_path is None else str(self.file_path)
        if self.file_path is not None:
            if not self.file_path.exists():
                raise SourceError(f"Video file does not exist: {self.file_path}")
            if not self.file_path.is_file():
                raise SourceError(f"Video source is not a file: {self.file_path}")
        self._capture = cv2.VideoCapture(parsed if isinstance(parsed, int) else str(parsed))
        self._closed = False
        self._eof = False
        self._next_index = 0
        try:
            if not self._capture.isOpened():
                raise SourceError(f"Cannot open {self.source_type} source: {self.source_name}")
            self.width = _positive(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = _positive(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.nominal_fps = _positive(self._capture.get(cv2.CAP_PROP_FPS))
            count = _positive(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.total_frames = int(count) if self.source_type == "file" and count is not None and count >= 1 else None
            self.duration = (self.total_frames / self.nominal_fps
                             if self.total_frames is not None and self.nominal_fps is not None else None)
            self._started = time.perf_counter()
        except BaseException:
            self.close()
            raise

    def read(self) -> FramePacket:
        if self._closed:
            raise SourceError("VideoSource is closed")
        if self._eof:
            return FramePacket(None, self._next_index, None, self.source_type, eof=True)
        try:
            ok, frame = self._capture.read()
            if not ok:
                if self.source_type == "webcam":
                    raise SourceError(f"Camera disconnected or frame read failed: {self.source_name}")
                if self._next_index == 0 or (self.total_frames is not None and self._next_index < self.total_frames):
                    raise SourceError(f"Video decoding stopped unexpectedly at frame {self._next_index}: {self.source_name}")
                self._eof = True
                return FramePacket(None, self._next_index, None, self.source_type, eof=True)
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0 or frame.dtype != np.uint8:
                raise SourceError(f"Invalid decoded BGR frame {self._next_index}: {self.source_name}")
            if self.source_type == "webcam":
                timestamp = time.perf_counter() - self._started
            else:
                milliseconds = self._capture.get(cv2.CAP_PROP_POS_MSEC)
                timestamp = milliseconds / 1000 if math.isfinite(milliseconds) and milliseconds >= 0 else None
            packet = FramePacket(frame, self._next_index, timestamp, self.source_type)
            self._next_index += 1
            return packet
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self._closed:
            self._closed = True
            self._capture.release()

    def __enter__(self):
        if self._closed:
            raise SourceError("VideoSource is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
