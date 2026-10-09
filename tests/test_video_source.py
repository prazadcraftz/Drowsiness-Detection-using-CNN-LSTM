"""Source acquisition semantics without requiring a camera."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from driver_monitor.vision import video_source as module
from driver_monitor.vision.video_source import SourceError, VideoSource, parse_source


class FakeCapture:
    def __init__(self, *, opened=True, count=2, fps=25.0, timestamps=(40.0, 80.0), frames=None):
        self.opened = opened
        self.count = count
        self.fps = fps
        self.timestamps = timestamps
        self.frames = [np.zeros((24, 32, 3), np.uint8) for _ in range(2)] if frames is None else frames
        self.index = 0
        self.releases = 0

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: 32, cv2.CAP_PROP_FRAME_HEIGHT: 24,
                cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: self.count,
                cv2.CAP_PROP_POS_MSEC: self.timestamps[max(0, self.index - 1)]}[prop]

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        self.releases += 1


@pytest.mark.parametrize("value,expected", [("0", 0), ("1", 1), ("2", 2), ("12", 12), (2, 2)])
def test_camera_parsing(value, expected):
    assert parse_source(value) == expected


@pytest.mark.parametrize("value", ["video.mp4", "123.mp4", "videos/1.mp4", "./123", Path("123"), "1.0"])
def test_file_parsing(value):
    assert parse_source(value) == Path(value).resolve()


@pytest.mark.parametrize("value", [-1, "-2", True, "", " ", None])
def test_invalid_source(value):
    with pytest.raises(ValueError):
        parse_source(value)


def test_missing_file_and_directory_before_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda _: pytest.fail("Capture should not open"))
    with pytest.raises(SourceError, match="does not exist"):
        VideoSource(tmp_path / "missing.mp4")
    with pytest.raises(SourceError, match="not a file"):
        VideoSource(tmp_path)


@pytest.mark.parametrize("source", [0, "file"])
def test_unavailable_source_releases(tmp_path, monkeypatch, source):
    fake = FakeCapture(opened=False)
    path = tmp_path / "bad.avi"
    path.touch()
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda _: fake)
    with pytest.raises(SourceError, match="Cannot open"):
        VideoSource(0 if source == 0 else path)
    assert fake.releases == 1


def test_file_metadata_timestamps_indices_eof_cleanup(tmp_path, monkeypatch):
    path = tmp_path / "123.mp4"
    path.touch()
    fake = FakeCapture()
    args = []
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda value: args.append(value) or fake)
    with VideoSource(path) as source:
        assert source.source_type == "file" and source.camera_index is None
        assert source.file_path == path.resolve()
        assert (source.width, source.height, source.nominal_fps, source.total_frames, source.duration) == (32, 24, 25, 2, .08)
        first, second = source.read(), source.read()
        assert (first.frame_index, second.frame_index) == (0, 1)
        assert (first.timestamp_seconds, second.timestamp_seconds) == (.04, .08)
        assert source.read().eof and source.read().eof
    source.close()
    assert fake.releases == 1
    assert args == [str(path.resolve())]
    with pytest.raises(SourceError, match="closed"):
        source.read()


def test_camera_relative_monotonic_clock_and_failure(monkeypatch):
    fake = FakeCapture()
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda index: fake if index == 2 else pytest.fail())
    clock = iter([50.0, 50.1, 50.3])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(clock))
    with VideoSource("2") as source:
        assert source.total_frames is None and source.duration is None
        assert source.camera_index == 2 and source.file_path is None
        assert source.read().timestamp_seconds == pytest.approx(.1)
        assert source.read().timestamp_seconds == pytest.approx(.3)
        with pytest.raises(SourceError, match="disconnected"):
            source.read()
    assert fake.releases == 1


@pytest.mark.parametrize("count,frames,message", [(3, None, "unexpectedly"), (0, [], "unexpectedly"),
                                                    (2, [None], "Invalid decoded")])
def test_decode_failures_release(tmp_path, monkeypatch, count, frames, message):
    path = tmp_path / "broken.avi"
    path.touch()
    fake = FakeCapture(count=count, frames=frames)
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda _: fake)
    with VideoSource(path) as source:
        with pytest.raises(SourceError, match=message):
            for _ in range(4):
                source.read()
    assert fake.releases == 1


def test_unknown_metadata_eof_and_invalid_timestamp(tmp_path, monkeypatch):
    path = tmp_path / "unknown.avi"
    path.touch()
    fake = FakeCapture(count=float("nan"), fps=0, timestamps=(float("nan"), 0))
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda _: fake)
    with VideoSource(path) as source:
        assert source.total_frames is None and source.nominal_fps is None and source.duration is None
        assert source.read().timestamp_seconds is None
        assert source.read().timestamp_seconds == 0
        assert source.read().eof
    assert fake.releases == 1


def test_context_exception_releases(monkeypatch):
    fake = FakeCapture()
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda _: fake)
    with pytest.raises(RuntimeError, match="test"):
        with VideoSource(0):
            raise RuntimeError("test")
    assert fake.releases == 1
