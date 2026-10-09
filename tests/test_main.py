"""Application cleanup, shared engine, output saving and real-video parity."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import main
from driver_monitor.results import InferenceResult, ObservationResult
from driver_monitor.vision.video_source import FramePacket, SourceError

ROOT = Path(__file__).resolve().parents[1]


class FakeSource:
    instances = []

    def __init__(self, source):
        self.source_name = str(source)
        self.file_path = None if isinstance(source, int) else Path(source).resolve()
        self.nominal_fps = 25.0
        self.index = 0
        self.closed = 0
        self.frame = np.zeros((160, 320, 3), np.uint8)
        self.instances.append(self)

    def read(self):
        index = self.index
        self.index += 1
        return FramePacket(self.frame, index, index / 25, "webcam" if self.file_path is None else "file", index >= 3)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed += 1


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.closed = 0

    def process_frame(self, frame):
        self.calls.append(frame.copy())
        index = len(self.calls)
        if index == 1:
            return InferenceResult(ObservationResult(True, None, None, 1, False))
        if index == 2:
            return InferenceResult(ObservationResult(False, None, None, 1, False))
        return InferenceResult(ObservationResult(True, None, None, 16, True),
                               raw_class_id=2, raw_class_name="Eye Close", confidence=.913)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed += 1


class FakeWriter:
    def __init__(self, opened=True):
        self.opened = opened
        self.frames = []
        self.releases = 0

    def isOpened(self):
        return self.opened

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.releases += 1


@pytest.fixture
def app(monkeypatch):
    FakeSource.instances.clear()
    engine = FakeEngine()
    monkeypatch.setattr(main, "VideoSource", FakeSource)
    monkeypatch.setattr(main, "_create_engine", lambda: engine)
    for name in ("imshow", "waitKey", "destroyAllWindows"):
        monkeypatch.setattr(main.cv2, name, lambda *args: pytest.fail("Headless called GUI"))
    return engine


@pytest.mark.parametrize("source", [0, 1, 2, "video.mp4"])
def test_shared_engine_and_no_display(app, source):
    summary = main.run(source, no_display=True)
    assert (summary.frames_decoded, summary.valid_observations, summary.predictions,
            summary.warming_up_observations, summary.no_face_frames) == (3, 2, 1, 1, 1)
    assert summary.stop_reason == "eof"
    assert len(app.calls) == 3 and app.closed == 1 and FakeSource.instances[0].closed == 1
    assert all(not frame.any() for frame in app.calls)


def test_invalid_source_before_engine(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(main, "_create_engine", lambda: pytest.fail("Invalid source initialized engine"))
    assert main.main(["--source", str(tmp_path / "missing.avi"), "--no-display"]) == 1
    assert "does not exist" in capsys.readouterr().err


def test_corrupt_first_read_before_engine(tmp_path, monkeypatch):
    class BrokenSource(FakeSource):
        def read(self):
            raise SourceError("first decode failed")
    monkeypatch.setattr(main, "VideoSource", BrokenSource)
    monkeypatch.setattr(main, "_create_engine", lambda: pytest.fail("Decoder failure initialized engine"))
    with pytest.raises(SourceError):
        main.run(tmp_path / "bad.avi", no_display=True)
    assert FakeSource.instances[-1].closed == 1


@pytest.mark.parametrize("key", [ord("q"), 27])
def test_keyboard_gui_cleanup(app, monkeypatch, key):
    events = []
    monkeypatch.setattr(main.cv2, "imshow", lambda *args: events.append("show"))
    monkeypatch.setattr(main.cv2, "waitKey", lambda _: key)
    monkeypatch.setattr(main.cv2, "destroyAllWindows", lambda: events.append("destroy"))
    summary = main.run(0)
    assert summary.stop_reason == "keyboard" and summary.frames_decoded == 1
    assert events == ["show", "destroy"]
    assert app.closed == FakeSource.instances[0].closed == 1


@pytest.mark.parametrize("failure", ["engine", "read", "gui", "interrupt"])
def test_exception_cleanup(app, monkeypatch, tmp_path, failure):
    writer = FakeWriter()
    monkeypatch.setattr(main.cv2, "VideoWriter", lambda *args: writer)
    def fail(*args):
        if failure == "interrupt":
            raise KeyboardInterrupt()
        raise RuntimeError("test failure")
    if failure in ("engine", "interrupt"):
        monkeypatch.setattr(app, "process_frame", fail)
    elif failure == "read":
        original = FakeSource.read
        monkeypatch.setattr(FakeSource, "read", lambda self: original(self) if self.index == 0 else fail())
    events = []
    monkeypatch.setattr(main.cv2, "imshow", fail if failure == "gui" else lambda *args: None)
    monkeypatch.setattr(main.cv2, "waitKey", lambda _: -1)
    monkeypatch.setattr(main.cv2, "destroyAllWindows", lambda: events.append("destroy"))
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else RuntimeError):
        main.run(0, output=tmp_path / "out.avi")
    assert app.closed == FakeSource.instances[0].closed == writer.releases == 1
    assert events == ["destroy"]


def test_engine_initialization_failure_releases_writer_source(app, monkeypatch, tmp_path):
    writer = FakeWriter()
    monkeypatch.setattr(main.cv2, "VideoWriter", lambda *args: writer)
    def fail():
        raise RuntimeError("engine initialization")
    monkeypatch.setattr(main, "_create_engine", fail)
    with pytest.raises(RuntimeError, match="initialization"):
        main.run(0, no_display=True, output=tmp_path / "out.avi")
    assert writer.releases == FakeSource.instances[0].closed == 1


def test_writer_failure_before_engine(app, monkeypatch, tmp_path):
    writer = FakeWriter(opened=False)
    monkeypatch.setattr(main.cv2, "VideoWriter", lambda *args: writer)
    with pytest.raises(RuntimeError, match="Cannot open output"):
        main.run(0, no_display=True, output=tmp_path / "out.avi")
    assert writer.releases == FakeSource.instances[0].closed == 1
    assert not app.calls and app.closed == 0


@pytest.mark.parametrize("output", ["out.avi", "results/nested/out.avi"])
def test_real_writer_bare_nested_and_input_fps(app, monkeypatch, tmp_path, output):
    monkeypatch.chdir(tmp_path)
    summary = main.run("input.avi", no_display=True, output=output)
    assert summary.output_fps == 25 and Path(output).is_file()
    capture = cv2.VideoCapture(str(Path(output).resolve()))
    try:
        assert capture.isOpened() and capture.get(cv2.CAP_PROP_FPS) == 25
        count = 0
        while capture.read()[0]:
            count += 1
        assert count == 3
    finally:
        capture.release()
    assert app.closed == FakeSource.instances[0].closed == 1


def test_default_filename_and_fps_fallback(app, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    original = FakeSource.__init__
    def init(self, source):
        original(self, source)
        self.nominal_fps = None
    monkeypatch.setattr(FakeSource, "__init__", init)
    summary = main.run(0, no_display=True, save_output=True)
    assert summary.output_fps == 20 and Path("model_output.avi").exists()
    assert len(app.calls) == 3  # Writer metadata does not change inference cadence.


@pytest.mark.parametrize("output", ["input.avi", "out.txt", "existing.avi"])
def test_invalid_output_releases_source(app, monkeypatch, tmp_path, output):
    monkeypatch.chdir(tmp_path)
    Path("existing.avi").touch()
    with pytest.raises(ValueError):
        main.run("input.avi", no_display=True, output=output)
    assert FakeSource.instances[0].closed == 1 and not app.calls


def test_render_states_and_raw_semantics(app, monkeypatch):
    texts = []
    monkeypatch.setattr(main.cv2, "putText", lambda image, text, *args: texts.append(text))
    source = FakeSource(0)
    for _ in range(3):
        packet = source.read()
        image = main.render_result(packet.frame, packet, app.process_frame(packet.frame), source.source_name)
        assert image is not packet.frame
    assert "WARMING UP" in texts and "NO FACE" in texts
    assert "Prediction: Eye Close" in texts and "Confidence: 91.3%" in texts
    assert not any("DROWSY" in text or "FATIGUE" in text for text in texts)


@pytest.mark.vision
@pytest.mark.parametrize("filename", ["Normal/Normal_1.avi", "Yawning/Yawning_1.avi",
                                     "Eye Close/Eye Close_60.avi", "Yawning/Yawning_60.avi"])
def test_real_application_parity(filename, frozen_phase2, monkeypatch, capsys, record_property):
    from driver_monitor.inference.engine import DrowsinessInferenceEngine
    manifest, arrays = frozen_phase2
    source = next(item for item in manifest["sources"] if item["filename"] == filename)
    # Identify archive row keys from the frozen schema, without creating new fixtures.
    frame_rows = np.arange(source["first_frame_row"], source["last_frame_row_exclusive"])
    window_rows = np.flatnonzero(arrays["source_ids"][arrays["window_frame_rows"]] == manifest["sources"].index(source))
    engine = DrowsinessInferenceEngine()
    original = engine.process_frame
    observed = {"frames": 0, "windows": 0, "valid": 0, "warm": 0, "max_difference": 0.0}
    def compare(frame):
        result = original(frame)
        row = frame_rows[observed["frames"]]
        observed["frames"] += 1
        obs = result.observation
        assert obs.face_detected == bool(arrays["face_detected"][row])
        assert obs.sequence_fill == arrays["sequence_fill"][row]
        assert (result.raw_class_id is not None) == (arrays["frame_window_indices"][row] >= 0)
        if obs.face_detected:
            observed["valid"] += 1
            for actual, expected in ((obs.raw_features, arrays["raw_features"][row]),
                                     (obs.normalized_features, arrays["normalized_features"][row])):
                np.testing.assert_array_equal(actual, expected)
            assert obs.sequence_fill == min(observed["valid"], 16)
            if result.raw_class_id is None:
                observed["warm"] += 1
        if result.raw_class_id is not None:
            window = window_rows[observed["windows"]]
            observed["windows"] += 1
            for key, actual in (("sequences", result.sequence), ("input_tensors", result.input_tensor),
                                ("logits", result.logits), ("probabilities", result.probabilities)):
                np.testing.assert_array_equal(actual, arrays[key][window])
            assert result.raw_class_id == arrays["predicted_class_ids"][window]
        return result
    monkeypatch.setattr(engine, "process_frame", compare)
    monkeypatch.setattr(main, "_create_engine", lambda: engine)
    for name in ("imshow", "waitKey", "destroyAllWindows"):
        monkeypatch.setattr(main.cv2, name, lambda *args: pytest.fail("Headless called GUI"))
    try:
        assert main.main(["--source", str(ROOT / filename), "--no-display"]) == 0
        summary = json.loads(capsys.readouterr().out)
    finally:
        engine.close()
    assert summary["frames_decoded"] == observed["frames"] == source["decoded_frame_count"]
    assert summary["valid_observations"] == observed["valid"]
    assert summary["predictions"] == observed["windows"] == len(window_rows)
    assert summary["warming_up_observations"] == observed["warm"] == 15
    assert summary["stop_reason"] == "eof" and engine._closed and engine.detector._closed
    record_property("application_summary", json.dumps(summary))
    record_property("max_numeric_difference", 0.0)


def test_writer_size_change_releases_resources(app, monkeypatch, tmp_path):
    writer = FakeWriter()
    monkeypatch.setattr(main.cv2, "VideoWriter", lambda *args: writer)
    original = FakeSource.read
    def read(self):
        packet = original(self)
        if packet.frame_index == 1:
            return FramePacket(np.zeros((20, 20, 3), np.uint8), 1, .04, "file")
        return packet
    monkeypatch.setattr(FakeSource, "read", read)
    with pytest.raises(RuntimeError, match="dimensions changed"):
        main.run("input.avi", no_display=True, output=tmp_path / "out.avi")
    assert writer.releases == app.closed == FakeSource.instances[0].closed == 1
