"""Recommended local application: python main.py --source 0 (or a file)."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
import inspect
import json
from pathlib import Path
import sys
import time

import cv2

from driver_monitor.vision.video_source import SourceError, VideoSource
from driver_monitor.metrics import SessionMetrics
from driver_monitor.alerts import AlertManager, ALERT_COOLDOWN_SECONDS
from driver_monitor.ui.overlay import render_result

WRITER_FPS_FALLBACK = 20.0  # Output metadata only; never a sampling rate.


def _create_engine(config=None):
    # Invalid/unreadable sources fail before model/detector imports and loading.
    from driver_monitor.inference.engine import DrowsinessInferenceEngine
    return DrowsinessInferenceEngine(config)


def _create_decision_layer(config=None, *, enabled=True):
    from driver_monitor.inference.decision import DecisionLayer
    return DecisionLayer(config, enabled=enabled)


@dataclass(frozen=True)
class RunSummary:
    source_name: str
    frames_decoded: int
    valid_observations: int
    predictions: int
    warming_up_observations: int
    no_face_frames: int
    engine_processing_seconds: float
    loop_wall_seconds: float
    total_wall_seconds: float
    processing_throughput_fps: float
    stop_reason: str
    output_path: str | None
    output_fps: float | None
    sampled_valid_observations: int
    not_sampled_valid_frames: int
    gap_resets: int
    missed_sampling_deadlines: int
    source_timestamp_span_seconds: float | None
    capture_decode_source_fps: float | None
    effective_model_observation_fps: float | None
    target_model_fps: float | None
    temporal_runtime_enabled: bool
    session_metrics: dict


def _open_writer(path: Path, fps: float, frame):
    codecs = {".avi": "MJPG", ".mp4": "mp4v", ".mov": "mp4v"}
    if path.suffix.lower() not in codecs:
        raise ValueError("Output extension must be .avi, .mp4 or .mov")
    if path.exists():
        raise ValueError(f"Output already exists; choose a new filename: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codecs[path.suffix.lower()]),
                             fps, (width, height))
    try:
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open output video writer: {path}")
    except BaseException:
        writer.release()
        raise
    return writer


def run(source, *, no_display=False, save_output=False, output=None,
        window_name="Driver Monitor", target_model_fps=None,
        max_face_gap_seconds=None, compatibility_mode=False,
        smoothing_mode=None, prediction_history_size=None, no_audio=False,
        alert_cooldown_seconds=ALERT_COOLDOWN_SECONDS, sqlite=False,
        db_path=None, model_profile='legacy') -> RunSummary:
    started = time.perf_counter()
    if db_path is not None and not sqlite:
        raise ValueError("--db-path requires --sqlite")
    output_path = Path(output or "model_output.avi").expanduser().resolve() if save_output or output is not None else None
    if not window_name.strip():
        raise ValueError("Window name cannot be empty")
    frames = valid = predictions = warming = no_face = 0
    engine_seconds = 0.0
    output_fps = None
    stop_reason = "eof"
    sampled = not_sampled = resets = missed = 0
    first_timestamp = last_timestamp = first_observation_time = last_observation_time = None
    latest_prediction = None
    config = None
    if model_profile != 'legacy' or any(value is not None for value in (target_model_fps, max_face_gap_seconds,
                                           smoothing_mode, prediction_history_size)):
        from driver_monitor.config import RuntimeConfig
        from driver_monitor.inference.model_profile import get_profile
        get_profile(model_profile)
        overrides = {}
        if target_model_fps is not None:
            overrides["target_model_fps"] = target_model_fps
        if max_face_gap_seconds is not None:
            overrides["max_face_gap_seconds"] = max_face_gap_seconds
        if smoothing_mode is not None:
            overrides["smoothing_mode"] = smoothing_mode
        if prediction_history_size is not None:
            overrides["prediction_history_size"] = prediction_history_size
        config = replace(RuntimeConfig.from_ini(model_profile=model_profile), **overrides)
    metrics = SessionMetrics()
    with ExitStack() as resources:
        alerts = resources.enter_context(AlertManager(enabled=not (no_audio or no_display), cooldown_seconds=alert_cooldown_seconds))
        video = resources.enter_context(VideoSource(source))
        # Probe decoding before expensive engine initialization, even for corrupt files.
        packet = video.read()
        if output_path is not None:
            if output_path == video.file_path:
                raise ValueError("Output must differ from the input video")
            output_fps = video.nominal_fps or WRITER_FPS_FALLBACK
            writer = _open_writer(output_path, output_fps, packet.frame)
            resources.callback(writer.release)
            writer_shape = packet.frame.shape[:2]
        else:
            writer = None
        engine = resources.enter_context(_create_engine() if config is None else _create_engine(config))
        # Keep single-argument legacy adapters usable without changing old tests.
        # The production engine always accepts timestamps; this is resolved once.
        parameters = inspect.signature(engine.process_frame).parameters
        accepts_timestamp = "timestamp_seconds" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
        timed = not compatibility_mode and accepts_timestamp
        target_fps = getattr(getattr(engine, "config", None), "target_model_fps", None)
        decision_layer = _create_decision_layer(getattr(engine, "config", None), enabled=timed)
        persistence = None
        if sqlite:
            from driver_monitor.storage import SQLiteStore, SessionPersistence, session_provenance
            store = resources.enter_context(SQLiteStore(db_path or "data/drowsiness.db"))
            provenance = session_provenance(engine, timed=timed, audio_enabled=alerts.enabled,
                                            alert_cooldown_seconds=alert_cooldown_seconds)
            session_id = store.start_session(source_type="file" if video.file_path is not None else "webcam",
                                             source_name=video.file_path.name if video.file_path is not None else video.source_name,
                                             provenance=provenance)
            persistence = SessionPersistence(store, session_id)
        if not no_display:
            # Register before imshow so partially created windows also get cleanup.
            resources.callback(cv2.destroyAllWindows)
        loop_started = time.perf_counter()
        while not packet.eof:
            tick = time.perf_counter()
            timestamp_args = {"timestamp_seconds": packet.timestamp_seconds} if timed else {}
            result = engine.process_frame(packet.frame, **timestamp_args)
            engine_seconds += time.perf_counter() - tick
            policy_result = decision_layer.update(result, timestamp_seconds=packet.timestamp_seconds)
            metrics.update(policy_result, timestamp_seconds=packet.timestamp_seconds)
            if persistence is not None:
                persistence.update(metrics, policy_result)
            alerts.update(policy_result)
            frames += 1
            timestamp = packet.timestamp_seconds
            if timestamp is not None:
                if first_timestamp is None:
                    first_timestamp = timestamp
                last_timestamp = timestamp
            observation = result.observation
            resets += int(observation.gap_reset)
            missed += observation.missed_deadlines
            if observation.gap_reset or not observation.face_detected:
                latest_prediction = None
            if result.observation.face_detected:
                valid += 1
                if observation.sampled:
                    sampled += 1
                    if timestamp is not None:
                        if first_observation_time is None:
                            first_observation_time = timestamp
                        last_observation_time = timestamp
                    if result.raw_class_id is None:
                        warming += 1
                else:
                    not_sampled += 1
            else:
                no_face += 1
            predictions += int(result.raw_class_id is not None)
            if result.raw_class_id is not None:
                latest_prediction = (result.raw_class_name, result.confidence)
            if writer is not None or not no_display:
                display = render_result(packet.frame, packet, result, video.source_name,
                                        latest_prediction=latest_prediction, source_fps=video.nominal_fps,
                                        target_fps=target_fps if timed else None, policy=policy_result.driver,
                                        metrics=metrics, audio=alerts.snapshot(packet.timestamp_seconds),
                                        model_profile=model_profile)
                if writer is not None:
                    if display.shape[:2] != writer_shape:
                        raise RuntimeError("Decoded frame dimensions changed; cannot write fixed-size output")
                    writer.write(display)
                if not no_display:
                    cv2.imshow(window_name, display)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        stop_reason = "keyboard"
                        break
            packet = video.read()
        loop_seconds = time.perf_counter() - loop_started
        if persistence is not None:
            # Finish audio before taking the final aggregate; all SQL stays on
            # this thread. Exceptional exits skip this block and remain unfinished.
            alerts.close()
            metrics.finalize()
            persistence.finalize(metrics, audio=alerts.snapshot(last_timestamp),
                                 offline_processing_fps=(frames / loop_seconds if loop_seconds > 0 and video.file_path is not None else None))
    metrics.finalize()
    aggregate = metrics.snapshot(alerts.snapshot(last_timestamp))
    total_seconds = time.perf_counter() - started
    span = last_timestamp - first_timestamp if last_timestamp is not None and first_timestamp is not None else None
    observation_span = (last_observation_time - first_observation_time
                        if last_observation_time is not None and first_observation_time is not None else None)
    return RunSummary(video.source_name, frames, valid, predictions, warming, no_face,
                      engine_seconds, loop_seconds, total_seconds,
                      frames / loop_seconds if loop_seconds > 0 else 0.0, stop_reason,
                      str(output_path) if output_path is not None else None, output_fps,
                      sampled, not_sampled, resets, missed, span,
                      (frames - 1) / span if span is not None and span > 0 else None,
                      (sampled - 1) / observation_span if observation_span is not None and observation_span > 0 else None,
                      target_fps, timed, aggregate)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Local driver monitor with raw predictions, policy, events and dashboard")
    parser.add_argument("--source", required=True, help="Camera index (0, 1, 2, ...) or local video path")
    parser.add_argument("--model-profile", choices=('legacy','lstm32'), default='legacy',
                        help="Local model contract (default: legacy; lstm32 is development-selected, not externally validated)")
    parser.add_argument("--no-display", action="store_true", help="Do not call OpenCV GUI functions")
    parser.add_argument("--save-output", action="store_true", help="Save annotated video (default: model_output.avi)")
    parser.add_argument("--output", help="Output .avi/.mp4/.mov path; also enables saving; must not already exist")
    parser.add_argument("--window-name", default="Driver Monitor")
    parser.add_argument("--target-model-fps", type=float, help="Target observation cadence (default: runtime config, 20)")
    parser.add_argument("--max-face-gap-seconds", type=float, help="Engineering buffer-reset threshold (default: runtime config, 0.5)")
    parser.add_argument("--compatibility-mode", action="store_true", help="Untimed Phase 4 behavior: no sampling or gap reset")
    parser.add_argument("--smoothing-mode", choices=("probability_average", "majority_vote"), help="Policy smoothing mode (default: probability_average)")
    parser.add_argument("--prediction-history-size", type=int, help="Bounded policy history (default: 5 predictions)")
    parser.add_argument("--no-audio", action="store_true", help="Disable local warning audio (recommended for automated runs)")
    parser.add_argument("--alert-cooldown-seconds", type=float, default=ALERT_COOLDOWN_SECONDS, help="Audio anti-spam interval (default: 5 seconds)")
    parser.add_argument("--sqlite", action="store_true", help="Persist local sessions and event transitions (disabled by default)")
    parser.add_argument("--db-path", help="SQLite database path (requires --sqlite; default: data/drowsiness.db)")
    args = parser.parse_args(argv)
    try:
        summary = run(**vars(args))
    except KeyboardInterrupt:
        print("Interrupted; resources released.", file=sys.stderr)
        return 130
    except (SourceError, ValueError, RuntimeError, OSError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
