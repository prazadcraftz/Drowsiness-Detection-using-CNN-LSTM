"""Compact OpenCV dashboard with independent raw, smooth and policy labels."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from driver_monitor.inference.state_machine import DriverState

COLORS = {
    "WARMING_UP": (168, 135, 83), "NO_FACE": (136, 126, 120),
    "ATTENTIVE": (121, 174, 55), "FATIGUE": (45, 186, 235),
    "SUSPECTED_DROWSY": (38, 137, 245), "DROWSY": (48, 83, 244),
    "ALERT": (67, 53, 242),
}


@lru_cache(maxsize=4)
def _dashboard_background(shape, dtype):
    """Bounded invariant pixels only; never retain a source or rendered frame."""
    image = np.empty(shape, dtype=dtype)
    image[:] = (24, 20, 17)
    height, width = shape[:2]
    sx, sy = width / 640, height / 480
    cv2.line(image, (round(14*sx), round(357*sy)), (round(626*sx), round(357*sy)), (70, 61, 55), 1)
    image.flags.writeable = False
    return image


@dataclass(frozen=True)
class TextItem:
    text: str
    x: int
    y: int
    scale: float
    available_width: int
    color: tuple = (228, 234, 241)


def dashboard_text(frame, packet, result, source_name, *, latest_prediction=None,
                   source_fps=None, target_fps=None, policy=None, metrics=None, audio=None,
                   model_profile=None):
    """Layout is inspectable without a GUI or fragile pixel snapshots."""
    height, width = frame.shape[:2]
    sx, sy = width / 640, height / 480
    unit = min(sx, sy)
    observation = result.observation
    state = policy.state.value if policy is not None else (
        "NO_FACE" if not observation.face_detected else "WARMING_UP" if observation.warming_up else "RAW MODE")
    texts = []

    def add(text, x, y, area, size=.46, color=(228, 234, 241)):
        x, y, area = round(x * sx), round(y * sy), max(1, round(area * sx))
        scale = size * unit
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if text_width > area:
            scale *= area / text_width * .97
        texts.append(TextItem(text, x, y, scale, area, color))

    add("DRIVER DROWSINESS MONITOR", 14, 25, 612, .62)
    add(f"State: {state}", 14, 62, 610, .70, COLORS.get(state, (180, 180, 180)))
    status = "TEMPORAL RESET" if observation.gap_reset else (
        "NO FACE" if not observation.face_detected else "WARMING UP" if observation.warming_up else
        "NOT SAMPLED" if not observation.sampled else "LIVE OBSERVATION")
    add(status, 14, 84, 340, .38)
    if observation.gap_reset and not observation.face_detected:
        add("NO FACE", 400, 84, 222, .38)
    add(f"Source: {Path(source_name).name}", 14, 328, 340, .38)
    if model_profile == 'lstm32':
        add('Model: LSTM32', 400, 328, 222, .36)
    add(f"Frame: {packet.frame_index}", 14, 347, 340, .36)
    right, area = 374, 250
    raw_name = result.raw_class_name if result.raw_class_id is not None else None
    raw_confidence = result.confidence
    if raw_name is not None:
        add(f"Raw: {raw_name} {raw_confidence:.0%}", right, 111, area, .47)
        # Preserve readable legacy diagnostics through the rendering migration.
        add(f"Prediction: {raw_name}", right, 134, area, .38)
        add(f"Confidence: {raw_confidence:.1%}", right, 154, area, .38)
    elif observation.face_detected and not observation.sampled and latest_prediction is not None:
        name, confidence = latest_prediction
        add(f"Raw: held {name} {confidence:.0%}", right, 111, area)
        add(f"Latest prediction: {name}", right, 134, area, .38)
    else:
        add("Raw: unavailable", right, 111, area)
    if policy is not None and policy.smoothed_class_name is not None:
        add(f"Smooth: {policy.smoothed_class_name} {policy.smoothed_confidence:.0%}", right, 178, area, .47)
    else:
        add("Smooth: unavailable", right, 178, area)
    add(f"Sequence: {observation.sequence_fill} / {observation.sequence_capacity}", right, 209, area)
    if target_fps is not None:
        add(f"Target observations/sec: {target_fps:g}", right, 232, area, .39)
    if source_fps is not None:
        add(f"Source nominal FPS: {source_fps:g}", right, 255, area, .39)
    interval = observation.effective_observation_interval
    add("Current observation FPS: --" if interval is None else f"Current observation FPS: {1 / interval:.1f}", right, 278, area, .39)
    if policy is not None:
        add(f"Eye-close timer: {policy.eye_close_persistence_seconds:.2f} s", right, 307, area, .39)
    features = observation.raw_features
    if observation.face_detected and features is not None:
        add(f"EAR-R: {features[0]:.3f}   EAR-L: {features[1]:.3f}   MAR: {features[2]:.3f}", 14, 374, 610, .43)
        add(f"Pose phi: {features[3]:.3f}   Pose theta: {features[4]:.3f}  (legacy features)", 14, 398, 610, .40)
    else:
        add("Features: unavailable", 14, 374, 610, .43)
    if metrics is not None:
        events = metrics.events
        add(f"Yawns: {events['yawn'].count}    Eye closures: {events['eye_closure'].count}    Drowsy events: {events['drowsiness'].count}    Alerts: {events['alert'].count}", 14, 425, 610, .44)
        duration = metrics.snapshot()["session_duration_seconds"]
        minutes, seconds = divmod(int(duration), 60)
        audio_text = "--" if audio is None else str(audio["audible_trigger_count"])
        fraction = metrics.snapshot()["observed_eye_evidence_fraction"]
        fraction_text = "--" if fraction is None else f"{fraction:.0%}"
        add(f"Session: {minutes:02d}:{seconds:02d}    Audible requests: {audio_text}    Eye-Evidence Fraction: {fraction_text}", 14, 453, 610, .41)
    else:
        add("Engineering policy / local monitor", 14, 453, 610, .40)
    return state, texts


def render_result(frame, packet, result, source_name, *, latest_prediction=None,
                  source_fps=None, target_fps=None, policy=None, metrics=None, audio=None,
                  model_profile=None):
    """Return a same-size BGR dashboard copy, with source video in a left pane."""
    image = _dashboard_background(frame.shape, frame.dtype.str).copy()
    height, width = frame.shape[:2]
    state, texts = dashboard_text(frame, packet, result, source_name,
                                 latest_prediction=latest_prediction, source_fps=source_fps,
                                 target_fps=target_fps, policy=policy, metrics=metrics, audio=audio,
                                 model_profile=model_profile)
    sx, sy = width / 640, height / 480
    x1, y1, x2, y2 = (round(14*sx), round(98*sy), round(354*sx), round(311*sy))
    if x2 > x1 and y2 > y1:
        # Preserve aspect ratio inside the video pane, never stretch face features.
        factor = min((x2-x1)/width, (y2-y1)/height)
        video = cv2.resize(frame, (max(1, round(width*factor)), max(1, round(height*factor))))
        vy, vx = y1+(y2-y1-video.shape[0])//2, x1+(x2-x1-video.shape[1])//2
        image[vy:vy+video.shape[0], vx:vx+video.shape[1]] = video
    color = COLORS.get(state, (180, 180, 180))
    banner = tuple(int(channel*.20) for channel in color)
    cv2.rectangle(image, (round(8*sx), round(36*sy)), (round(632*sx), round(70*sy)), banner, -1)
    cv2.rectangle(image, (round(8*sx), round(36*sy)), (round(11*sx), round(70*sy)), color, -1)
    cv2.line(image, (round(14*sx), round(91*sy)), (round(626*sx), round(91*sy)), color, max(1, round(2*min(sx,sy))))
    for item in texts:
        cv2.putText(image, item.text, (item.x, item.y), cv2.FONT_HERSHEY_SIMPLEX,
                    item.scale, item.color, 1, cv2.LINE_AA)
    return image
