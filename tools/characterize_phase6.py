"""Offline Phase 6 evidence, not session analytics or an accuracy evaluation."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import numpy as np
import main
from driver_monitor.config import CLASS_MAPPING, RuntimeConfig
from driver_monitor.inference.decision import DecisionLayer
from driver_monitor.inference.state_machine import DriverState


def characterize(output):
    output = Path(output).resolve()
    frozen_dir = ROOT / "tests" / "fixtures"
    if output == frozen_dir or frozen_dir in output.parents:
        raise ValueError("Characterization must never write into frozen fixtures")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Characterization output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((frozen_dir / "manifest.json").read_text())
    assert hashlib.sha256((frozen_dir / "legacy_baseline.npz").read_bytes()).hexdigest() == manifest["fixture_sha256"]
    with np.load(frozen_dir / "legacy_baseline.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    runs = []
    for source_id, source in enumerate(manifest["sources"]):
        filename = source["filename"]
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == source["sha256"]
        stats = {"source": filename, "raw_counts": {name: 0 for name in CLASS_MAPPING.values()},
                 "smoothed_counts": {name: 0 for name in CLASS_MAPPING.values()},
                 "state_frame_counts": {state.value: 0 for state in DriverState},
                 "state_seconds": {state.value: 0.0 for state in DriverState},
                 "raw_switches": 0, "smoothed_switches": 0, "state_transitions": 0,
                 "longest_drowsy_duration_seconds": 0.0, "alert_ever_reached": False,
                 "raw_windows_matched": 0, "max_raw_numeric_difference": 0.0}
        last_raw = last_smooth = previous_state = previous_timestamp = None
        received = 0
        class Observer(DecisionLayer):
            def update(self, raw_result, **kwargs):
                nonlocal last_raw, last_smooth, previous_state, previous_timestamp, received
                row = source["first_frame_row"] + received
                received += 1
                obs = raw_result.observation
                assert obs.timestamp_seconds == arrays["timestamps_seconds"][row]
                assert obs.face_detected == bool(arrays["face_detected"][row])
                assert obs.sampled == obs.face_detected and not obs.gap_reset
                assert obs.sequence_fill == arrays["sequence_fill"][row]
                window = int(arrays["frame_window_indices"][row])
                assert (raw_result.raw_class_id is not None) == (window >= 0)
                if window >= 0:
                    for key, actual in [("logits", raw_result.logits), ("probabilities", raw_result.probabilities),
                                        ("sequences", raw_result.sequence), ("input_tensors", raw_result.input_tensor)]:
                        np.testing.assert_array_equal(actual, arrays[key][window])
                    assert raw_result.raw_class_id == arrays["predicted_class_ids"][window]
                    stats["raw_windows_matched"] += 1
                result = super().update(raw_result, **kwargs)
                assert result.raw_result is raw_result
                driver = result.driver
                assert driver is not None
                if previous_timestamp is not None:
                    stats["state_seconds"][previous_state] += driver.timestamp_seconds - previous_timestamp
                stats["state_frame_counts"][driver.state.value] += 1
                stats["state_transitions"] += int(driver.transition_occurred)
                stats["alert_ever_reached"] |= driver.alert_active
                if driver.state in (DriverState.DROWSY, DriverState.ALERT):
                    stats["longest_drowsy_duration_seconds"] = max(stats["longest_drowsy_duration_seconds"], driver.drowsy_duration_seconds)
                if raw_result.raw_class_id is not None:
                    stats["raw_counts"][raw_result.raw_class_name] += 1
                    stats["smoothed_counts"][driver.smoothed_class_name] += 1
                    stats["raw_switches"] += int(last_raw is not None and last_raw != raw_result.raw_class_id)
                    stats["smoothed_switches"] += int(last_smooth is not None and last_smooth != driver.smoothed_class_id)
                    last_raw, last_smooth = raw_result.raw_class_id, driver.smoothed_class_id
                previous_timestamp, previous_state = driver.timestamp_seconds, driver.state.value
                return result
        with patch.object(main, "_create_decision_layer", lambda config=None, *, enabled=True: Observer(config, enabled=enabled)):
            summary = main.run(str(ROOT / filename), no_display=True)
        assert received == summary.frames_decoded == source["decoded_frame_count"]
        stats["application"] = asdict(summary)
        assert sum(stats["raw_counts"].values()) == summary.predictions
        assert sum(stats["state_frame_counts"].values()) == summary.frames_decoded
        (output / f"source_{source_id}.json").write_text(json.dumps(stats, indent=2) + "\n")
        runs.append(stats)
        print(filename, "raw", stats["raw_counts"], "states", {k:v for k,v in stats["state_frame_counts"].items() if v}, flush=True)
    assert sum(run["raw_windows_matched"] for run in runs) == 699
    config = RuntimeConfig.from_ini()
    fields = ("smoothing_mode", "prediction_history_size", "normal_enter_threshold", "yawning_enter_threshold",
              "yawning_exit_threshold", "eye_close_enter_threshold", "eye_close_exit_threshold",
              "yawning_persistence_seconds", "drowsy_persistence_seconds", "alert_persistence_seconds")
    report = {"status": "PASS", "purpose": "characterization only; no accuracy/medical/false-positive claim",
              "defaults": {field: getattr(config, field) for field in fields}, "runs": runs,
              "raw_windows_matched": 699, "max_raw_numeric_difference": 0.0,
              "measurement_policy": "Counts are per raw model prediction or decoded-frame display state; state occupancy uses previous-state-held intervals, final frame has no invented duration; drowsy duration includes logical ALERT progression and excludes explicitly observed no-face intervals; switches count consecutive predictions within a source. These are external report diagnostics, not production event aggregation.",
              "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                                ["main.py", "driver_monitor/config.py", "driver_monitor/inference/smoothing.py",
                                 "driver_monitor/inference/state_machine.py", "driver_monitor/inference/decision.py",
                                 "tools/characterize_phase6.py"]}}
    (output / "characterization.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    characterize(args.output)
