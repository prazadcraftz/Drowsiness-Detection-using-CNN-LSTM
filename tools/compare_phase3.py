"""Phase 3 only: fresh legacy vs engine replay and observational CPU timings.

Uses all six immutable Phase 2 sources, the unchanged Phase 2 capture tool, and
an instrumented candidate with the same timing boundaries/warm-up policy.
Neither frozen fixtures nor legacy production scripts are replaced.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["acquisition", "mediapipe", "features", "normalization", "cnn_lstm", "processing"]
ATOL, RTOL = 1e-6, 1e-5
IDENTICAL = ["source_ids", "frame_indices", "frame_sha256", "face_detected", "sequence_fill",
             "warming_up", "frame_window_indices", "window_frame_rows", "observation_frame_rows",
             "predicted_class_ids", "predicted_class_names"]
NUMERIC = ["timestamps_seconds", "raw_features", "normalized_features", "landmarks_xy",
           "sequences", "input_tensors", "logits", "probabilities", "normalization_ranges"]


def compare(actual, reference, np):
    maximum = {}
    for name in IDENTICAL:
        np.testing.assert_array_equal(actual[name], reference[name], err_msg=name)
    for name in NUMERIC:
        np.testing.assert_allclose(actual[name], reference[name], atol=ATOL, rtol=RTOL,
                                   equal_nan=True, err_msg=name)
        finite = np.isfinite(reference[name])
        maximum[name] = float(np.max(np.abs(actual[name][finite] - reference[name][finite])))
    return {"max_absolute_difference": maximum,
            "exact_raw_class_matches": int(np.sum(actual["predicted_class_ids"] == reference["predicted_class_ids"])),
            "window_count": len(reference["input_tensors"]), "frame_count": len(reference["frame_indices"])}


def run(output):
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    from tools.capture_baseline import capture, sha256, verify_artifacts, distribution, environment
    import numpy as np
    import cv2
    import mediapipe as mp
    import torch
    from driver_monitor import DrowsinessInferenceEngine
    from driver_monitor.vision import features

    output = Path(output).resolve()
    frozen_dir = ROOT / "tests" / "fixtures"
    if output == frozen_dir.resolve() or frozen_dir.resolve() in output.parents:
        raise ValueError("Phase 3 outputs must never be written inside the frozen Phase 2 fixture directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Comparison output must be empty to preserve run evidence: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((frozen_dir / "manifest.json").read_text())
    assert sha256(frozen_dir / "legacy_baseline.npz") == manifest["fixture_sha256"]
    for name, digest in manifest["source_code_sha256"].items():
        assert sha256(ROOT / name) == digest, name
    verify_artifacts()
    env = environment(torch, cv2, mp, np)
    # Require the recorded software/thread/device setup; do not tune it for timing.
    for name in ["packages", "device", "torch_num_threads", "torch_num_interop_threads",
                 "opencv_num_threads", "torch_deterministic_algorithms", "torch_mkldnn_enabled"]:
        assert env[name] == manifest["environment"][name], f"Unmatched environment setting: {name}"
    print("Capturing fresh unchanged canonical path on all six sources...", flush=True)
    with (output / "legacy_run.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        capture(output / "legacy", [source["filename"] for source in manifest["sources"]])
    with np.load(frozen_dir / "legacy_baseline.npz", allow_pickle=False) as archive:
        frozen = {name: archive[name] for name in archive.files}
    with np.load(output / "legacy" / "legacy_baseline.npz", allow_pickle=False) as archive:
        legacy = {name: archive[name] for name in archive.files}
    legacy_check = compare(legacy, frozen, np)
    print("Legacy replay matches frozen outputs; processing candidate frames...", flush=True)
    rows, windows = [], []

    for source_id, source in enumerate(manifest["sources"]):
        path = ROOT / source["filename"]
        assert sha256(path) == source["sha256"], source["filename"]
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open Phase 2 source: {path}")
        valid_rows = []
        current = {}
        try:
            with DrowsinessInferenceEngine() as engine, contextlib.ExitStack() as stack:
                def timed(function, stage):
                    def wrapped(*args, **kwargs):
                        if stage == "normalization":
                            current["timings"][stage] = 0.0
                        started = time.perf_counter()
                        value = function(*args, **kwargs)
                        current["timings"][stage] += (time.perf_counter() - started) * 1000
                        return value
                    return wrapped

                # Same detector core and four feature-function boundaries as Phase 2.
                stack.enter_context(patch.object(engine.detector._detector, "detect",
                    timed(engine.detector._detector.detect, "mediapipe")))
                for name in ("calculate_eye_ratio", "calculate_mouth_ratio", "calculate_head_pose"):
                    stack.enter_context(patch.object(features, name, timed(getattr(features, name), "features")))
                stack.enter_context(patch.object(engine.normalizer, "normalize", timed(engine.normalizer.normalize, "normalization")))
                detector_method = engine.detector.detect

                def observe_landmarks(frame):
                    landmarks = detector_method(frame)
                    current["landmarks"] = None if landmarks is None else landmarks.copy()
                    return landmarks

                stack.enter_context(patch.object(engine.detector, "detect", observe_landmarks))

                def before(module, args):
                    current["model_started"] = time.perf_counter()

                def after(module, args, logits):
                    current["timings"]["cnn_lstm"] = (time.perf_counter() - current["model_started"]) * 1000
                    # Mirror the Phase 2 model observer snapshots after stopping its timer.
                    current["observed_tensor"] = args[0].detach().cpu().numpy().copy()
                    current["observed_logits"] = logits.detach().cpu().numpy().copy()

                handles = [engine.model.register_forward_pre_hook(before), engine.model.register_forward_hook(after)]
                for handle in handles:
                    stack.callback(handle.remove)
                traced = {engine.process_frame.__func__.__code__, engine.process_features.__func__.__code__,
                          engine.infer_sequence.__func__.__code__}

                def line_observer(frame, event, arg):
                    # Same per-line observation mechanism; candidate code has different
                    # line/copy counts, so timing remains observational, not a speed claim.
                    return line_observer if frame.f_code in traced else None

                previous_trace = sys.gettrace()
                stack.callback(sys.settrace, previous_trace)
                sys.settrace(line_observer)
                index = 0
                while True:
                    acquired_at = time.perf_counter()
                    ok, frame = cap.read()
                    acquisition = (time.perf_counter() - acquired_at) * 1000
                    if not ok:
                        break
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    frame_hash = hashlib.sha256(frame.tobytes()).hexdigest()
                    current = {"timings": {stage: 0.0 for stage in STAGES}, "landmarks": None}
                    current["timings"]["acquisition"] = acquisition
                    current["timings"]["normalization"] = np.nan
                    current["timings"]["cnn_lstm"] = np.nan
                    started = time.perf_counter()
                    result = engine.process_frame(frame)
                    current["timings"]["processing"] = (time.perf_counter() - started) * 1000
                    observation = result.observation
                    row_id = len(rows)
                    row = {"source_id": source_id, "frame_index": index, "timestamp": timestamp,
                           "frame_sha256": frame_hash, "face_detected": observation.face_detected,
                           "sequence_fill": observation.sequence_fill, "warming_up": observation.warming_up,
                           "raw_features": observation.raw_features if observation.face_detected else np.full(5, np.nan),
                           "normalized_features": observation.normalized_features if observation.face_detected else np.full(5, np.nan),
                           "landmarks_xy": current["landmarks"] if observation.face_detected else np.full((478, 2), np.nan),
                           "frame_window_index": -1, "timings": current["timings"]}
                    assert not observation.reset_occurred
                    if observation.face_detected:
                        valid_rows.append(row_id)
                    if result.logits is not None:
                        row["frame_window_index"] = len(windows)
                        assert result.input_tensor_shape == (1, 5, 16)
                        np.testing.assert_array_equal(result.input_tensor, current["observed_tensor"])
                        np.testing.assert_array_equal(result.logits, current["observed_logits"])
                        windows.append({"frame_row": row_id, "observation_frame_rows": valid_rows[-16:].copy(),
                                        "result": result})
                    rows.append(row)
                    index += 1
                assert index == source["decoded_frame_count"]
                assert engine.buffer.capacity == 16 and engine.buffer.feature_storage_bytes <= 640
        finally:
            cap.release()
        print(f"Candidate completed {source['filename']}: {index} frames", flush=True)

    arrays = {name: np.array([row[field] for row in rows]) for name, field in [
        ("source_ids", "source_id"), ("frame_indices", "frame_index"),
        ("timestamps_seconds", "timestamp"), ("frame_sha256", "frame_sha256"),
        ("face_detected", "face_detected"), ("sequence_fill", "sequence_fill"),
        ("warming_up", "warming_up"), ("frame_window_indices", "frame_window_index")]}
    for name in ["raw_features", "normalized_features", "landmarks_xy"]:
        arrays[name] = np.stack([row[name] for row in rows])
    arrays.update({"window_frame_rows": np.array([window["frame_row"] for window in windows]),
                   "observation_frame_rows": np.array([window["observation_frame_rows"] for window in windows]),
                   "normalization_ranges": np.load(ROOT / "norm.npy", allow_pickle=False),
                   "stage_times_ms": np.array([[row["timings"][stage] for stage in STAGES] for row in rows])})
    for name, field in [("sequences", "sequence"), ("input_tensors", "input_tensor"),
                        ("logits", "logits"), ("probabilities", "probabilities"),
                        ("predicted_class_ids", "raw_class_id"), ("predicted_class_names", "raw_class_name")]:
        arrays[name] = np.array([getattr(window["result"], field) for window in windows])
    candidate_check = compare(arrays, frozen, np)
    direct_check = compare(arrays, legacy, np)
    np.savez_compressed(output / "engine_candidate.npz", **arrays)
    steady_rows = []
    for source_id in range(len(manifest["sources"])):
        valid = np.flatnonzero((arrays["source_ids"] == source_id) & arrays["face_detected"])
        steady_rows.extend(valid[20:])
    candidate_times = {stage: distribution(arrays["stage_times_ms"][steady_rows, index], np)
                       for index, stage in enumerate(STAGES)}
    performance = json.loads((output / "legacy" / "baseline_performance.json").read_text())
    frozen_performance = json.loads((frozen_dir / "baseline_performance.json").read_text())
    summary = {"status": "PASS", "environment": env, "tolerance": {"atol": ATOL, "rtol": RTOL},
               "sources": manifest["sources"], "legacy_vs_frozen": legacy_check,
               "engine_vs_frozen": candidate_check, "engine_vs_fresh_legacy": direct_check,
               "steady_frame_rows": [int(row) for row in steady_rows],
               "candidate_steady_state": candidate_times,
               "fresh_legacy_steady_state": performance["steady_state"],
               "frozen_phase2_steady_state": frozen_performance["steady_state"],
               "timing_policy": {"excluded_first_valid_observations_per_source": 20,
                                 "includes_observer_overhead": True,
                                 "processing_excludes_acquisition_frame_hash_overlay_GUI_and_encoding": True,
                                 "legacy_harness": "unchanged tools/capture_baseline.py",
                                 "candidate_harness": "core detector/feature timers, model hooks, engine line tracing",
                                 "line_and_snapshot_overhead_not_identical": True,
                                 "phase9_optimization_or_percentage_claim": False},
               "package_sha256": {str(path.relative_to(ROOT).as_posix()): sha256(path)
                                  for path in sorted((ROOT / "driver_monitor").rglob("*.py"))},
               "comparison_tool_sha256": sha256(Path(__file__)),
               "frozen_manifest_sha256": sha256(frozen_dir / "manifest.json"),
               "frozen_fixture_sha256": sha256(frozen_dir / "legacy_baseline.npz")}
    (output / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    verify_artifacts()
    print(json.dumps({"status": "PASS", "frames": len(rows), "windows": len(windows),
                      "class_matches": direct_check["exact_raw_class_matches"],
                      "maximum_differences": direct_check["max_absolute_difference"]}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="A new/empty comparison directory, never frozen fixtures")
    run(parser.parse_args().output)
