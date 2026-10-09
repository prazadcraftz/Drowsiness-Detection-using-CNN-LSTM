"""Freeze outputs by observing the unchanged canonical inference.py main loop.

Only this development tool suppresses GUI calls and adds observers. It never
reimplements or corrects the canonical feature, normalization, sequence, model,
or softmax algorithms. No production inference modules are changed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FEATURE_ORDER = ["EAR-R", "EAR-L", "MAR", "phi", "theta"]
CLASS_NAMES = ["Normal", "Yawning", "Eye Close"]
EXPECTED_HASHES = {
    "saved_model.pt": "162426d6292a87162188b70d7bac6704e5bf217f8ff1fe5eaafc41085aa0c551",
    "norm.npy": "54b2a1fc66cc8eab3b68ba2b5fe081558aabc6e48c79f752d1d49026494e1900",
    "class_idx.npy": "4f06c199cfdb8031a6caf92501e19a073dcf08399c4661c3f9336c0fb81818b2",
    "face_landmarker.task": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
}
SOURCE_FILES = ["inference.py", "model.py", "dataset.py", "draw_landmarks.py",
                "config_parser.py", "config.ini"]
DEFAULT_SOURCES = [f"{name}/{name}_{index}.avi"
                   for name in CLASS_NAMES for index in (1, 60)]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts():
    """Fail before inference/imported model execution if a baseline asset changed."""
    actual = {name: sha256(ROOT / name) for name in EXPECTED_HASHES}
    for name, expected in EXPECTED_HASHES.items():
        if actual[name] != expected:
            raise ValueError(f"Artifact mismatch: {name}: {actual[name]} != {expected}")
    return actual


def environment(torch, cv2, mp, np):
    cpu = platform.processor()
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    names = ["torch", "opencv-contrib-python", "mediapipe", "numpy", "scikit-learn",
             "pytest", "matplotlib", "tqdm", "pyserial"]
    return {
        "python": platform.python_version(), "python_full": sys.version,
        "python_executable": sys.executable, "os": platform.platform(), "cpu": cpu,
        "logical_cpu_count": os.cpu_count(),
        "packages": {name: importlib.metadata.version(name) for name in names},
        "import_versions": {"torch": torch.__version__, "cv2": cv2.__version__,
                            "mediapipe": mp.__version__, "numpy": np.__version__},
        "device": "cpu", "cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
        "opencv_num_threads": cv2.getNumThreads(), "seed": 0,
    }


def distribution(values, np):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    return {"count": len(values), "mean_ms": float(values.mean()),
            "p50_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95))}


def capture(output, sources, overwrite=False, display=False):
    # Artifact validation deliberately precedes loading executable model assets.
    artifact_hashes = verify_artifacts()
    output = Path(output).resolve()
    if not overwrite and any((output / name).exists() for name in
                             ("manifest.json", "legacy_baseline.npz", "baseline_performance.json")):
        raise FileExistsError("Frozen fixtures exist; use a different output directory or --overwrite explicitly")
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    import numpy as np
    import cv2
    import mediapipe as mp
    import torch
    import sklearn  # noqa: F401 -- required environment verification
    import pytest  # noqa: F401 -- required environment verification
    import pickle
    import inference as canonical

    # Mapping's pickle is accepted only after its immutable audited hash passed.
    with (ROOT / "class_idx.npy").open("rb") as handle:
        mapping = {name: int(value) for name, value in pickle.load(handle).items()}
    assert mapping == dict(zip(CLASS_NAMES, range(3)))
    norm = np.load(ROOT / "norm.npy", allow_pickle=False)
    assert norm.shape == (5, 2) and np.isfinite(norm).all()
    assert np.all(norm[:, 1] > norm[:, 0])
    torch.manual_seed(0)  # initialization only; loaded weights and eval behavior are unchanged
    env = environment(torch, cv2, mp, np)
    code_lines = (ROOT / "inference.py").read_text(encoding="utf-8").splitlines()

    def anchor(prefix):
        matches = [i + 1 for i, line in enumerate(code_lines) if line.strip().startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Canonical observation anchor is ambiguous: {prefix}")
        return matches[0]

    normalize_start = anchor("r_ear_norm =")
    normalize_end = anchor("r_ear_li.append(")
    pipeline_end = anchor("h, w = frame.shape")
    no_face_end = anchor('cv2.putText(frame, "No face detected"')
    frames, windows, source_metadata = [], [], []
    original_capture, original_detector = cv2.VideoCapture, canonical.get_detector
    original_model, original_load = canonical.CNN_LSTM, torch.load
    original_ear, original_mar, original_pose = (canonical.calculate_ear,
                                                canonical.calculate_mar,
                                                canonical.calculate_head_pose)

    for source_id, source in enumerate(sources):
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        source_path = source_path.resolve()
        # Only existing local videos: camera, networking, and recording are not used.
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        relative = source_path.relative_to(ROOT).as_posix()
        current = {"row": None, "valid_rows": [], "detectors": [], "caps": [],
                   "handles": [], "normalizing_at": None}

        class ObservedCapture:
            def __init__(self, value):
                self.cap = original_capture(value)
                current["caps"].append(self)
                self.released = False
                self.index = -1
                self.metadata = {"filename": relative, "sha256": sha256(source_path),
                                 "whole_video_label": source_path.parent.name,
                                 "fps": self.cap.get(cv2.CAP_PROP_FPS),
                                 "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                 "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                 "header_frame_count": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                                 "first_frame_row": len(frames)}
                source_metadata.append(self.metadata)

            def isOpened(self):
                return self.cap.isOpened()

            def get(self, key):
                return self.cap.get(key)

            def read(self):
                started = time.perf_counter()
                ok, image = self.cap.read()
                elapsed = (time.perf_counter() - started) * 1000
                if not ok:
                    self.metadata["decoded_frame_count"] = self.index + 1
                    self.metadata["eof_observed"] = True
                    return ok, image
                self.index += 1
                timestamp = float(self.cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                row = {"source_id": source_id, "frame_index": self.index,
                       "canonical_frame_number": self.index + 1,
                       "timestamp_seconds": timestamp, "face_detected": False,
                       "frame_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                       "raw_features": np.full(5, np.nan),
                       "normalized_features": np.full(5, np.nan), "landmarks": None,
                       "sequence_fill": len(current["valid_rows"][-16:]),
                       "warming_up": True, "window_index": -1,
                       "timings": {"acquisition": elapsed, "mediapipe": np.nan,
                                   "features": 0.0, "normalization": np.nan,
                                   "cnn_lstm": np.nan, "processing": np.nan},
                       "started": time.perf_counter()}
                frames.append(row)
                current["row"] = row
                return ok, image

            def release(self):
                self.cap.release()
                self.released = True

        class ObservedDetector:
            def __init__(self, value):
                self.detector = original_detector(value)
                current["detectors"].append(self.detector)

            def detect(self, image):
                started = time.perf_counter()
                result = self.detector.detect(image)
                current["row"]["timings"]["mediapipe"] = (time.perf_counter() - started) * 1000
                return result

        def timed_features(function):
            def observed(*args, **kwargs):
                started = time.perf_counter()
                value = function(*args, **kwargs)
                current["row"]["timings"]["features"] += (time.perf_counter() - started) * 1000
                return value
            return observed

        def make_model(config):
            model = original_model(config).to("cpu")

            def before(module, args):
                current["model_started"] = time.perf_counter()

            def after(module, args, logits):
                # Stop the timer before copying observer data; softmax remains in inference.py.
                row = current["row"]
                row["timings"]["cnn_lstm"] = (time.perf_counter() - current["model_started"]) * 1000
                assert not module.training and args[0].device.type == "cpu"
                assert args[0].shape == (1, 5, 16) and logits.shape == (1, 3)
                current["tensor"] = args[0].detach().cpu().numpy().copy()
                current["logits"] = logits.detach().cpu().numpy().copy()

            current["handles"].extend([model.register_forward_pre_hook(before),
                                       model.register_forward_hook(after)])
            return model

        def cpu_load(*args, **kwargs):
            kwargs["map_location"] = "cpu"
            return original_load(*args, **kwargs)

        def observe(frame, event, arg):
            if frame.f_code is not canonical.main.__code__:
                return None
            if event != "line" or current["row"] is None:
                return observe
            row, local = current["row"], frame.f_locals
            if frame.f_lineno == normalize_start:
                row["face_detected"] = True
                row["landmarks"] = local["landmarks"].copy()
                row["raw_features"] = np.array([local["r_ear"], local["l_ear"], local["mar"],
                                                local["phi"][0], local["theta"][0]])
                current["normalizing_at"] = time.perf_counter()
            elif frame.f_lineno == normalize_end:
                row["timings"]["normalization"] = (time.perf_counter() - current["normalizing_at"]) * 1000
                row["normalized_features"] = np.array([local[name] for name in
                    ("r_ear_norm", "l_ear_norm", "mar_norm", "phi_norm", "theta_norm")])
                current["valid_rows"].append(len(frames) - 1)
            elif frame.f_lineno in (pipeline_end, no_face_end):
                row["timings"]["processing"] = (time.perf_counter() - row["started"]) * 1000
                row["sequence_fill"] = len(local["frame_buffer"])
                row["warming_up"] = len(local["frame_buffer"]) < 16
                if row["face_detected"] and len(local["frame_buffer"]) == 16:
                    row["window_index"] = len(windows)
                    windows.append({"frame_row": len(frames) - 1,
                                    "observation_frame_rows": current["valid_rows"][-16:].copy(),
                                    "sequence": local["fr"].copy(),
                                    "tensor": current["tensor"], "logits": current["logits"],
                                    "probabilities": local["softmax"].detach().cpu().numpy().copy(),
                                    "predicted_class_id": local["predicted_class"]})
            return observe

        previous_directory, previous_trace = Path.cwd(), sys.gettrace()
        try:
            os.chdir(ROOT)
            with contextlib.ExitStack() as stack:
                replacements = [(cv2, "VideoCapture", ObservedCapture),
                                (canonical, "get_detector", ObservedDetector),
                                (canonical, "CNN_LSTM", make_model),
                                (torch, "load", cpu_load),
                                (canonical, "calculate_ear", timed_features(original_ear)),
                                (canonical, "calculate_mar", timed_features(original_mar)),
                                (canonical, "calculate_head_pose", timed_features(original_pose))]
                for target, name, replacement in replacements:
                    stack.enter_context(patch.object(target, name, replacement))
                if not display:
                    for name, replacement in [("imshow", lambda *args: None),
                                              ("waitKey", lambda *args: -1),
                                              ("destroyAllWindows", lambda: None)]:
                        stack.enter_context(patch.object(cv2, name, replacement))
                stack.enter_context(patch.object(sys, "argv", ["inference.py", "--source", str(source_path)]))
                sys.settrace(observe)
                # Run the actual canonical script, including its acquisition, warm-up,
                # inference, softmax, overlays, EOF and normal cleanup code.
                canonical.main()
        finally:
            sys.settrace(previous_trace)
            os.chdir(previous_directory)
            for handle in current["handles"]:
                handle.remove()
            for detector in current["detectors"]:
                detector.close()  # tool cleanup: canonical script omits explicit closure
            for cap in current["caps"]:
                cap.metadata["canonical_capture_released"] = cap.released
                cap.metadata["last_frame_row_exclusive"] = len(frames)
                if not cap.released:
                    cap.release()
        if not current["caps"] or not current["caps"][0].metadata.get("eof_observed"):
            raise RuntimeError(f"Canonical path did not reach file EOF: {source}")
        print(f"Captured {relative}: {current['caps'][0].metadata['decoded_frame_count']} frames")

    assert windows, "No inference windows were produced"
    landmark_count = next(row["landmarks"].shape[0] for row in frames if row["landmarks"] is not None)
    stages = ["acquisition", "mediapipe", "features", "normalization", "cnn_lstm", "processing"]
    arrays = {
        "source_ids": np.array([r["source_id"] for r in frames], dtype=np.int64),
        "frame_indices": np.array([r["frame_index"] for r in frames], dtype=np.int64),
        "timestamps_seconds": np.array([r["timestamp_seconds"] for r in frames]),
        "frame_sha256": np.array([r["frame_sha256"] for r in frames]),
        "face_detected": np.array([r["face_detected"] for r in frames], dtype=bool),
        "raw_features": np.stack([r["raw_features"] for r in frames]),
        "normalized_features": np.stack([r["normalized_features"] for r in frames]),
        "landmarks_xy": np.stack([r["landmarks"] if r["landmarks"] is not None else
                                  np.full((landmark_count, 2), np.nan) for r in frames]),
        "sequence_fill": np.array([r["sequence_fill"] for r in frames], dtype=np.int64),
        "warming_up": np.array([r["warming_up"] for r in frames], dtype=bool),
        "frame_window_indices": np.array([r["window_index"] for r in frames], dtype=np.int64),
        "window_frame_rows": np.array([w["frame_row"] for w in windows], dtype=np.int64),
        "observation_frame_rows": np.array([w["observation_frame_rows"] for w in windows], dtype=np.int64),
        "sequences": np.stack([w["sequence"] for w in windows]),
        "input_tensors": np.stack([w["tensor"] for w in windows]),
        "logits": np.stack([w["logits"] for w in windows]),
        "probabilities": np.stack([w["probabilities"] for w in windows]),
        "predicted_class_ids": np.array([w["predicted_class_id"] for w in windows], dtype=np.int64),
        "predicted_class_names": np.array([CLASS_NAMES[w["predicted_class_id"]] for w in windows]),
        "normalization_ranges": norm,
        "stage_times_ms": np.array([[r["timings"][name] for name in stages] for r in frames]),
    }
    assert all(np.isfinite(arrays[name]).all() for name in
               ("input_tensors", "sequences", "logits", "probabilities"))
    valid_rows = np.flatnonzero(arrays["face_detected"])
    # Exclude first 20 valid observations per video (including 5 ready windows).
    steady_rows = []
    for source_id in range(len(sources)):
        steady_rows.extend(valid_rows[arrays["source_ids"][valid_rows] == source_id][20:])
    performance = {
        "environment": env, "stages": stages,
        "policy": {"excluded_first_valid_observations_per_video": 20,
                   "model_initialization_and_asset_load_excluded": True,
                   "full_processing_ends_before_overlay": True,
                   "GUI_render_and_video_encoding_excluded": True,
                   "instrumentation": "original main line trace, feature timers, model hooks; observer overhead remains",
                   "optimization_performed": False},
        "steady_frame_rows": [int(row) for row in steady_rows],
        "steady_state": {stage: distribution(arrays["stage_times_ms"][steady_rows, index], np)
                         for index, stage in enumerate(stages)},
        "all_frames": {stage: distribution(arrays["stage_times_ms"][:, index], np)
                       for index, stage in enumerate(stages)},
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "legacy_baseline.npz", **arrays)
    gaps = []
    for source_id in range(len(sources)):
        rows = np.flatnonzero(arrays["source_ids"] == source_id)
        missing = ~arrays["face_detected"][rows]
        padded = np.r_[False, missing, False].astype(np.int8)
        for start, end in zip(np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)):
            first, last = int(rows[start]), int(rows[end - 1])
            gaps.append({"source_id": source_id, "first_frame_row": first,
                         "last_frame_row": last, "frames": int(end - start),
                         "first_timestamp_seconds": float(arrays["timestamps_seconds"][first]),
                         "last_timestamp_seconds": float(arrays["timestamps_seconds"][last])})
    manifest = {
        "schema_version": 1, "canonical_reference": "inference.py:main",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": env, "artifact_sha256": artifact_hashes,
        "source_code_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "capture_tool_sha256": sha256(Path(__file__)),
        "fixture_sha256": sha256(output / "legacy_baseline.npz"),
        "checkpoint": "saved_model.pt", "normalization": "norm.npy",
        "feature_order": FEATURE_ORDER, "class_mapping": mapping,
        "sequence_length": 16, "stride_valid_observations": 1,
        "frame_index_origin": 0, "timestamps": "OpenCV CAP_PROP_POS_MSEC after read, in seconds; no fallback or sampling",
        "sources": source_metadata, "frame_count": len(frames),
        "valid_observation_count": len(valid_rows), "window_count": len(windows),
        "raw_prediction_counts": {name: int(np.sum(arrays["predicted_class_ids"] == i))
                                  for i, name in enumerate(CLASS_NAMES)},
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                   for name, value in arrays.items()},
        "expected_shapes": {"feature_vector": [5], "window": [5, 16],
                            "tensor": [1, 5, 16], "output": [1, 3]},
        "tolerance": {"atol": 1e-6, "rtol": 1e-5, "raw_class_agreement": "exact"},
        "face_loss_runs": gaps,
        "annotations": "Directory labels are whole-video labels, not frame/onset annotations. No transitions or missing-face events were fabricated.",
        "behavior": {"BGR_tagged_SRGB": True, "legacy_head_pose": True,
                     "legacy_unbounded_feature_histories": True, "no_sampling": True,
                     "no_face_gap_reset": True, "no_smoothing": True,
                     "eval": True, "no_grad": True, "softmax_dim": 1},
        "runtime_adaptations": ["explicit CPU model/load mapping; existing storages already CPU",
                                "GUI calls suppressed unless --display; original overlays still run",
                                "tool closes detector after original main returns; source EOF/release recorded"],
    }
    for name, document in [("manifest.json", manifest), ("baseline_performance.json", performance)]:
        (output / name).write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    verify_artifacts()  # ensure running the original path did not modify any assets
    print(json.dumps({"frames": len(frames), "windows": len(windows),
                      "prediction_counts": manifest["raw_prediction_counts"],
                      "face_loss_runs": len(gaps), "output": str(output)}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--output", type=Path, default=ROOT / "tests" / "fixtures")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace frozen fixtures")
    parser.add_argument("--display", action="store_true", help="Show original OpenCV windows")
    args = parser.parse_args()
    capture(args.output, args.sources, args.overwrite, args.display)


if __name__ == "__main__":
    main()
