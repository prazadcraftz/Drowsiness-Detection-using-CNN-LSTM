"""Split raw sources first, extract train first, fit train-only legacy scaling."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from . import CLASS_NAMES, FEATURE_NAMES
from .manifest import discover, read_manifest, sha256, write_json, write_manifest
from .reproducibility import RECIPE, environment, seed_everything
from .splits import SPLITS, create_splits, validate_splits

ROOT = Path(__file__).resolve().parents[1]


def fit_train_normalization(features_by_source, rows, assignments):
    ranges = np.zeros((5, 2), dtype=np.float64)  # Deliberately preserve historical zero anchors.
    source_ids = sorted(row["source_id"] for row in rows if assignments[row["source_id"]] == "train")
    if not source_ids:
        raise ValueError("No training sources")
    for source_id in source_ids:
        features = features_by_source[source_id]
        if features.ndim != 2 or features.shape[1] != 5 or len(features) == 0 or not np.isfinite(features).all():
            raise ValueError("Invalid train features")
        ranges[:, 0] = np.minimum(ranges[:, 0], features.min(axis=0))
        ranges[:, 1] = np.maximum(ranges[:, 1], features.max(axis=0))
    if not np.isfinite(ranges).all() or not (ranges[:, 1] > ranges[:, 0]).all():
        raise ValueError("Nonfinite or zero-denominator train normalization")
    return ranges, source_ids


def create_windows(features, frame_indices, ranges, *, stride=8):
    features, frame_indices = np.asarray(features), np.asarray(frame_indices)
    if features.ndim != 2 or features.shape[1] != 5 or len(features) == 0 or frame_indices.shape != (len(features),):
        raise ValueError("Invalid per-source observations")
    if not np.isfinite(features).all() or not np.all(np.diff(frame_indices) > 0):
        raise ValueError("Nonfinite features or invalid frame order")
    if stride != 8 or ranges.shape != (5, 2) or not (ranges[:, 1] > ranges[:, 0]).all():
        raise ValueError("Primary contract requires stride 8 and positive 5-feature ranges")
    normalized = (features - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0])
    if not np.isfinite(normalized).all():
        raise ValueError("Invalid normalized values")
    windows, frames, padded = [], [], []
    for start in range(0, len(features), 8):
        indices = np.minimum(np.arange(start, start + 16), len(features) - 1)
        windows.append(normalized[indices].T)
        frames.append(frame_indices[indices])
        padded.append(start + 16 > len(features))
    return np.asarray(windows), np.asarray(frames), np.asarray(padded)


def verify_integrity(rows, splits, datasets, normalization_source_ids):
    validate_splits(rows, splits)
    if sorted(normalization_source_ids) != splits["source_ids"]["train"]:
        raise ValueError("Normalization includes nontraining sources or omits training sources")
    row_by_id = {row["source_id"]: row for row in rows}
    identities = {}
    counts = {}
    for split in SPLITS:
        dataset = datasets[split]
        n = len(dataset["labels"])
        if dataset["windows"].shape != (n, 5, 16) or dataset["frame_indices"].shape != (n, 16) or dataset["source_ids"].shape != (n,) or dataset["padded"].shape != (n,):
            raise ValueError("Window metadata shapes differ")
        if not np.isfinite(dataset["windows"]).all():
            raise ValueError("Nonfinite windows")
        if set(dataset["source_ids"].tolist()) != set(splits["source_ids"][split]):
            raise ValueError("Window source coverage differs from split")
        for source_id, label, frame_ids in zip(dataset["source_ids"], dataset["labels"], dataset["frame_indices"]):
            if splits["assignments"][source_id] != split or row_by_id[source_id]["class_id"] != int(label):
                raise ValueError("Window source/label crosses splits")
            if (frame_ids < 0).any() or (np.diff(frame_ids) < 0).any():
                raise ValueError("Invalid within-source frame identity")
            for frame_id in frame_ids:
                identity = (str(source_id), int(frame_id))
                old = identities.setdefault(identity, split)
                if old != split:
                    raise ValueError("Observation identity appears in multiple splits")
        counts[split] = {"windows": n, "sources": len(set(dataset["source_ids"].tolist())), "unique_source_observation_identities": sum(value == split for value in identities.values())}
    return {"passed": True, "source_group_overlap": 0, "cross_split_observation_overlap": 0, "normalization_train_only": True, "counts": counts}


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    # Never allow experiment outputs to target the repository/corpus itself.
    if output == root or output in root.parents or any(output == root / name for name in CLASS_NAMES):
        raise ValueError("Choose a distinct experimental artifact directory")
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw_features").mkdir()
    rows = discover(root)
    write_manifest(output / "dataset_manifest_raw.csv", rows)
    splits = create_splits(rows)
    code_files = [*sorted((ROOT / "evaluation").glob("*.py")), ROOT / "dataset.py", ROOT / "model.py", ROOT / "draw_landmarks.py",
                  ROOT / "driver_monitor/vision/features.py", ROOT / "driver_monitor/vision/face_detector.py", ROOT / "driver_monitor/vision/video_source.py"]
    code_hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in code_files}
    splits.update(manifest_sha256=sha256(output / "dataset_manifest_raw.csv"), manifest_file="dataset_manifest_raw.csv", code_sha256=code_hashes,
                  config_sha256=__import__("hashlib").sha256(json.dumps(RECIPE, sort_keys=True).encode()).hexdigest())
    write_json(output / "splits.json", splits)  # Before any feature extraction.
    seed_everything()
    write_json(output / "protocol.json", {"recipe": RECIPE, "environment": environment(), "source_code_sha256": code_hashes,
                                          "detector_asset_sha256": sha256(root / "face_landmarker.task"), "split_sha256": sha256(output / "splits.json"),
                                          "declared_before_training": True, "subject_metadata": "No trustworthy subject IDs found; video-independent only"})
    from driver_monitor.vision.face_detector import FaceDetector
    from driver_monitor.vision.features import extract_features
    from driver_monitor.vision.video_source import VideoSource
    raw_features = {}
    raw_frames = {}
    started = time.perf_counter()
    with FaceDetector(root / "face_landmarker.task") as detector, (output / "preprocessing_log.jsonl").open("x", encoding="utf-8") as log:
        for split in SPLITS:
            for row in rows:
                source_id = row["source_id"]
                if splits["assignments"][source_id] != split:
                    continue
                row["split"] = split
                features, frame_ids, decoded = [], [], 0
                try:
                    if sha256(root / row["relative_path"]) != row["file_sha256"]:
                        raise ValueError("Source hash changed after split")
                    with VideoSource(root / row["relative_path"]) as video:
                        while True:
                            packet = video.read()
                            if packet.eof:
                                break
                            decoded += 1
                            landmarks = detector.detect(packet.frame)
                            if landmarks is not None:
                                feature = extract_features(packet.frame, landmarks)
                                if feature.shape != (5,) or not np.isfinite(feature).all():
                                    raise ValueError("Nonfinite or invalid legacy feature vector")
                                features.append(feature)
                                frame_ids.append(packet.frame_index)
                    if not features:
                        raise ValueError("No valid face observations")
                    array, indices = np.asarray(features), np.asarray(frame_ids, dtype=np.int64)
                    raw_features[source_id], raw_frames[source_id] = array, indices
                    np.savez_compressed(output / "raw_features" / (source_id + ".npz"), features=array, frame_indices=indices)
                    row.update(successfully_decoded_frame_count=decoded, valid_face_observation_count=len(array), preprocessing_status="ok")
                except Exception as error:
                    row.update(successfully_decoded_frame_count=decoded, valid_face_observation_count=len(features), preprocessing_status="failed", notes=f"{type(error).__name__}: {error}; action=fail experiment")
                    write_manifest(output / "dataset_manifest.csv", rows)
                    write_json(output / "preprocessing_failure.json", row)
                    raise RuntimeError(f"Preprocessing failed for {source_id}: {error}") from error
                log.write(json.dumps(row, sort_keys=True) + "\n")
                log.flush()
                print(f"{split} {row['relative_path']}: decoded={decoded}, valid={len(features)}", flush=True)
            if split == "train":
                ranges, norm_sources = fit_train_normalization(raw_features, rows, splits["assignments"])
                np.save(output / "train_norm.npy", ranges)
                write_json(output / "train_norm.json", {"feature_names": list(FEATURE_NAMES), "train_source_ids": norm_sources,
                    "min_max": ranges.tolist(), "finite": True, "positive_denominators": True, "denominators": (ranges[:, 1]-ranges[:, 0]).tolist(),
                    "sha256": sha256(output / "train_norm.npy"), "fit_method": "legacy zero-initialized extrema on TRAIN sources only", "split_sha256": sha256(output / "splits.json")})
    datasets = {}
    stats = {}
    for split in SPLITS:
        data = {name: [] for name in ("windows", "labels", "source_ids", "frame_indices", "padded")}
        for row in rows:
            if row["split"] != split:
                continue
            source_id = row["source_id"]
            windows, frames, padded = create_windows(raw_features[source_id], raw_frames[source_id], ranges)
            row.update(generated_window_count=len(windows), padded_window_count=int(padded.sum()))
            data["windows"].extend(windows)
            data["frame_indices"].extend(frames)
            data["padded"].extend(padded)
            data["labels"].extend([row["class_id"]]*len(windows))
            data["source_ids"].extend([source_id]*len(windows))
        datasets[split] = {name: np.asarray(values) for name, values in data.items()}
        np.savez_compressed(output / (split + ".npz"), **datasets[split])
        stats[split] = {}
        for name in CLASS_NAMES:
            select = [row for row in rows if row["split"] == split and row["class_name"] == name]
            count = sum(row["generated_window_count"] for row in select)
            pad = sum(row["padded_window_count"] for row in select)
            stats[split][name] = {"videos": len(select), "decoded_frames": sum(row["successfully_decoded_frame_count"] for row in select),
                "valid_observations": sum(row["valid_face_observation_count"] for row in select), "windows": count, "padded_windows": pad,
                "padded_share": pad/count, "window_share": count/len(data["labels"]),
                "face_detection_failures": sum(row["successfully_decoded_frame_count"]-row["valid_face_observation_count"] for row in select)}
    integrity = verify_integrity(rows, splits, datasets, norm_sources)
    write_json(output / "integrity.json", integrity)
    write_manifest(output / "dataset_manifest.csv", rows)
    write_json(output / "dataset_statistics.json", {"splits": stats, "sources_discovered": len(rows), "usable_sources": len(rows), "excluded_sources": [],
        "duplicate_content_groups": len(rows)-len({row["group_id"] for row in rows}), "preprocessing_wall_seconds": time.perf_counter()-started,
        "final_manifest_sha256": sha256(output / "dataset_manifest.csv"), "artifact_sha256": {name: sha256(output / name) for name in ("train.npz", "validation.npz", "test.npz", "train_norm.npy", "splits.json")}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--output", default=str(ROOT / "results/phase11"))
    args = parser.parse_args()
    prepare(args.root, args.output)


if __name__ == "__main__":
    main()
