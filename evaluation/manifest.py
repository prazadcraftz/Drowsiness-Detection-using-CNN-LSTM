"""Deterministic raw-video inventory with explicit labels and content identity."""

import csv
import hashlib
import json
from pathlib import Path

from . import CLASS_MAPPING

FIELDS = ("source_id", "class_name", "class_id", "relative_path", "file_sha256", "nominal_fps", "width", "height", "header_frame_count",
          "successfully_decoded_frame_count", "valid_face_observation_count", "generated_window_count", "padded_window_count", "split", "group_id", "preprocessing_status", "notes")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_manifest(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["source_id"]))


def read_manifest(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["class_id"] = int(row["class_id"])
    validate_manifest(rows)
    return rows


def validate_manifest(rows):
    if not rows or len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("Empty manifest or duplicate source IDs")
    hashes = {}
    for row in rows:
        if CLASS_MAPPING.get(row["class_name"]) != row["class_id"]:
            raise ValueError("Unexpected class or numeric label")
        path = Path(row["relative_path"])
        if path.is_absolute() or ".." in path.parts or path.parts[0] != row["class_name"]:
            raise ValueError("Unsafe source path or class/path mismatch")
        previous = hashes.setdefault(row["file_sha256"], row["class_id"])
        if previous != row["class_id"]:
            raise ValueError("Identical video bytes have conflicting class labels")


def discover(root):
    import cv2
    root = Path(root).resolve()
    rows = []
    for name, class_id in CLASS_MAPPING.items():
        directory = root / name
        if not directory.is_dir():
            raise ValueError(f"Missing expected class directory: {name}")
        for path in sorted(directory.iterdir(), key=lambda value: value.name):
            if not path.is_file() or path.suffix.lower() != ".avi":
                raise ValueError(f"Unexpected dataset entry: {path}")
            relative = path.relative_to(root).as_posix()
            file_hash = sha256(path)
            source_id = hashlib.sha256(relative.encode()).hexdigest()[:20]
            capture = cv2.VideoCapture(str(path))
            try:
                opened = capture.isOpened()
                properties = [capture.get(key) for key in (cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT)]
            finally:
                capture.release()
            row = dict.fromkeys(FIELDS, "")
            row.update(source_id=source_id, class_name=name, class_id=class_id, relative_path=relative, file_sha256=file_hash,
                       nominal_fps=properties[0], width=int(properties[1]), height=int(properties[2]), header_frame_count=int(properties[3]),
                       group_id=file_hash, preprocessing_status="pending" if opened else "header_failed", notes="" if opened else "capture could not open")
            rows.append(row)
    rows.sort(key=lambda row: row["source_id"])
    validate_manifest(rows)
    return rows
