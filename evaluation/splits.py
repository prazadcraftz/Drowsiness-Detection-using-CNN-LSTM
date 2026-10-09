"""Stratify raw-video content groups before extraction/learned artifacts."""

import hashlib
import json
import random

from . import CLASS_NAMES, SEED
from .manifest import validate_manifest

SPLITS = ("train", "validation", "test")


def assignment_hash(assignments):
    return hashlib.sha256(json.dumps(assignments, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def create_splits(rows, seed=SEED):
    validate_manifest(rows)
    assignments = {}
    rng = random.Random(seed)
    for name in CLASS_NAMES:
        groups = sorted({row["group_id"] for row in rows if row["class_name"] == name})
        if len(groups) < 3:
            raise ValueError("Need at least three independent video groups per class")
        rng.shuffle(groups)
        validation_count = max(1, round(len(groups) * .15))
        test_count = max(1, round(len(groups) * .15))
        train_count = len(groups) - validation_count - test_count
        if train_count < 1:
            raise ValueError("Not enough training groups")
        group_split = {group: ("train" if index < train_count else "validation" if index < train_count + validation_count else "test") for index, group in enumerate(groups)}
        assignments.update({row["source_id"]: group_split[row["group_id"]] for row in rows if row["class_name"] == name})
    result = {"seed": seed, "strategy": "class-stratified raw-video content groups; exact duplicate SHA256 bytes co-grouped; no subject IDs",
              "proportions": {"train": .70, "validation": .15, "test": .15}, "assignments": dict(sorted(assignments.items())),
              "source_ids": {split: sorted(source for source, value in assignments.items() if value == split) for split in SPLITS},
              "class_video_counts": {split: {name: sum(row["class_name"] == name and assignments[row["source_id"]] == split for row in rows) for name in CLASS_NAMES} for split in SPLITS},
              "assignment_sha256": assignment_hash(assignments)}
    validate_splits(rows, result)
    return result


def validate_splits(rows, splits):
    assignments = splits["assignments"]
    if set(assignments) != {row["source_id"] for row in rows} or any(value not in SPLITS for value in assignments.values()):
        raise ValueError("Missing/extra source or invalid split")
    groups = {}
    for row in rows:
        old = groups.setdefault(row["group_id"], assignments[row["source_id"]])
        if old != assignments[row["source_id"]]:
            raise ValueError("Video group crosses splits: leakage")
    if assignment_hash(assignments) != splits["assignment_sha256"]:
        raise ValueError("Split assignment hash mismatch")
    for split in SPLITS:
        if sorted(source for source, value in assignments.items() if value == split) != splits["source_ids"][split]:
            raise ValueError("Split source-ID list differs from assignments")
