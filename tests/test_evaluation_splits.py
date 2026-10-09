"""Leakage regression: raw video groups, not overlapping windows, are split."""

from copy import deepcopy

import pytest

from evaluation.splits import assignment_hash, create_splits, validate_splits
from test_evaluation_manifest import rows


def test_fixed_split_reproducibility_class_counts_and_disjoint_ids():
    source = rows()
    first, second = create_splits(source, 20260927), create_splits(source[::-1], 20260927)
    assert first == second
    assert create_splits(source, 123)["assignments"] != first["assignments"]
    for split, count in (("train", 42), ("validation", 9), ("test", 9)):
        assert set(first["class_video_counts"][split].values()) == {count}
    assert not set(first["source_ids"]["train"]) & set(first["source_ids"]["test"])


def test_identical_video_content_is_grouped():
    source = rows(6)
    source[1]["group_id"] = source[0]["group_id"]
    source[1]["file_sha256"] = source[0]["file_sha256"]
    splits = create_splits(source)
    assert splits["assignments"][source[0]["source_id"]] == splits["assignments"][source[1]["source_id"]]


def test_tampered_assignment_or_missing_source_fails():
    source = rows(6)
    splits = create_splits(source)
    splits["assignments"].pop(source[0]["source_id"])
    with pytest.raises(ValueError):
        validate_splits(source, splits)


def test_cross_group_split_leakage_is_a_hard_failure():
    source = rows(6)
    source[1]["group_id"] = source[0]["group_id"]
    splits = create_splits(source)
    splits["assignments"][source[1]["source_id"]] = "test" if splits["assignments"][source[0]["source_id"]] != "test" else "train"
    splits["assignment_sha256"] = assignment_hash(splits["assignments"])
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(source, splits)
