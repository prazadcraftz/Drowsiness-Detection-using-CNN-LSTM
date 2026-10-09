"""Train-only extrema, historical stride/padding, source/observation isolation."""

import numpy as np
import pytest

from evaluation.preprocess import create_windows, fit_train_normalization, verify_integrity
from evaluation.splits import create_splits
from test_evaluation_manifest import rows


def test_test_only_extreme_cannot_change_train_normalizer():
    source = rows(3)
    splits = create_splits(source)
    features = {row["source_id"]: np.stack([np.ones(5), 2*np.ones(5)]) for row in source}
    before, fitted_ids = fit_train_normalization(features, source, splits["assignments"])
    for source_id in splits["source_ids"]["test"] + splits["source_ids"]["validation"]:
        features[source_id] = np.full((2, 5), 1e9)
    after, after_ids = fit_train_normalization(features, source, splits["assignments"])
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(before, np.tile([0, 2], (5, 1)))
    assert fitted_ids == after_ids == splits["source_ids"]["train"]


def test_zero_denominators_and_nonfinite_train_fail():
    source = rows(3)
    splits = create_splits(source)
    for array in (np.zeros((2, 5)), np.full((2, 5), np.nan)):
        with pytest.raises(ValueError):
            fit_train_normalization({row["source_id"]: array for row in source}, source, splits["assignments"])


def test_historical_stride_eight_orientation_and_two_tail_windows():
    features = np.arange(17*5, dtype=float).reshape(17, 5)
    ranges = np.tile([0, 100], (5, 1))
    windows, identities, padded = create_windows(features, np.arange(17), ranges)
    assert windows.shape == (3, 5, 16) and padded.tolist() == [False, True, True]
    np.testing.assert_array_equal(windows[0], (features[:16]/100).T)
    np.testing.assert_array_equal(windows[0, :, 8:], windows[1, :, :8])
    np.testing.assert_array_equal(windows[2], np.repeat((features[-1]/100)[:, None], 16, axis=1))
    assert identities[2].tolist() == [16]*16


def fixture_data():
    source = rows(3)
    splits = create_splits(source)
    datasets = {}
    ranges = np.tile([0, 2], (5, 1))
    for split in ("train", "validation", "test"):
        windows, frames, padded, ids, labels = [], [], [], [], []
        for row in source:
            if splits["assignments"][row["source_id"]] == split:
                w, f, p = create_windows(np.ones((20, 5)), np.arange(20), ranges)
                windows.extend(w); frames.extend(f); padded.extend(p)
                ids.extend([row["source_id"]]*len(w)); labels.extend([row["class_id"]]*len(w))
        datasets[split] = {"windows": np.asarray(windows), "frame_indices": np.asarray(frames), "padded": np.asarray(padded), "source_ids": np.asarray(ids), "labels": np.asarray(labels)}
    return source, splits, datasets


def test_overlapping_windows_stay_with_video_and_observation_ids_disjoint():
    source, splits, data = fixture_data()
    assert verify_integrity(source, splits, data, splits["source_ids"]["train"])["passed"]
    # Every video has overlapping windows, all in one split, even though frame
    # numbers repeat in unrelated videos; identities include the source ID.
    for split, dataset in data.items():
        assert set(dataset["source_ids"].tolist()) == set(splits["source_ids"][split])


def test_injected_cross_split_window_and_normalization_leakage_fail():
    source, splits, data = fixture_data()
    with pytest.raises(ValueError, match="Normalization"):
        verify_integrity(source, splits, data, splits["source_ids"]["train"] + splits["source_ids"]["test"][:1])
    data["test"]["source_ids"][0] = data["train"]["source_ids"][0]
    with pytest.raises(ValueError):
        verify_integrity(source, splits, data, splits["source_ids"]["train"])
