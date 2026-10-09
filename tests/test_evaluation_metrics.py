"""Hand-computed metrics, fixed class order and equal video weighting."""

import numpy as np
import pytest

from evaluation.metrics import aggregate_videos, classification_metrics


def test_metrics_macro_weighted_orientation_and_zero_support():
    result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 0, 0]]
    assert result["class_order"] == ["Normal", "Yawning", "Eye Close"]
    assert result["accuracy"] == .75
    assert result["macro_precision"] == pytest.approx((1+2/3+0)/3)
    assert result["macro_recall"] == .5
    assert result["macro_f1"] == pytest.approx((2/3+.8)/3)
    assert result["weighted_f1"] == pytest.approx((2/3+.8)/2)
    assert result["per_class"]["Eye Close"]["support"] == result["per_class"]["Eye Close"]["f1"] == 0


def test_probability_mean_per_video_not_majority_or_long_video_weight():
    result = aggregate_videos([0, 0, 0, 1], np.array([[.51, .49, 0], [.51, .49, 0], [0, 1, 0], [0, 1, 0]]), ["a", "a", "a", "b"])
    assert result["support"] == 2 and result["accuracy"] == .5
    assert result["videos"][0]["predicted_class_id"] == 1
    assert result["videos"][0]["window_count"] == 3


@pytest.mark.parametrize("labels,predictions", [([], []), ([3], [0]), ([0], [0, 1])])
def test_invalid_metric_inputs_fail(labels, predictions):
    with pytest.raises(ValueError):
        classification_metrics(labels, predictions)


def test_conflicting_labels_within_video_fail():
    with pytest.raises(ValueError, match="conflicting"):
        aggregate_videos([0, 1], np.eye(3)[[0, 1]], ["a", "a"])
