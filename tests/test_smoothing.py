import numpy as np
import pytest

from driver_monitor.inference.smoothing import PredictionSmoother


def test_empty_single_fill_and_probability_average():
    smoother = PredictionSmoother(3)
    assert smoother.current is None and len(smoother) == 0
    first = smoother.update([.8, .1, .1])
    assert first.class_id == 0 and first.confidence == .8 and first.history_fill == 1
    smoother.update([.6, .2, .2])
    result = smoother.update([.1, .2, .7])
    np.testing.assert_allclose(result.probabilities, [.5, 1/6, 1/3])
    assert result.probabilities.sum() == pytest.approx(1)
    result = smoother.update([0, 0, 1])
    np.testing.assert_allclose(result.probabilities, np.mean([[.6,.2,.2],[.1,.2,.7],[0,0,1]],axis=0))
    assert result.history_fill == len(smoother) == smoother.capacity == 3


def test_majority_vote_differs_from_probability_average():
    average = PredictionSmoother(3)
    vote = PredictionSmoother(3, "majority_vote")
    for vector in [[.34,.33,.33],[.34,.33,.33],[0,0,1]]:
        mean = average.update(vector)
        majority = vote.update(vector)
    assert mean.class_id == 2 and majority.class_id == 0
    np.testing.assert_array_equal(mean.probabilities, majority.probabilities)
    assert majority.confidence == majority.probabilities[0]


def test_vote_tie_uses_averaged_probabilities_then_smallest_id():
    smoother = PredictionSmoother(2, "majority_vote")
    smoother.update([.8,.1,.1])
    assert smoother.update([.1,.2,.7]).class_id == 0
    smoother.clear()
    smoother.update([1,0,0])
    assert smoother.update([0,1,0]).class_id == 0


@pytest.mark.parametrize("vector", [[.1,.2], [[.1,.2,.7]], [.1,.2,.3], [np.nan,0,1],
                                     [np.inf,0,0], [-.1,.1,1], [1.1,0,-.1], [1j,0,0],
                                     [".1",".2",".7"], [True,False,False]])
def test_invalid_probabilities_rejected_without_append(vector):
    smoother = PredictionSmoother()
    smoother.update([1,0,0])
    with pytest.raises(ValueError):
        smoother.update(vector)
    assert len(smoother)==1


def test_sum_tolerance_reset_and_snapshot_ownership():
    smoother = PredictionSmoother()
    vector = np.array([.6,.3,.1000001])
    result = smoother.update(vector)
    vector[:] = 0
    result.probabilities[:] = 0
    assert smoother.current.probabilities.sum() == pytest.approx(1,abs=1e-6)
    with pytest.raises(ValueError,match="sum"):
        smoother.update([.6,.3,.10001])
    smoother.clear()
    assert smoother.current is None and smoother.storage_bytes == 0


@pytest.mark.parametrize("size,mode", [(0,"probability_average"),(-1,"probability_average"),
                                       (1.2,"majority_vote"),(True,"majority_vote"),(5,"unknown")])
def test_invalid_smoother_configuration(size,mode):
    with pytest.raises(ValueError):
        PredictionSmoother(size,mode)


def test_one_hundred_thousand_predictions_are_bounded(record_property):
    smoother = PredictionSmoother()
    maximum = 0
    for _ in range(100000):
        smoother.update([.9,.05,.05])
        maximum=max(maximum,len(smoother))
    assert maximum == 5 and smoother.storage_bytes == 5*3*8
    record_property("phase6_prediction_count",100000)
    record_property("phase6_max_retained_predictions",maximum)
    record_property("phase6_smoother_payload_bytes",smoother.storage_bytes)
