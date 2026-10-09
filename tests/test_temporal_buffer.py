from collections import deque

import numpy as np
import pytest

from driver_monitor.inference.temporal_buffer import TemporalBuffer


def test_empty_buffer_and_first_sixteen_observations():
    buffer = TemporalBuffer()
    assert len(buffer) == 0 and not buffer.ready and buffer.capacity == 16
    with pytest.raises(RuntimeError, match="warming up: 0/16"):
        buffer.window()
    for index in range(1, 17):
        buffer.append(np.arange(5, dtype=float) + index)
        assert len(buffer) == index
        assert buffer.ready == (index == 16)
    expected = np.stack([np.arange(5) + index for index in range(1, 17)], axis=1)
    np.testing.assert_array_equal(buffer.window(), expected)
    assert buffer.window().shape == (5, 16)


@pytest.mark.parametrize("values", [np.zeros((5, 1)), np.zeros((1, 5)), np.zeros(4),
                                    np.zeros(6), np.zeros((5, 16)), [1, 2, 3, 4, np.nan],
                                    [1, 2, 3, 4, np.inf], [1, 2, 3, 4, -np.inf],
                                    np.full(5, 1 + 2j)])
def test_invalid_observation_does_not_mutate_buffer(values):
    buffer = TemporalBuffer()
    buffer.append(np.zeros(5))
    with pytest.raises(ValueError, match="shape|finite"):
        buffer.append(values)
    assert len(buffer) == 1


def test_stride_one_and_snapshot_ownership():
    buffer = TemporalBuffer()
    supplied = np.zeros(5)
    buffer.append(supplied)
    supplied[:] = 1000  # caller mutation must not modify the already appended value
    for index in range(1, 16):
        buffer.append(np.full(5, index))
    first = buffer.window()
    buffer.append(np.full(5, 16))
    second = buffer.window()
    buffer.append(np.full(5, 17))
    third = buffer.window()
    np.testing.assert_array_equal(first, np.tile(np.arange(16), (5, 1)))
    np.testing.assert_array_equal(second, np.tile(np.arange(1, 17), (5, 1)))
    np.testing.assert_array_equal(third, np.tile(np.arange(2, 18), (5, 1)))
    np.testing.assert_array_equal(first[:, 1:], second[:, :-1])
    first[:] = -999
    np.testing.assert_array_equal(buffer.window(), third)
    assert len(buffer) == 16
    assert isinstance(buffer._observations, deque)
    assert buffer._observations.maxlen == 16
