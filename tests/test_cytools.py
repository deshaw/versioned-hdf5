import math

from hypothesis import given
from hypothesis import strategies as st

from versioned_hdf5.cytools import (
    ceil_a_over_b,
    count2stop,
    smallest_step_after,
    stop2count,
)


def free_slices_st(size: int):
    """Hypothesis draw of a slice object to slice an array of up to size elements"""
    start_st = st.integers(0, size)
    stop_st = st.integers(0, size)
    # only non-negative steps are allowed
    step_st = st.integers(1, size)
    return st.builds(slice, start_st, stop_st, step_st)


@given(free_slices_st(5))
def test_stop2count_count2stop(s):
    count = stop2count(s.start, s.stop, s.step)
    assert count == len(range(s.start, s.stop, s.step))

    stop = count2stop(s.start, count, s.step)
    # When count == 0 or when step>1, multiple stops yield the same count,
    # so stop won't necessarily be equal to s.stop
    assert count == len(range(s.start, stop, s.step))


@given(st.integers(0, 10), st.integers(1, 10))
def test_ceil_a_over_b(a, b):
    expect = math.ceil(a / b)
    actual = ceil_a_over_b(a, b)
    assert actual == expect


@given(st.data(), st.integers(0, 10), st.integers(1, 4))
def test_smallest_step_after(data, x, m):
    a = data.draw(st.integers(0, x))
    expect = a
    while expect < x:
        expect += m
    actual = smallest_step_after(x, a, m)
    assert actual == expect
