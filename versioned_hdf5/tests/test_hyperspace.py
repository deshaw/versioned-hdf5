import hypothesis
import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ..hyperspace import empty_view, fill_hyperspace, view_from_tuple


@pytest.mark.parametrize("n", [0, 1, 2])
def test_empty_view(n):
    l = list(range(n))
    v = empty_view(n)
    for i in range(n):
        v[i] = l[i]
    npt.assert_array_equal(v, np.asarray(l, dtype=np.intp), strict=True)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_view_from_tuple(n):
    t = tuple(range(n))
    v = view_from_tuple(t)
    npt.assert_array_equal(v, np.asarray(t, dtype=np.intp), strict=True)


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_fill_empty_hyperspace(ndim):
    shape = (3,) * ndim
    obstacles = np.empty((0, ndim), dtype=np.intp)
    actual = fill_hyperspace(obstacles, shape)
    # Exactly one hyperrectangle can cover the whole hyperspace
    expect = np.array([(0,) * ndim + shape], dtype=np.intp)
    npt.assert_array_equal(actual, expect, strict=True)


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_fill_full_hyperspace(ndim):
    shape = (3,) * ndim
    obstacles = np.array(np.nonzero(np.ones(shape))).T
    actual = fill_hyperspace(obstacles, shape)
    # The hyperspace is fully covered by the obstacles so no hyperrectangles are needed
    expect = np.empty((0, ndim * 2), dtype=np.intp)
    npt.assert_array_equal(actual, expect, strict=True)


@given(st.integers(1, 4), st.data())
def test_fill_hyperspace_size_zero(ndim, data):
    shape = [10] * ndim
    shape[data.draw(st.integers(0, ndim - 1))] = 0
    obstacles = np.empty((0, ndim), dtype=np.intp)
    hyperrectangles = fill_hyperspace(obstacles, tuple(shape))
    expect = np.empty((0, ndim * 2), dtype=np.intp)
    npt.assert_array_equal(hyperrectangles, expect)


@given(
    # shape from (1, ) to (4, 4, 4, 4)
    shape=st.lists(st.integers(1, 4), min_size=1, max_size=4),
    data=st.data(),
)
@hypothesis.settings(database=None, max_examples=1000, deadline=None)
def test_fill_hyperspace(shape, data):
    ndim = len(shape)
    arr = np.zeros(shape)

    obstacles_flat = data.draw(st.lists(st.integers(0, arr.size - 1), unique=True))
    arr.flat[obstacles_flat] = 1
    obstacles = np.asarray(np.nonzero(arr)).T
    hyperrectangles = np.asarray(fill_hyperspace(obstacles, tuple(shape)))

    # Test that all coordinates are within the shape
    assert (hyperrectangles >= 0).all()
    for axis in range(ndim):
        assert (hyperrectangles[:, axis] <= shape[axis]).all()
        assert (hyperrectangles[:, axis + ndim] <= shape[axis]).all()
        # Test that all hyperrectangles have at least size 1
        assert (hyperrectangles[:, axis] < hyperrectangles[:, axis + ndim]).all()

    # Fill the empty space
    for rect in hyperrectangles:
        idx = tuple(slice(start, stop) for start, stop in zip(rect[:ndim], rect[ndim:]))
        assert (arr[idx] == 0).all(), "overlapping hyperrectangles"
        arr[idx] = 1
    assert (arr == 1).all(), "incomplete coverage"
