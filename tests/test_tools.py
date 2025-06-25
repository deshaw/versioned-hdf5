import hypothesis
import numpy as np
import numpy.ma as ma
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as stnp
from numpy.testing import assert_array_equal

from versioned_hdf5.tools import asarray, ix_with_slices

from .test_typing import MinimalArray


def test_asarray():
    a = np.array([1, -1], dtype="i2")
    b = asarray(a)
    assert b is a
    b = asarray(a, dtype="i2")
    assert b is a
    b = asarray(a, dtype="u2")
    assert_array_equal(b, a.astype("u2"), strict=True)
    assert b.base is a
    b = asarray(a, dtype="i4")
    assert_array_equal(b, a.astype("i4"), strict=True)
    assert b.base is None

    # Don't just test itemsize
    a = np.array([1, -1], dtype="i4")
    b = asarray(a, dtype="f4")
    assert_array_equal(b, a.astype("f4"), strict=True)
    assert b.base is None

    # non-arrays are coerced to np.ndarray
    a = [1, -1]
    b = asarray(a)
    assert_array_equal(b, np.asarray(a), strict=True)
    b = asarray(a, dtype=np.float32)
    assert_array_equal(b, np.asarray(a, dtype=np.float32), strict=True)

    a = 1
    b = asarray(a)
    assert isinstance(b, np.ndarray)
    assert_array_equal(b, np.asarray(a), strict=True)

    a = np.int16(1)
    b = asarray(a)
    assert isinstance(b, np.ndarray)
    assert_array_equal(b, np.asarray(a), strict=True)

    # array-likes aren't coerced to np.ndarray
    a = ma.masked_array([1, -1], mask=[0, 1], dtype="i2")
    b = asarray(a)
    assert b is a
    b = asarray(a, dtype="u2")
    assert type(b) is type(a)
    assert b.base is a
    assert_array_equal(b, a.astype("u2"), strict=True)
    b = asarray(a, dtype="i4")
    assert type(b) is type(a)
    assert b.base is None
    assert_array_equal(b, a.astype("i4"), strict=True)

    a = MinimalArray(np.asarray([1], dtype="i8"))
    b = asarray(a)
    assert b is a
    b = asarray(a, dtype="i8")
    assert b is a


@pytest.mark.skipif(np.__version__ < "2", reason="StringDType requires NumPy >=2.0")
def test_asarray_np2_strings():
    """Test workaround to bug when converting from object array of bytes to NpyStrings
    This issue affected NumPy >=2.0.0,<2.2.3
    https://github.com/numpy/numpy/issues/28269

    TODO remove this test when NumPy >=2.2.3 is required
    """
    a = np.array([b"foo", b"bar"], dtype="O")
    b = asarray(a, dtype="T")
    assert_array_equal(b, np.array(["foo", "bar"], dtype="T"), strict=True)


@st.composite
def ix_idx_st(draw, max_ndim: int = 4) -> tuple[tuple[int, ...], tuple]:
    """Hypothesis draw of slice, integer array, and boolean array indices, with no
    limitation on the number of fancy indices
    """

    def slices(size: int):
        start = st.one_of(st.none(), st.integers(-size - 1, size + 1))
        stop = st.one_of(st.none(), st.integers(-size - 1, size + 1))
        step = st.one_of(
            st.none(), st.integers(1, size + 1), st.integers(-size - 1, -1)
        )
        return st.builds(slice, start, stop, step)

    def fancy(size: int):
        return stnp.arrays(
            np.intp,
            shape=st.integers(0, size + 1),
            elements=st.integers(-size, size - 1),
            unique=False,
        )

    def mask(size: int):
        return stnp.arrays(np.bool_, shape=size, elements=st.booleans())

    shape_st = st.lists(st.integers(1, 10), min_size=1, max_size=max_ndim)
    shape = tuple(draw(shape_st))

    idx_st = st.tuples(
        *(st.one_of(slices(size), fancy(size), mask(size)) for size in shape)
    )
    return shape, draw(idx_st)


@given(ix_idx_st())
@hypothesis.settings(max_examples=1000)
def test_ix_with_slices(args):
    shape, idx = args
    a = np.arange(np.prod(shape)).reshape(shape)
    expect = a
    for axis, i in enumerate(idx):
        expect = expect[(slice(None),) * axis + (i,)]

    actual = a[ix_with_slices(*idx, shape=shape)]
    assert_array_equal(actual, expect)
