import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal

from ..tools import asarray


def test_asarray():
    a = np.array([1, -1], dtype="i2")
    b = asarray(a)
    assert b is a
    b = asarray(a, "i2")
    assert b is a
    b = asarray(a, "u2")
    assert_array_equal(b, a.astype("u2"), strict=True)
    assert b.base is a
    b = asarray(a, "i4")
    assert_array_equal(b, a.astype("i4"), strict=True)
    assert b.base is None

    # Don't just test itemsize
    a = np.array([1, -1], dtype="i4")
    b = asarray(a, "f4")
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
