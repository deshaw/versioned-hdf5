import numpy as np


def asarray(a, dtype=None):
    """Variant of np.asarray(a, dtype=dtype), with two differences:

    1. If a is a numpy-like array, don't coerce it to a numpy.ndarray
    2. If a has a ABI-compatible dtype, return a view instead of a copy
       (works around https://github.com/numpy/numpy/issues/27509)
    """
    if not hasattr(a, "__array__") or np.isscalar(a):
        return np.asarray(a, dtype=dtype)

    if dtype is None:
        return a

    dtype = np.dtype(dtype)
    if a.dtype == dtype:
        return a

    if (
        dtype.itemsize == a.itemsize
        and dtype.kind in ("i", "u")
        and a.dtype.kind in ("i", "u")
        and hasattr(a, "view")
    ):
        # Note that this does not reduce the amount of safety checks:
        # np.array(-1).astype("u1") doesn't raise and returns 255!
        return a.view(dtype)

    return a.astype(dtype)
