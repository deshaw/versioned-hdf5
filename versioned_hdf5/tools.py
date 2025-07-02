from __future__ import annotations

from typing import Any

import ndindex
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from versioned_hdf5.typing_ import ArrayProtocol

NP_VERSION = tuple(int(i) for i in np.__version__.split(".")[:3])


def asarray(a: ArrayLike, /, *, dtype: DTypeLike | None = None):
    """Variant of np.asarray(a, dtype=dtype), with some differences:

    1. If a is a numpy-like array, don't coerce it to a numpy.ndarray
    2. If a has a ABI-compatible dtype, return a view instead of a copy
       (works around https://github.com/numpy/numpy/issues/27509)
    3. Work around https://github.com/numpy/numpy/issues/28269
       on NumPy >=2.0.0,<2.2.3 when converting from arrays of object strings to
       NpyStrings
    """
    if not isinstance(a, ArrayProtocol) or np.isscalar(a):
        return np.asarray(a, dtype=dtype)

    if dtype is None:
        return a

    dtype = np.dtype(dtype)
    if a.dtype == dtype:
        return a

    if (
        dtype.itemsize == a.dtype.itemsize
        and dtype.kind in ("i", "u")
        and a.dtype.kind in ("i", "u")
        and hasattr(a, "view")
    ):
        # Note that this does not reduce the amount of safety checks:
        # np.array(-1).astype("u1") doesn't raise and returns 255!
        return a.view(dtype)

    if NP_VERSION < (2, 2, 3) and a.dtype.kind == "O" and dtype.kind == "T":
        # Work around bug in conversion from array of bytes objects to NpyStrings
        # https://github.com/numpy/numpy/issues/28269
        # Note that this can be memory intensive.
        return asarray(asarray(a, dtype="U"), dtype=dtype)

    if hasattr(a, "astype"):
        return a.astype(dtype)
    return np.asarray(a, dtype=dtype)


def ix_with_slices(*idx: Any, shape: tuple[int, ...]) -> tuple:
    """Variant of numpy.ix_ with added support for mixing with slices.

    Given a numpy ndindex where each element could be either a 1D fancy integer
    array index or a slice, convert it to an index that numpy understands as 'for
    each axis, take these indices'.

    Scalar indices, None and np.newaxis are NOT supported.

    Example
    -------
    >>> a = np.arange(3 * 3 * 2).reshape(3, 3, 2)
    >>> a
    array([[[ 0,  1],
            [ 2,  3],
            [ 4,  5]],
           [[ 6,  7],
            [ 8,  9],
            [10, 11]],
           [[12, 13],
            [14, 15],
            [16, 17]]])
    >>> # Select planes 1 and 2, then the first 2 rows, then columns 1 and 0
    >>> a[[1, 2], :, :][: ,:2, :][:, :, [1, 0]]
    array([[[ 7,  6],
            [ 9,  8]],
           [[13, 12],
            [15, 14]]])
    >>> a[np.ix_([1, 2], [0, 1], [1, 0])]
    array([[[ 7,  6],
            [ 9,  8]],
           [[13, 12],
            [15, 14]]])
    >>> a[ix_with_slices([1, 2], slice(2), [1, 0], shape=a.shape)]
    array([[[ 7,  6],
            [ 9,  8]],
           [[13, 12],
            [15, 14]]])
    """
    if len(idx) < 2:
        return idx

    # A single fancy index can be mixed with slices without problems
    n_fancy = 0
    for i in idx:
        if not isinstance(i, slice):
            if i is None or i is np.newaxis or np.isscalar(i):
                raise NotImplementedError("Scalars, None and newaxis are not supported")
            n_fancy += 1
    if n_fancy < 2:
        return idx

    idx2 = []
    for axis, i in enumerate(idx):
        if isinstance(i, slice):
            i = ndindex.Slice(i).reduce(shape[axis])
            i = np.arange(i.start, max(-1, i.stop), i.step)
        idx2.append(i)

    return np.ix_(*idx2)


def format_ndindex(idx: Any) -> str:
    """Format a numpy ndindex for pretty-printing.

    >>> format_ndindex(slice(None))
    ':'
    >>> format_ndindex(slice(None, 10, 2))
    ':10:2'
    >>> format_ndindex(())
    '()'
    >>> format_ndindex((1, slice(2, 3, 1), np.array([4, 5, 6])))
    '1, 2:3, [4, 5, 6]'
    """
    if isinstance(idx, tuple):
        if idx == ():
            return "()"
    else:
        idx = (idx,)

    idx_s = []
    for i in idx:
        if isinstance(i, slice):
            start = "" if i.start is None else i.start
            stop = "" if i.stop is None else i.stop
            step = "" if i.step in (1, None) else f":{i.step}"
            idx_s.append(f"{start}:{stop}{step}")
        elif isinstance(i, np.ndarray):
            idx_s.append(str(i.tolist()))
        else:
            idx_s.append(str(i))
    return ", ".join(idx_s)
