from __future__ import annotations

import cython

if cython.compiled:  # pragma: nocover
    # This is repeated here from the header in order to silence mypy, which now treats
    # hsize_t as Any, without breaking cython.
    from cython.cimports.versioned_hdf5.cytools import hsize_t

# numpy equivalent dtype for hsize_t
from numpy import uint64 as np_hsize_t  # noqa: F401


@cython.ccall
@cython.nogil
@cython.exceptval(check=False)
def stop2count(start: hsize_t, stop: hsize_t, step: hsize_t) -> hsize_t:
    """Given a start:stop:step slice or range, return the number of elements yielded.

    This is functionally identical to::

        len(range(start, stop, step))

    Doesn't assume that stop >= start. Assumes that step >= 1.
    """
    # Note that hsize_t is unsigned so stop - start could underflow.
    if stop <= start:
        return 0
    return (stop - start - 1) // step + 1


@cython.ccall
@cython.nogil
@cython.exceptval(check=False)
def count2stop(start: hsize_t, count: hsize_t, step: hsize_t) -> hsize_t:
    """Inverse of stop2count.

    When count == 0 or when step>1, multiple stops can yield the same count.
    This function returns the smallest stop >= start.
    """
    if count == 0:
        return start
    return start + (count - 1) * step + 1


@cython.ccall
@cython.nogil
@cython.exceptval(check=False)
def ceil_a_over_b(a: hsize_t, b: hsize_t) -> hsize_t:
    """Returns ceil(a/b). Assumes a >= 0 and b > 0.

    Note
    ----
    This module is compiled with the cython.cdivision flag and the data type is
    unsigned. This causes behaviour to change if one accidentally passes negative a or b
    when running in pure python mode.
    """
    return a // b + (a % b > 0)


@cython.ccall
@cython.nogil
@cython.exceptval(check=False)
def smallest_step_after(x: hsize_t, a: hsize_t, m: hsize_t) -> hsize_t:
    """Find the smallest integer y >= x where y = a + k*m for whole k's
    Assumes 0 <= a <= x and m >= 1.

    a                  x    y
    | <-- m --> | <-- m --> |
    """
    return a + ceil_a_over_b(x - a, m) * m
