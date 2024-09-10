# Note: this entire module is compiled by cython with wraparound=False
# See meson.build for details

from __future__ import annotations

import array

import cython
import numpy as np
from cython import Py_ssize_t

_empty_tpl = array.array("l", [])
_py_ssize_t_nbytes = _empty_tpl.itemsize
# assert that "long" is the same length as void* on the current platform
# Note: Py_ssize_t is always the same as np.intp
assert np.intp().nbytes == _py_ssize_t_nbytes

if cython.compiled:  # pragma: nocover
    from cython.cimports.cpython import array  # type: ignore

    @cython.ccall
    @cython.inline
    def empty_view(n: Py_ssize_t) -> Py_ssize_t[:]:
        """Functionally the same, but faster, as

        v: Py_ssize_t[:] = np.empty(n, dtype=np.intp)

        Note that this is limited to one dimension.
        """
        # array.clone exists only in compiled Cython
        return array.clone(_empty_tpl, n, zero=False)  # type: ignore[attr-defined]

    @cython.ccall
    @cython.inline
    def view_from_tuple(t: tuple[int, ...]) -> Py_ssize_t[:]:
        """Functionally the same, but faster, as

        v: Py_ssize_t[:] = array.array("l", t)
        """
        n = len(t)
        v: Py_ssize_t[:] = empty_view(n)
        for i in range(n):
            v[i] = t[i]
        return v

else:

    def empty_view(n: Py_ssize_t) -> Py_ssize_t[:]:
        return array.array("l", b" " * _py_ssize_t_nbytes * n)

    def view_from_tuple(t: tuple[int, ...]) -> Py_ssize_t[:]:
        return array.array("l", t)


@cython.ccall
def fill_hyperspace(
    obstacles: Py_ssize_t[:, :],
    shape: tuple[int, ...],
) -> Py_ssize_t[:, :]:
    """Given a N-dimensional space of the given shape and a series of obstacles each one
    point in size, generate hyperrectangles that cover the whole remaining space
    without ever crossing an obstacle or another hyperrectangle.

    Parameters
    ----------
    obstacles:
        A list of points to avoid, for example as returned by
        versioned_hdf5.subchunk_map._modified_chunks_in_selection
    shape:
        The shape of the space to be covered

    Returns
    -------
    Edges of hyperrectangles, one hyperrectangle per row, with the first half of the
    columns being the coordinates of the top-left corner (included) and the right half
    of the columns being the coordinates of the bottom-right corner (excluded).

    See Also
    --------
    https://en.wikipedia.org/wiki/Dimension#High-dimensional_space
    https://en.wikipedia.org/wiki/Hyperrectangle

    Example
    -------
    >>> obstacles = np.array([(2, 4), (2, 5), (5, 7)])
    >>> np.asarray(fill_hyperspace(obstacles, shape=(8, 10)))
    array([[ 0,  0,  2, 10],
           [ 2,  0,  3,  4],
           [ 2,  6,  3, 10],
           [ 3,  0,  5, 10],
           [ 5,  0,  6,  7],
           [ 5,  8,  6, 10],
           [ 6,  0,  8, 10]])

        Input         Output
      0123456789    0123456789
    0 ..........  0 aaaaaaaaaa
    1 ..........  1 aaaaaaaaaa
    2 ....XX....  2 bbbbXXcccc
    3 ..........  3 dddddddddd
    4 ..........  4 dddddddddd
    5 .......X..  5 eeeeeeeXff
    6 ..........  6 gggggggggg
    7 ..........  7 gggggggggg

    FIXME
    -----
    The current implementation performs suboptimally in the common use case of an
    uninterrupted sequence of obstacles along an axis that is not the innermost one:

        Input     Current output  Optimal output
      0123456789     0123456789      0123456789
    0 .X........   0 aXbbbbbbbb    0 aXbbbbbbbb
    1 .X........   1 cXdddddddd    1 aXbbbbbbbb
    2 .X........   2 eXffffffff    2 aXbbbbbbbb
    """
    shape_v = view_from_tuple(shape)
    nobs = len(obstacles)
    ndim = len(shape_v)
    assert ndim > 0

    # For ndim=1, there can be only a single 1d segment between two obstacles
    # For ndim=2, there can be 1x 2d rectangles and 2x 1d segments
    # For ndim=3, there can be 1x 3d cuboid, 2x 2d rectangles, and 4x 1d lines
    # etc.
    max_spaces = ((1 << ndim) - 1) * (nobs + 2)

    out: Py_ssize_t[:, :] = np.empty((max_spaces, ndim * 2), dtype=np.intp)

    for n in shape_v:
        if n == 0:
            return out[:0, :]

    out_a = out[:, :ndim]
    out_b = out[:, ndim:]
    cur = 0

    for i in range(nobs + 1):
        if i == 0:
            a = _one_before(shape_v)
        else:
            a = obstacles[i - 1, :]
        if i == nobs:
            b = _one_after(shape_v)
        else:
            b = obstacles[i, :]

        cur = _hyperrectangles_between(a, b, shape_v, out_a, out_b, cur)

    return out[:cur, :]


@cython.cfunc
@cython.exceptval(check=False)
def _hyperrectangles_between(
    obstacle1: Py_ssize_t[:],
    obstacle2: Py_ssize_t[:],
    shape: Py_ssize_t[:],
    out_a: Py_ssize_t[:, :],
    out_b: Py_ssize_t[:, :],
    cur: Py_ssize_t,
) -> Py_ssize_t:
    """Helper of fill_hyperspace.

    Given two ordered points in a N-dimensional hyperspace, yield the top-left
    corner (included) and bottom-right corner (excluded) of the N-dimensional
    hyperrectangles that cover the whole space between the two points (both excluded).

    In other words, fill the flattened vector in the range ]obstacle1, obstacle2[.

    Parameters
    ----------
    obstacle1:
        The first point to avoid in the N-dimensional hyperspace
    obstacle2:
        The second point to avoid in the N-dimensional hyperspace.
        It must be after the first point along the flattened vector.
    shape:
        The shape of the whole N-dimensional hyperspace to cover
    out_a:
        Empty buffer to write the top-left corners into
    out_b:
        Empty buffer to write the bottom-right corners into
    cur:
        row index of out_a and out_b to start writing from.
        out_a and out_b must be at least 2**ndim - 1 + cur rows in size.

    Returns
    -------
    Row index of out_a and out_b after the one that was last written to

    Example:
    --------
    Given two points at coordinates (3, 4, 5) and (3, 7, 8)
    in a space of shape (10, 10, 10), this function yields

    - [ 3,   4, 6:] -> (3, 4, 6), (4, 5, 10)
    - [ 3, 5:7,  :] -> (3, 5, 0), (4, 7, 10)
    - [ 3,   7, :8] -> (3, 7, 0), (4, 8,  8)

    >>> out = np.empty((2**3 - 1, 6), dtype=np.intp)
    >>> _hyperrectangles_between(
    ...     np.array([3, 4, 5]),
    ...     np.array([3, 7, 8]),
    ...     np.array([10, 10, 10]),
    ...     out[:, :3],
    ...     out[:, 3:],
    ...     0
    ... )
    3
    >>> out[:3]
    array([[ 3,  4,  6,  4,  5, 10],
           [ 3,  5,  0,  4,  7, 10],
           [ 3,  7,  0,  4,  8,  8]])
    """
    ndim = len(obstacle1)  # Always >0

    if ndim == 1:
        # Trivial use case in mono-dimensional space.
        # There can be at most one 1D line between the two obstacles.
        #
        # ..XaaX...

        i, j = obstacle1[0], obstacle2[0]
        if i + 1 < j:
            out_a[cur, 0] = i + 1
            out_b[cur, 0] = j
            cur += 1
        return cur

    if obstacle1[0] == obstacle2[0]:
        # The two obstacles are on the same outermost hyperspace/space/surface/row.
        # Simplify the problem by recursing into a hyperspace with one less dimension.
        #
        # .........
        # ..XaaX...
        # .........

        new_cur = _hyperrectangles_between(
            obstacle1[1:],
            obstacle2[1:],
            shape[1:],
            out_a[:, 1:],
            out_b[:, 1:],
            cur,
        )
        prefix = obstacle1[0]
        out_a[cur:new_cur, 0] = prefix
        out_b[cur:new_cur, 0] = prefix + 1
        return new_cur

    # General case with the obstacles laying on different rows/planes/spaces/etc.
    #
    # If you have a 2d plane with two obstacles on different rows, you will have
    # - up to one 1D line on the same row as obstacle1, from obstacle1 to the end of
    #   the plane
    # - up to one 2D surface on the rows between obstacle1 and obstacle2
    # - up to one 1D line on the same row as obstacle2, from the beginning of the
    #   plane to  obstacle2
    #
    # .........
    # ..Xaaaaaa
    # bbbbbbbbb
    # bbbbbbbbb
    # ccccX....
    # .........
    #
    # If you have a 3D space with two obstacles on different planes, you will have
    # - up to two 1D and 2D objects laying in the 2D plane of obstacle1, as described
    #   in the previous paragraph
    # - up to one 3D space covering the planes between obstacle1 and obstacle2,
    # - and up to two 1D and 2D objects laying in the 2d plane of obstacle2.
    #
    # .........  ccccccccc  ccccccccc  ddddddddd
    # ..Xaaaaaa  ccccccccc  ccccccccc  ddddddddd
    # bbbbbbbbb  ccccccccc  ccccccccc  ddddddddd
    # bbbbbbbbb  ccccccccc  ccccccccc  ddddddddd
    # bbbbbbbbb  ccccccccc  ccccccccc  eeeeX....
    # bbbbbbbbb  ccccccccc  ccccccccc  .........
    #
    # And so on for higher dimensionionalities.

    # Recurse with one less dimension on the same row, plane, etc. as obstacle1
    # until the end of the same row, plane, etc.
    new_cur = _hyperrectangles_between(
        obstacle1[1:],
        _one_after(shape[1:]),
        shape[1:],
        out_a[:, 1:],
        out_b[:, 1:],
        cur,
    )
    prefix = obstacle1[0]
    out_a[cur:new_cur, 0] = prefix
    out_b[cur:new_cur, 0] = prefix + 1
    cur = new_cur

    # Process the 2D surface, or 3D plane, etc. between the two obstacles
    if obstacle1[0] + 1 < obstacle2[0]:
        out_a[cur, 0] = obstacle1[0] + 1
        out_a[cur, 1:] = 0
        out_b[cur, 0] = obstacle2[0]
        out_b[cur, 1:] = shape[1:]
        cur += 1

    # Recurse with one less dimension on the same row, plane, etc. as obstacle2
    # from the beginning of the same row, plane, etc. until obstacle2
    new_cur = _hyperrectangles_between(
        _one_before(shape[1:]),
        obstacle2[1:],
        shape[1:],
        out_a[:, 1:],
        out_b[:, 1:],
        cur,
    )
    prefix = obstacle2[0]
    out_a[cur:new_cur, 0] = prefix
    out_b[cur:new_cur, 0] = prefix + 1
    return new_cur


@cython.cfunc
@cython.inline
def _one_before(shape: Py_ssize_t[:]) -> Py_ssize_t[:]:
    """Given an ND array 'a' of the given shape, return the coordinates of the point at
    a.flat[-1], without bounds check or wrap around.

    >>> tuple(_one_before((2, 5)))
    (-1, 4)

        x
    .....
    .....

    """
    ndim = len(shape)
    out: Py_ssize_t[:] = empty_view(ndim)
    out[0] = -1
    for i in range(1, ndim):
        out[i] = shape[i] - 1
    return out


@cython.cfunc
@cython.inline
def _one_after(shape: Py_ssize_t[:]) -> Py_ssize_t[:]:
    """Given an ND array 'a' of the given shape, return the coordinates of the point at
    a.flat[a.size], without bounds check.

    >>> tuple(_one_after((2, 5)))
    (2, 0)

    .....
    .....
    x

    """
    ndim = len(shape)
    out: Py_ssize_t[:] = empty_view(ndim)
    out[0] = shape[0]
    for i in range(1, ndim):
        out[i] = 0
    return out
