import hypothesis
import numpy as np
import pytest
from h5py._hl.selections import Selection
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_equal
from versioned_hdf5.slicetools import (
    build_slab_indices_and_offsets,
    read_many_slices,
    spaceid_to_slice,
)

from versioned_hdf5 import VersionedHDF5File
from versioned_hdf5.cytools import count2stop

from .test_typing import MinimalArray

max_examples = 10_000


def test_spaceid_to_slice(h5file):
    shape = 10
    a = h5file.create_dataset("a", data=np.arange(shape))

    for start in range(0, shape):
        for count in range(0, shape):
            for stride in range(1, shape):
                for block in range(0, shape):
                    if count != 1 and block != 1:
                        # Not yet supported. Doesn't seem to be supported
                        # by HDF5 either (?)
                        continue

                    spaceid = a.id.get_space()
                    spaceid.select_hyperslab((start,), (count,), (stride,), (block,))
                    sel = Selection((shape,), spaceid)
                    try:
                        a[sel]
                    except (ValueError, OSError):
                        # HDF5 doesn't allow stride/count combinations
                        # that are impossible (the count must be the exact
                        # number of elements in the selected block).
                        # Rather than trying to enumerate those here, we
                        # just check what doesn't give an error.
                        continue
                    try:
                        s = spaceid_to_slice(spaceid)
                    except:
                        print(start, count, stride, block)
                        raise
                    assert_equal(a[s.raw], a[sel], f"{(start, count, stride, block)}")


def test_build_slab_indices_and_offsets_dense(h5file):
    chunks = (2, 3)
    data = np.array(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
        ]
    )
    expect_raw_data = np.array(
        [
            # chunk (0, 0), offset 0
            [1, 2, 3],
            [6, 7, 8],
            # chunk (0, 1), offset 2
            [4, 5, 0],
            [9, 10, 0],
            # chunk (1, 0), offset 4
            [11, 12, 13],
            [0, 0, 0],
            # chunk (1, 1), offset 6
            [14, 15, 0],
            [0, 0, 0],
        ]
    )
    vf = VersionedHDF5File(h5file)
    with vf.stage_version("r0") as sv:
        sv.create_dataset("a", data=data, chunks=chunks)
    virt_dset = h5file["_version_data/versions/r0/a"]
    raw_data = h5file["_version_data/a/raw_data"]
    np.testing.assert_array_equal(virt_dset[:], data, strict=True)
    np.testing.assert_array_equal(raw_data[:], expect_raw_data, strict=True)

    dcpl = virt_dset.id.get_create_plist()
    indices, offsets = build_slab_indices_and_offsets(dcpl, data.shape, chunks)
    np.testing.assert_array_equal(indices, [[1, 1], [1, 1]])
    np.testing.assert_array_equal(offsets, [[0, 2], [4, 6]])


def test_build_slab_indices_and_offsets_sparse(h5file):
    chunks = (2, 3)
    shape = (3, 5)
    vf = VersionedHDF5File(h5file)
    with vf.stage_version("r0") as sv:
        sv.create_dataset(
            "a", shape=shape, data=None, chunks=chunks, dtype=np.int64, fillvalue=123
        )
    virt_dset = h5file["_version_data/versions/r0/a"]
    expect = np.full(shape, fill_value=123, dtype=np.int64)
    np.testing.assert_array_equal(
        virt_dset[:], np.full(shape, fill_value=123, dtype=np.int64), strict=True
    )

    dcpl = virt_dset.id.get_create_plist()
    indices, offsets = build_slab_indices_and_offsets(dcpl, shape, chunks)
    np.testing.assert_array_equal(indices, [[0, 0], [0, 0]])
    np.testing.assert_array_equal(offsets, [[0, 0], [0, 0]])

    # This test is recursive, as in order to create r1 on disk versioned_hdf5 is going
    # to call build_slab_indices_and_offsets.
    with vf.stage_version("r1") as sv:
        dset = sv["a"]
        dset[1, 3] = 456
        dset[2, 0] = 789

    virt_dset = h5file["_version_data/versions/r1/a"]
    raw_data = h5file["_version_data/a/raw_data"]

    expect[1, 3] = 456
    expect[2, 0] = 789
    np.testing.assert_array_equal(virt_dset[:], expect, strict=True)
    expect_raw = np.array(
        [
            [123, 123, 123],
            [456, 123, 123],
            [789, 123, 123],
            [123, 123, 123],
            # FIXME Spurious extra chunk. Discussion:
            # https://github.com/deshaw/versioned-hdf5/pull/385#discussion_r1817313138
            [123, 123, 123],
            [123, 123, 123],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(raw_data[:], expect_raw, strict=True)

    dcpl = virt_dset.id.get_create_plist()
    indices, offsets = build_slab_indices_and_offsets(dcpl, shape, chunks)
    np.testing.assert_array_equal(indices, [[0, 1], [1, 0]])
    np.testing.assert_array_equal(offsets, [[0, 0], [2, 0]])


@st.composite
def bound_slices_st(draw, arr_size: int, view_size: int) -> slice:
    """Hypothesis draw of a slice object to slice an array of <arr_size> points
    along an axis, returning a view of <view_size> points::

        arr = np.empty(arr_size)
        idx = bound_slices_st(arr_size, view_size).example()
        assert arr[idx].size == view_size
    """
    start = draw(st.integers(0, arr_size - view_size))

    # Don't want to figure out the exact formula to avoid off-by-one errors.
    # Just add some margin and then count down.
    for max_step in range((arr_size - start + 1) // view_size + 1, 0, -1):
        tmp_stop = count2stop(start, view_size, max_step)
        if tmp_stop <= arr_size:
            break

    step = draw(st.integers(1, max_step))
    stop = count2stop(start, view_size, step)
    assert len(range(start, min(stop, arr_size), step)) == view_size
    return slice(start, stop, step)


@st.composite
def matching_slices_st(draw, src_size: int, dst_size: int) -> tuple[slice, slice]:
    """Hypothesis draw to slice src and dst arrays along the same axis
    dst[idx_dst] = src[idx_src]

    Returns tuples of idx_src, idx_dst
    where src[idx_src].size == dst[idx_dst].size
    """
    view_size = draw(st.integers(1, min(src_size, dst_size)))
    slices_st = st.tuples(
        bound_slices_st(src_size, view_size),
        bound_slices_st(dst_size, view_size),
    )
    return draw(slices_st)


@st.composite
def many_slices_st(
    draw, max_ndim: int = 4, max_size: int = 20, max_nslices: int = 4
) -> tuple[
    # shape of src array
    tuple[int, ...],
    # shape of dst array, with same dimensionality as src
    tuple[int, ...],
    # list of indices to slice src array with
    list[tuple[slice, ...]],
    # matching list of indices to slice dst array with
    list[tuple[slice, ...]],
]:
    src_shape_st = st.lists(st.integers(1, max_size), min_size=1, max_size=max_ndim)
    src_shape = tuple(draw(src_shape_st))
    ndim = len(src_shape)
    dst_shape_st = st.lists(st.integers(1, max_size), min_size=ndim, max_size=ndim)
    dst_shape = tuple(draw(dst_shape_st))

    nslices = draw(st.integers(1, max_nslices))
    src_indices = []
    dst_indices = []
    for _ in range(nslices):
        src_idx = []
        dst_idx = []
        for src_size, dst_size in zip(src_shape, dst_shape, strict=True):
            src_slice, dst_slice = draw(matching_slices_st(src_size, dst_size))
            src_idx.append(src_slice)
            dst_idx.append(dst_slice)
        src_indices.append(tuple(src_idx))
        dst_indices.append(tuple(dst_idx))

    return src_shape, dst_shape, src_indices, dst_indices


@pytest.mark.slow
@given(args=many_slices_st())
@hypothesis.settings(
    max_examples=max_examples,
    deadline=None,
    # h5file is not reset between hypothesis examples
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
)
def test_read_many_slices(h5file, args):
    shape_from, shape_to, indices_from, indices_to = args

    src = np.arange(1, np.prod(shape_from) + 1, dtype=np.int32).reshape(shape_from)

    expect = np.zeros(shape_to, dtype=src.dtype)
    src_start = []
    dst_start = []
    count = []
    src_step = []
    dst_step = []
    for idx_from, idx_to in zip(indices_from, indices_to, strict=True):
        expect[idx_to] = src[idx_from]

        src_start.append([s.start for s in idx_from])
        src_step.append([s.step for s in idx_from])
        dst_start.append([s.start for s in idx_to])
        dst_step.append([s.step for s in idx_to])
        count.append([len(range(s.start, s.stop, s.step)) for s in idx_from])

    # Test numpy->numpy
    dst = np.zeros(shape_to, dtype=src.dtype)
    read_many_slices(src, dst, src_start, dst_start, count, src_step, dst_step)
    np.testing.assert_array_equal(dst, expect, strict=True)

    # h5file fixture is not reset between hypothesis examples
    if "a" in h5file:
        del h5file["a"]
    dset = h5file.create_dataset("a", data=src)

    # Test h5py->numpy
    for kwargs in ({}, {"fast": True}, {"fast": False}):
        dst = np.zeros(shape_to, dtype=src.dtype)
        read_many_slices(
            dset, dst, src_start, dst_start, count, src_step, dst_step, **kwargs
        )
        np.testing.assert_array_equal(dst, expect, strict=True)


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("start", [0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("use_h5", [True, False])
def test_read_many_slices_clip_src_count(h5file, use_h5, start, step):
    """Counts are clipped automatically because either
    src_start or src_stride are too large.
    This causes some slices to become size 0 along one dimension.
    """
    src = np.arange(1, 21).reshape(5, 4)
    src_view = src[start::step]
    expect = np.zeros((8, 4), dtype=src.dtype)
    expect[: len(src_view)] = src_view

    # Test that completely skipping a slice because its clipped count is zero doesn't
    # cause the next rows to be skipped
    expect[7, 3] = src[4, 3]

    if use_h5:
        src = h5file.create_dataset("a", data=src)

    dst = np.zeros_like(expect)
    read_many_slices(
        src,
        dst,
        src_start=[(start, 0), (4, 3)],
        dst_start=[(0, 0), (7, 3)],
        count=[(999, 999), (999, 999)],
        src_stride=[(step, 1), (1, 1)],
    )
    np.testing.assert_array_equal(dst, expect, strict=True)


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("start", [0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("use_h5", [True, False])
def test_read_many_slices_clip_dst_count(h5file, use_h5, start, step):
    """Counts are clipped automatically because either
    dst_start or dst_stride are too large.
    This causes some slices to become size 0 along one dimension.
    """
    src = np.arange(1, 41).reshape(10, 4)
    expect = np.zeros((5, 4), dtype=src.dtype)
    dst_view = expect[start::step]
    expect[start::step] = src[: len(dst_view)]

    # Test that completely skipping a slice because its clipped count is zero doesn't
    # cause the next slices to be skipped
    expect[4, 3] = src[7, 3]

    if use_h5:
        src = h5file.create_dataset("a", data=src)

    dst = np.zeros_like(expect)
    read_many_slices(
        src,
        dst,
        src_start=[(0, 0), (7, 3)],
        dst_start=[(start, 0), (4, 3)],
        count=[(999, 999), (999, 999)],
        src_stride=None,
        dst_stride=[(step, 1), (1, 1)],
    )
    np.testing.assert_array_equal(dst, expect, strict=True)


@pytest.mark.parametrize("all_1d", [True, False])
@pytest.mark.parametrize("use_h5", [True, False])
def test_read_many_slices_broadcast_indices(h5file, use_h5, all_1d):
    """1D indices are automatically broadcast to 2D"""
    src = np.arange(1, 41).reshape(10, 4)
    expect = np.zeros((5, 4), dtype=src.dtype)
    expect[0:3, 2:4] = src[1:4, 1:3]
    if not all_1d:
        expect[2:5, 1:3] = src[1:4, 1:3]

    if use_h5:
        src = h5file.create_dataset("a", data=src)

    dst = np.zeros_like(expect)
    read_many_slices(
        src,
        dst,
        src_start=(1, 1),
        dst_start=(0, 2) if all_1d else [(0, 2), (2, 1)],
        count=(3, 2),
    )
    np.testing.assert_array_equal(dst, expect, strict=True)


@pytest.mark.parametrize("use_h5", [True, False])
def test_read_many_slices_no_indices(h5file, use_h5):
    """Edge case where there are no slices of data to copy"""
    src = np.arange(1, 41).reshape(10, 4)
    dst = np.zeros((5, 4), dtype=src.dtype)
    if use_h5:
        src = h5file.create_dataset("a", data=src)

    read_many_slices(src, dst, [], [], [])
    assert (dst == 0).all()


@pytest.mark.parametrize("contiguous_cols", [True, False])
@pytest.mark.parametrize("use_h5", [True, False])
def test_read_many_slices_noncontiguous_idx(h5file, use_h5, contiguous_cols):
    """one or more coordinate array is not C-contiguous along the rows or both
    columns and rows.

    Non-contiguous rows are supported thanks to Cython strided views.
    Non-contiguous columns are not supported in hdf5 fast mode and are transparently
    deep-copied before use.
    """
    src = np.arange(1, 51).reshape(10, 5)

    src_start = np.array([(0, 1, 2), (9, 9, 9), (2, 3, 4)])
    if contiguous_cols:
        src_start = src_start[::2, :2]
        expect = [[src[0, 1]], [src[2, 3]]]
    else:
        src_start = src_start[::2, ::2]
        expect = [[src[0, 2]], [src[2, 4]]]

    assert not src_start.flags.c_contiguous

    if use_h5:
        src = h5file.create_dataset("a", data=src)

    dst = np.zeros((2, 1), dtype=src.dtype)
    read_many_slices(src, dst, src_start, [(0, 0), (1, 0)], (1, 1))
    np.testing.assert_equal(dst, expect)


@pytest.mark.parametrize("use_h5", [True, False])
def test_read_many_slices_src_size_zero(h5file, use_h5):
    """src array has size 0"""
    src = np.empty((3, 0))
    dst = np.zeros((3, 3), dtype=src.dtype)
    if use_h5:
        src = h5file.create_dataset("a", data=src)
    read_many_slices(src, dst, [(0, 0)], [(0, 0)], [(3, 3)])
    assert (dst == 0).all()


@pytest.mark.parametrize("use_h5", [True, False])
def test_read_many_slices_dst_size_zero(h5file, use_h5):
    """dst array has size 0"""
    src = np.arange(1, 10).reshape(3, 3)
    dst = np.empty((0, 3), dtype=src.dtype)
    if use_h5:
        src = h5file.create_dataset("a", data=src)
    read_many_slices(src, dst, [(0, 0)], [(0, 0)], [(3, 3)])


def test_read_many_slices_noncontiguous_dst(h5file):
    """dst is a numpy array that's not C-contiguous.
    It is supported, but only with fast=False.
    """
    src = np.arange(1, 17).reshape(4, 4)
    expect = np.zeros((8, 12), dtype=src.dtype)
    expect[::2, ::3] = src[()]

    dst = np.zeros_like(expect)
    read_many_slices(src, dst[::2, ::3], [(0, 0), (2, 0)], [(0, 0), (2, 0)], (2, 4))
    np.testing.assert_equal(dst, expect)

    src = h5file.create_dataset("a", data=src)
    dst = np.zeros_like(expect)
    read_many_slices(
        src, dst[::2, ::3], [(0, 0), (2, 0)], [(0, 0), (2, 0)], (2, 4), fast=False
    )
    np.testing.assert_equal(dst, expect)

    # h5py src only works in slow mode
    with pytest.raises(NotImplementedError, match="contiguous"):
        read_many_slices(
            src, dst[::2, ::3], [(0, 0), (2, 0)], [(0, 0), (2, 0)], (2, 4), fast=True
        )

    # For the sake of awareness, user must explicitly pass fast=False
    with pytest.raises(NotImplementedError, match="contiguous"):
        read_many_slices(src, dst[::2, ::3], [(0, 0), (2, 0)], [(0, 0), (2, 0)], (2, 4))


def test_read_many_slices_not_fast_read_ok(h5file):
    """src is a h5py dataset that doesn't support fast read"""
    src = np.array(["foo", "bar", "baz"]).astype(object)
    expect = np.array([b"bar"]).astype(object)
    src = h5file.create_dataset("a", data=src)
    assert not src._fast_read_ok

    dst = np.zeros_like(expect)
    read_many_slices(src, dst, [[1]], [[0]], [[1]])
    np.testing.assert_array_equal(dst, expect, strict=True)

    dst = np.zeros_like(expect)
    read_many_slices(src, dst, [[1]], [[0]], [[1]], fast=False)
    np.testing.assert_array_equal(dst, expect, strict=True)

    with pytest.raises(ValueError, match="fast transfer is not possible"):
        read_many_slices(src, dst, [[1]], [[0]], [[1]], fast=True)


def test_read_many_slices_array_protocol():
    """Test that the src array can be anything that implements ArrayProtocol"""
    src = MinimalArray(np.arange(10))
    dst = np.zeros(4, dtype=src.dtype)
    expect = np.asarray([0, 2, 3, 0])
    read_many_slices(src, dst, src_start=[(2,)], dst_start=[(1,)], count=[(2,)])
    np.testing.assert_equal(dst, expect)


def test_read_many_slices_fail():
    src = np.arange(1, 5)
    dst = np.zeros(4, dtype=src.dtype)

    # Negative indices in a list (np.asarray with unsigned dtype fails)
    with pytest.raises(OverflowError):
        read_many_slices(src, dst, [[-1]], [[0]], [[1]])
    # Negative indices in numpy array (needs explicit validation)
    with pytest.raises(OverflowError):
        read_many_slices(src, dst, np.array([[-1]]), [[0]], [[1]])

    # Stride 0
    with pytest.raises(ValueError, match="Strides must be strictly greater than zero"):
        read_many_slices(src, dst, [[0]], [[0]], [[1]], [[0]], [[1]])
    with pytest.raises(ValueError, match="Strides must be strictly greater than zero"):
        read_many_slices(src, dst, [[0]], [[0]], [[1]], [[1]], [[0]])

    # src/dst ndim=0
    with pytest.raises(ValueError, match="at least one dimension"):
        read_many_slices(np.array(0), np.array(0), [[]], [[]], [[]])

    # src.ndim != dst.ndim  # noqa: ERA001
    with pytest.raises(ValueError, match="same dtype and dimensionality"):
        read_many_slices(src.reshape(2, 2), dst, [[0]], [[0]], [[1]])

    # src.dtype != dst.dtype  # noqa: ERA001
    with pytest.raises(ValueError, match="same dtype and dimensionality"):
        read_many_slices(src.astype(float), dst, [[0]], [[0]], [[1]])

    # Fail to broadcast coordinates
    with pytest.raises(ValueError, match="cannot be broadcast"):
        read_many_slices(src, dst, [[0], [1]], [[0], [1], [2]], [[1]])

    # 0-sized coordinates
    with pytest.raises(ValueError, match="must have 1 or 2 dimensions"):
        read_many_slices(src, dst, 0, 0, 1)

    # coordinates with 3+ dims
    with pytest.raises(ValueError, match="must have 1 or 2 dimensions"):
        read_many_slices(src, dst, np.zeros((1, 1, 1)), [[0]], [[1]])

    # indices.shape[1] != src.ndim  # noqa: ERA001
    with pytest.raises(ValueError, match="as many columns as src.ndim"):
        read_many_slices(src, dst, [(0, 0)], [(0, 0)], [(1, 1)])

    # force fast transfer on numpy src
    with pytest.raises(ValueError, match="fast transfer is not possible"):
        read_many_slices(src, dst, [[0]], [[0]], [[1]], fast=True)

    # dst is not writeable
    dst.setflags(write=False)
    with pytest.raises(ValueError, match="writeable"):
        read_many_slices(src, dst, [[0]], [[0]], [[1]])
