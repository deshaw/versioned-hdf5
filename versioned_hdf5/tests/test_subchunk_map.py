from __future__ import annotations

import itertools
from typing import Any

import hypothesis
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_array_equal

from ..cytools import np_hsize_t
from ..slicetools import read_many_slices
from ..subchunk_map import (
    DROP_AXIS,
    EverythingMapper,
    SliceMapper,
    TransferType,
    as_subchunk_map,
    index_chunk_mappers,
    read_many_slices_params_nd,
)

max_examples = 10_000


def non_negative_step_slices(size: int):
    start = st.one_of(st.none(), st.integers(-size - 1, size + 1))
    stop = st.one_of(st.none(), st.integers(-size - 1, size + 1))
    # only non-negative steps (or None) are allowed
    step = st.one_of(st.none(), st.integers(1, size + 1))
    return st.builds(slice, start, stop, step)


@st.composite
def basic_idx_st(draw, shape: tuple[int, ...]) -> Any:
    """Hypothesis draw of slice and integer indexes"""
    nidx = draw(st.integers(0, len(shape)))
    idx_st = st.tuples(
        *(
            # FIXME we should push the scalar use case into non_negative_step_slices
            # However ndindex fails when mixing scalars and slices and array indices
            # https://github.com/Quansight-Labs/ndindex/issues/188
            # Note: ..., None, and np.newaxis are not supported
            st.one_of(non_negative_step_slices(size), st.integers(-size, size - 1))
            if size > 0
            else non_negative_step_slices(size)
            for size in shape[:nidx]
        )
    )
    return draw(idx_st)


@st.composite
def fancy_idx_st(draw, shape: tuple[int, ...]) -> Any:
    """A single axis is indexed by either

    - a list[int] whose elements can be negative, non-unique, and not in order, or
    - a list[bool]

    All other axes are indexed by slices.
    """
    fancy_idx_axis = draw(st.integers(0, len(shape) - 1))
    size = shape[fancy_idx_axis]
    fancy_idx = st.one_of(
        st.lists(st.integers(-size, max(0, size - 1)), max_size=size * 2),
        st.lists(st.booleans(), min_size=size, max_size=size),
    )
    idx_st = st.tuples(
        *[non_negative_step_slices(shape[dim]) for dim in range(fancy_idx_axis)],
        fancy_idx,
        *[
            non_negative_step_slices(shape[dim])
            for dim in range(fancy_idx_axis + 1, len(shape))
        ],
    )
    return draw(idx_st)


@st.composite
def shape_chunks_st(
    draw, max_ndim: int = 4, min_size: int = 1, max_size: int = 20
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape_st = st.lists(st.integers(min_size, max_size), min_size=1, max_size=max_ndim)
    shape = tuple(draw(shape_st))

    chunks_st = st.tuples(*[st.integers(1, s + 1) for s in shape])
    chunks = draw(chunks_st)
    return shape, chunks


def idx_st(shape: tuple[int, ...]) -> Any:
    return st.one_of(basic_idx_st(shape), fancy_idx_st(shape))


@st.composite
def idx_shape_chunks_st(
    draw, max_ndim: int = 4
) -> tuple[Any, tuple[int, ...], tuple[int, ...]]:
    shape, chunks = draw(shape_chunks_st(max_ndim))
    idx = draw(idx_st(shape))
    return idx, shape, chunks


@pytest.mark.slow
@given(idx_shape_chunks_st())
@hypothesis.settings(max_examples=max_examples, deadline=None)
def test_as_subchunk_map(args):
    idx, shape, chunks = args

    source = np.arange(1, np.prod(shape) + 1, dtype=np.int32).reshape(shape)
    expect = source[idx]
    actual = np.zeros_like(expect)

    for chunk_idx, value_sub_idx, chunk_sub_idx in as_subchunk_map(idx, shape, chunks):
        chunk_idx = chunk_idx.raw

        # Test that chunk_idx selects whole chunks
        assert isinstance(chunk_idx, tuple)
        assert len(chunk_idx) == len(chunks)
        for i, c, d in zip(chunk_idx, chunks, shape):
            assert isinstance(i, slice)
            assert i.start % c == 0
            assert i.stop == min(i.start + c, d)
            assert i.step == 1

        assert not actual[value_sub_idx].any(), "overlapping value_sub_idx"
        actual[value_sub_idx] = source[chunk_idx][chunk_sub_idx]

    assert_array_equal(actual, expect)


@pytest.mark.slow
@given(idx_shape_chunks_st(max_ndim=1))
@hypothesis.settings(max_examples=max_examples, deadline=None)
def test_chunks_indexer(args):
    """Test IndexChunkMapper.chunks_indexer and IndexChunkMapper.whole_chunks_indexer"""
    idx, shape, chunks = args
    _, mappers = index_chunk_mappers(idx, shape, chunks)
    if not mappers:
        return  # Early exit for empty index
    assert len(shape) == len(chunks) == len(mappers) == 1
    dset_size = shape[0]
    mapper = mappers[0]
    assert mapper.dset_size == shape[0]
    assert mapper.chunk_size == chunks[0]

    source = np.arange(1, dset_size + 1)
    expect = source[idx]
    actual = np.zeros_like(expect)

    all_chunks = np.arange(mapper.n_chunks)
    sel_chunks = all_chunks[mapper.chunks_indexer()]
    whole_chunks = all_chunks[mapper.whole_chunks_indexer()]

    # Test that the slices of chunks are strictly monotonic ascending
    assert_array_equal(sel_chunks, np.unique(sel_chunks))
    assert_array_equal(whole_chunks, np.unique(whole_chunks))

    # Test that whole_chunks is a subset of sel_chunks
    assert np.setdiff1d(whole_chunks, sel_chunks, assume_unique=True).size == 0

    for i in sel_chunks:
        source_idx, value_sub_idx, chunk_sub_idx = mapper.chunk_submap(i)
        chunk = source[source_idx.raw]

        if value_sub_idx is DROP_AXIS:
            value_sub_idx = ()
        actual[value_sub_idx] = chunk[chunk_sub_idx]

        coverage = np.zeros_like(chunk)
        coverage[chunk_sub_idx] = 1
        assert coverage.any(), "chunk selected by chunk_indexer() is not covered"
        if i in whole_chunks:
            assert coverage.all(), "whole chunk is partially covered"
        else:
            assert not coverage.all(), "partial chunk is wholly covered"

    assert_array_equal(actual, expect)


@pytest.mark.slow
@given(idx_shape_chunks_st(max_ndim=1))
@hypothesis.settings(max_examples=max_examples, deadline=None)
def test_read_many_slices_param(args):
    idx, shape, chunks = args
    _, mappers = index_chunk_mappers(idx, shape, chunks)
    if not mappers:
        return  # Early exit for empty index
    assert len(shape) == len(chunks) == len(mappers) == 1
    dset_size = shape[0]
    chunk_size = chunks[0]
    mapper = mappers[0]
    assert mapper.dset_size == shape[0]
    assert mapper.chunk_size == chunks[0]

    source = np.arange(1, dset_size + 1, dtype=np.int32)
    expect = source[idx]
    actual = np.zeros_like(expect)

    slab_offsets = np.arange(0, dset_size, chunk_size, dtype=np_hsize_t)
    assert len(slab_offsets) == mapper.n_chunks

    # Randomize the order of the chunks on the slab.
    # Note that the last chunk may be smaller than chunk_size,
    # so we may end up with an area of the slab full off zeros.
    slab_offsets = np.random.permutation(slab_offsets)
    slab = np.zeros(mapper.n_chunks * chunk_size, dtype=source.dtype)
    for i, offset in enumerate(slab_offsets.tolist()):
        chunk = source[i * chunk_size : (i + 1) * chunk_size]
        slab[offset : offset + len(chunk)] = chunk

    n_sel_chunks = len(mapper.chunk_indices)
    slab_offsets = slab_offsets[mapper.chunks_indexer()]
    assert len(slab_offsets) == n_sel_chunks

    slices, chunks_to_slices = mapper.read_many_slices_params()
    if chunks_to_slices is None:
        assert slices.shape == (n_sel_chunks, 5)
    else:
        # Test that we are not generating a 1:N mapping in the special case of a fancy
        # index that contains exactly one point per chunk, and therefore
        # read_many_slices() needs to transfer exactly one slice per chunk;
        # e.g. chunk_size=10, idx=[3, 15, 21].
        assert slices.shape[0] > n_sel_chunks
        assert slices.shape[1] == 5

        assert chunks_to_slices.shape == (n_sel_chunks + 1,)
        assert chunks_to_slices[0] == 0
        assert chunks_to_slices[-1] == len(slices)

        # Replicate slab_offsets for each slice of each chunk
        slab_offsets_idx = []
        for i in range(n_sel_chunks):
            start = chunks_to_slices[i]
            stop = chunks_to_slices[i + 1]
            assert 0 <= start < stop <= len(slices)
            count = stop - start
            slab_offsets_idx += [i] * count
        slab_offsets = slab_offsets[slab_offsets_idx]
    assert len(slab_offsets) == len(slices)

    actual_view = actual[None] if actual.shape == () else actual

    # print("=" * 80)
    # print(f"{idx=} {chunks=} {shape=} {slab_offsets=}")
    # print(slices)

    read_many_slices(
        src=slab,
        dst=actual_view,
        src_start=slices[:, 0:1] + slab_offsets[:, None],
        dst_start=slices[:, 1:2],
        count=slices[:, 2:3],
        src_stride=slices[:, 3:4],
        dst_stride=slices[:, 4:5],
    )

    assert_array_equal(actual, expect)


@pytest.mark.slow
@given(idx_shape_chunks_st())
@hypothesis.settings(max_examples=max_examples, deadline=None)
def test_read_many_slices_param_nd(args):
    idx, shape, chunks = args
    _, mappers = index_chunk_mappers(idx, shape, chunks)
    if not mappers:
        return  # Early exit for empty index

    source = np.arange(1, np.prod(shape) + 1, dtype=np.int32).reshape(shape)
    expect = source[idx]

    chunk_idxidx = np.array(
        list(itertools.product(*[list(range(len(m.chunk_indices))) for m in mappers])),
        dtype=np_hsize_t,
    )

    # Generate the slab and the slab_offsets.
    # We don't care to populate the slab with the full contents of the source dataset,
    # we just need the chunks impacted by the index.
    n_chunks = np.prod([m.n_chunks for m in mappers])
    slab = np.zeros((n_chunks * chunks[0], *chunks[1:]), dtype=source.dtype)
    slab_offsets = np.random.choice(  # randomize slab order
        np.arange(0, n_chunks * chunks[0], chunks[0], dtype=np_hsize_t),
        size=len(chunk_idxidx),
        replace=False,
    )
    for chunk_idxidx_i, slab_offset_i in zip(chunk_idxidx, slab_offsets.tolist()):
        source_idx = []
        for j, chunk_idxidx_ij in enumerate(chunk_idxidx_i):
            chunk_idx_ij = int(mappers[j].chunk_indices[chunk_idxidx_ij])
            source_idx.append(
                slice(start := chunk_idx_ij * chunks[j], start + chunks[j])
            )
        chunk = source[tuple(source_idx)]
        # chunk may be smaller than the chunk size along the edges
        slab[slab_offset_i:][tuple(slice(c) for c in chunk.shape)] = chunk

    # Can't pass "hsize_t[:] | None" to cythonized functions
    DUMMY_SLAB_OFFSETS = np.empty(0, dtype=np_hsize_t)

    getitem_dst = np.zeros_like(expect)
    getitem_dst_view = getitem_dst[tuple(m.value_view_idx for m in mappers)]

    # __getitem__
    getitem_slices_nd = read_many_slices_params_nd(
        TransferType.getitem,
        mappers,
        chunk_idxidx,
        src_slab_offsets=slab_offsets,
        dst_slab_offsets=DUMMY_SLAB_OFFSETS,
    )
    read_many_slices(
        src=slab,
        dst=getitem_dst_view,
        src_start=getitem_slices_nd[:, 0, :],
        dst_start=getitem_slices_nd[:, 1, :],
        count=getitem_slices_nd[:, 2, :],
        src_stride=getitem_slices_nd[:, 3, :],
        dst_stride=getitem_slices_nd[:, 4, :],
    )

    # print("=" * 80)
    # print(f"{idx=} {chunks=} {shape=} {slab_offsets=}")
    # print("slab=\n", slab)
    # print(f"slices_nd=\n{np.asarray(getitem_slices_nd)}")
    # print("expect=\n", expect)
    # print("dst=\n", getitem_dst)

    assert_array_equal(getitem_dst, expect)

    # Now do it again in the other way for __setitem__
    setitem_dst_slab = np.zeros_like(slab)
    setitem_slices_nd = read_many_slices_params_nd(
        TransferType.setitem,
        mappers,
        chunk_idxidx,
        src_slab_offsets=DUMMY_SLAB_OFFSETS,
        dst_slab_offsets=slab_offsets,
    )
    read_many_slices(
        src=getitem_dst_view,
        dst=setitem_dst_slab,
        src_start=setitem_slices_nd[:, 0, :],
        dst_start=setitem_slices_nd[:, 1, :],
        count=setitem_slices_nd[:, 2, :],
        src_stride=setitem_slices_nd[:, 3, :],
        dst_stride=setitem_slices_nd[:, 4, :],
    )
    getitem_dst2 = np.zeros_like(getitem_dst)
    getitem_dst_view2 = getitem_dst2[tuple(m.value_view_idx for m in mappers)]
    read_many_slices(
        src=setitem_dst_slab,
        dst=getitem_dst_view2,
        src_start=getitem_slices_nd[:, 0, :],
        dst_start=getitem_slices_nd[:, 1, :],
        count=getitem_slices_nd[:, 2, :],
        src_stride=getitem_slices_nd[:, 3, :],
        dst_stride=getitem_slices_nd[:, 4, :],
    )
    assert_array_equal(getitem_dst2, expect)

    # And finally slab-to-slab
    slab2slab_dst = np.zeros_like(slab)
    dst_slab_offsets = np.random.permutation(slab_offsets)
    slab2slab_slices_nd = read_many_slices_params_nd(
        TransferType.slab_to_slab,
        mappers,
        chunk_idxidx,
        src_slab_offsets=slab_offsets,
        dst_slab_offsets=dst_slab_offsets,
    )
    read_many_slices(
        src=slab,
        dst=slab2slab_dst,
        src_start=slab2slab_slices_nd[:, 0, :],
        dst_start=slab2slab_slices_nd[:, 1, :],
        count=slab2slab_slices_nd[:, 2, :],
        src_stride=slab2slab_slices_nd[:, 3, :],
        dst_stride=slab2slab_slices_nd[:, 4, :],
    )
    # In order to test the new slab, read from it with __getitem__
    getitem_slices_nd3 = read_many_slices_params_nd(
        TransferType.getitem,
        mappers,
        chunk_idxidx,
        src_slab_offsets=dst_slab_offsets,
        dst_slab_offsets=DUMMY_SLAB_OFFSETS,
    )
    getitem_dst3 = np.zeros_like(getitem_dst)
    getitem_dst_view3 = getitem_dst3[tuple(m.value_view_idx for m in mappers)]
    read_many_slices(
        src=slab2slab_dst,
        dst=getitem_dst_view3,
        src_start=getitem_slices_nd3[:, 0, :],
        dst_start=getitem_slices_nd3[:, 1, :],
        count=getitem_slices_nd3[:, 2, :],
        src_stride=getitem_slices_nd3[:, 3, :],
        dst_stride=getitem_slices_nd3[:, 4, :],
    )
    assert_array_equal(getitem_dst3, expect)


def test_mapper_attributes():
    _, (mapper,) = index_chunk_mappers(slice(5), (6,), (3,))
    assert mapper.dset_size == 6
    assert mapper.chunk_size == 3
    assert mapper.n_chunks == 2
    assert mapper.last_chunk_size == 3

    _, (mapper,) = index_chunk_mappers((), (5,), (3,))
    assert mapper.dset_size == 5
    assert mapper.chunk_size == 3
    assert mapper.n_chunks == 2
    assert mapper.last_chunk_size == 2


def test_simplify_indices():
    """Test that

    - a slice or a fancy index that selects everything results in an EverythingMapper
    - a fancy index that can be redefined globally as a slice results in a SliceMapper
    """
    _, (mapper,) = index_chunk_mappers((), (4,), (2,))
    assert isinstance(mapper, EverythingMapper)

    _, (mapper,) = index_chunk_mappers(slice(None), (4,), (2,))
    assert isinstance(mapper, EverythingMapper)

    _, (mapper,) = index_chunk_mappers(slice(999), (4,), (2,))
    assert isinstance(mapper, EverythingMapper)

    _, (mapper,) = index_chunk_mappers([True, True, True, True], (4,), (2,))
    assert isinstance(mapper, EverythingMapper)

    _, (mapper,) = index_chunk_mappers([0, 1, 2, 3], (4,), (2,))
    assert isinstance(mapper, EverythingMapper)

    _, (mapper,) = index_chunk_mappers([True, True, False, False], (4,), (2,))
    assert isinstance(mapper, SliceMapper)
    assert mapper.start == 0
    assert mapper.stop == 2
    assert mapper.step == 1

    _, (mapper,) = index_chunk_mappers([False, True, False, True], (4,), (2,))
    assert isinstance(mapper, SliceMapper)
    assert mapper.start == 1
    assert mapper.stop == 4
    assert mapper.step == 2


def test_chunk_submap_simplifies_indices():
    """Test that, when a fancy index can't be globally simplified to a slice,
    as_subchunk_map still attemps to simplify the individual chunk subindices.
    """
    _, (mapper,) = index_chunk_mappers(
        [True, False, True, False]  # chunk 0
        + [True, True, False, False]  # chunk 1
        + [True, False, True, True],  # chunk 2
        (12,),
        (4,),
    )
    _, value_sub_idx, chunk_sub_idx = mapper.chunk_submap(0)
    assert value_sub_idx == slice(0, 2, 1)
    assert chunk_sub_idx == slice(0, 3, 2)
    _, value_sub_idx, chunk_sub_idx = mapper.chunk_submap(1)
    assert value_sub_idx == slice(2, 4, 1)
    assert chunk_sub_idx == slice(0, 2, 1)
    _, value_sub_idx, chunk_sub_idx = mapper.chunk_submap(2)
    assert value_sub_idx == slice(4, 7, 1)
    assert_array_equal(chunk_sub_idx, [0, 2, 3])  # Can't be simplified


def test_chunks_indexer_simplifies_indices():
    """Test that chunks_indexer() and whole_chunks_indexer() return a slice if possible"""
    _, (mapper,) = index_chunk_mappers(slice(None, None, 3), (10,), (2,))
    assert_array_equal(mapper.chunks_indexer(), [0, 1, 3, 4])

    _, (mapper,) = index_chunk_mappers(slice(None, None, 4), (10,), (2,))
    assert mapper.chunks_indexer() == slice(0, 5, 2)

    _, (mapper,) = index_chunk_mappers(
        [True, False, True, True, False, False], (6,), (2,)
    )
    assert mapper.chunks_indexer() == slice(0, 2, 1)
    assert mapper.whole_chunks_indexer() == slice(1, 2, 1)

    _, (mapper,) = index_chunk_mappers(
        [True, False, True, True, False, False, True, True, True, True], (10,), (2,)
    )
    assert_array_equal(mapper.chunks_indexer(), [0, 1, 3, 4])
    assert_array_equal(mapper.whole_chunks_indexer(), [1, 3, 4])

    _, (mapper,) = index_chunk_mappers(
        [True, False, False, True, True, False],
        (6,),
        (2,),
    )
    assert mapper.chunks_indexer() == slice(0, 3, 1)
    assert mapper.whole_chunks_indexer() == slice(0, 0, 1)
