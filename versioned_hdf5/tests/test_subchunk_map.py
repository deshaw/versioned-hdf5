from __future__ import annotations

from typing import Any

import hypothesis
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as stnp
from numpy.testing import assert_array_equal

from ..subchunk_map import DROP_AXIS, SliceMapper, as_subchunk_map, index_chunk_mappers

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
            st.one_of(
                non_negative_step_slices(size),
                st.integers(-size, size - 1),
            )
            # Note: ... is not supported
            for size in shape[:nidx]
        )
    )
    return draw(idx_st)


@st.composite
def fancy_idx_st(draw, shape: tuple[int, ...]) -> Any:
    """A single axis is indexed by a NDArray[np.intp], whose elements can be negative,
    non-unique, and not in order.
    All other axes are indexed by slices.
    """
    fancy_idx_axis = draw(st.integers(0, len(shape) - 1))
    size = shape[fancy_idx_axis]
    fancy_idx = stnp.arrays(
        np.intp,
        shape=st.integers(0, size * 2),
        elements=st.integers(-size, size - 1),
        unique=False,
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
def mask_idx_st(draw, shape: tuple[int, ...]) -> Any:
    """A single axis is indexed by a NDArray[np.bool], whereas all other axes
    may be indexed by slices.
    """
    ndim = len(shape)
    mask_idx_axis = draw(st.integers(0, ndim - 1))
    mask_idx = stnp.arrays(np.bool_, shape[mask_idx_axis], elements=st.booleans())
    idx_st = st.tuples(
        *[non_negative_step_slices(shape[dim]) for dim in range(mask_idx_axis)],
        mask_idx,
        *[
            non_negative_step_slices(shape[dim])
            for dim in range(mask_idx_axis + 1, ndim)
        ],
    )
    return draw(idx_st)


@st.composite
def idx_shape_chunks_st(
    draw, max_ndim: int = 4
) -> tuple[Any, tuple[int, ...], tuple[int, ...]]:
    shape_st = st.lists(st.integers(1, 20), min_size=1, max_size=max_ndim)
    shape = tuple(draw(shape_st))

    chunks_st = st.tuples(*[st.integers(1, s + 1) for s in shape])
    chunks = draw(chunks_st)

    idx_st = st.one_of(
        basic_idx_st(shape),
        fancy_idx_st(shape),
        mask_idx_st(shape),
    )
    idx = draw(idx_st)

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


@pytest.mark.slow
@given(idx_shape_chunks_st(max_ndim=1))
@hypothesis.settings(max_examples=max_examples, deadline=None)
def test_chunks_indexer(args):
    """Test IndexChunkMapper.chunks_indexer and IndexChunkMapper.whole_chunks_idxidx"""
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
    whole_chunks = sel_chunks[mapper.whole_chunks_idxidx()]

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


def test_simplify_indices():
    """Test that a fancy index that can be redefined globally as a slice results in a
    SliceMapper
    """
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
    assert mapper.whole_chunks_idxidx() == slice(1, 2, 1)

    _, (mapper,) = index_chunk_mappers(
        [True, False, True, True, False, False, True, True, True, True], (10,), (2,)
    )
    assert_array_equal(mapper.chunks_indexer(), [0, 1, 3, 4])
    assert_array_equal(mapper.whole_chunks_idxidx(), slice(1, 4, 1))

    _, (mapper,) = index_chunk_mappers(
        [True, False, False, True, True, False],
        (6,),
        (2,),
    )
    assert mapper.chunks_indexer() == slice(0, 3, 1)
    assert mapper.whole_chunks_idxidx() == slice(0, 0, 1)
