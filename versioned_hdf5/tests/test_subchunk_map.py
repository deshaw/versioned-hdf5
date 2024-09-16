from __future__ import annotations

from typing import Any

import hypothesis
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as stnp
from numpy.testing import assert_array_equal

from ..subchunk_map import (
    DROP_AXIS,
    ChunkMapIndexError,
    as_subchunk_map,
    cartesian_product,
    index_chunk_mappers,
    zip_chunk_submap,
    zip_slab_submap,
)

max_examples = 10_000


def non_negative_step_slices(size: int):
    start = st.one_of(st.integers(-size - 1, size + 1), st.none())
    stop = st.one_of(st.integers(-size - 1, size + 1), st.none())
    # only non-negative steps (or None) are allowed
    step = st.one_of(st.integers(1, size + 1), st.none())
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
def idx_chunks_shape_st(
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

    return idx, chunks, shape


@pytest.mark.slow
@given(idx_chunks_shape_st())
@hypothesis.settings(max_examples=max_examples, deadline=None)
def test_as_subchunk_map(args):
    idx, chunks, shape = args

    source = np.arange(1, np.prod(shape) + 1, dtype=np.int32).reshape(shape)
    expect = source[idx]
    actual = np.zeros_like(expect)

    for chunk_idx, arr_subidx, chunk_subidx in as_subchunk_map(chunks, idx, shape):
        chunk_idx = chunk_idx.raw

        # Test that chunk_idx selects whole chunks
        assert isinstance(chunk_idx, tuple)
        assert len(chunk_idx) == len(chunks)
        for i, c, d in zip(chunk_idx, chunks, shape):
            assert isinstance(i, slice)
            assert i.start % c == 0
            assert i.stop == min(i.start + c, d)
            assert i.step == 1

        assert not actual[arr_subidx].any(), "overlapping arr_subidx"
        actual[arr_subidx] = source[chunk_idx][chunk_subidx]

    assert_array_equal(actual, expect)


@pytest.mark.slow
@given(idx_chunks_shape_st())
@hypothesis.settings(max_examples=max_examples, deadline=None)
def test_zip_chunk_submap(args):
    idx, chunks, shape = args

    source = np.arange(1, np.prod(shape) + 1, dtype=np.int32).reshape(shape)
    expect = source[idx]
    actual = np.zeros_like(expect)

    ndindex_idx, mappers = index_chunk_mappers(idx, chunks, shape)
    assert ndindex_idx.newshape(shape) == expect.shape

    for chunk_idx in cartesian_product([m.chunk_indices for m in mappers]):
        chunk_slicer = tuple(
            slice(c * s, (c + 1) * s) for c, s in zip(chunk_idx, chunks)
        )
        out, sub = zip_chunk_submap(mappers, chunk_idx)
        assert not actual[out].any(), "overlapping out indices"
        actual[out] = source[chunk_slicer][sub]

    assert_array_equal(actual, expect)


@st.composite
def chunk_slabs_st(
    draw, chunks: tuple[int, ...], shape: tuple[int, ...]
) -> tuple[memoryview, memoryview]:
    """Given an hyperspace of the given shape that's been cut into the given chunks,
    generate edges of the hyperrectangles [a, b[, where a and b are ND chunk indices,
    that completely cover the hyperspace.
    """
    starts = []
    stops = []
    for c, s in zip(chunks, shape):
        nchunks = s // c + (s % c > 0)
        if nchunks > 1:
            cuts = sorted(draw(st.lists(st.integers(1, nchunks - 1), unique=True)))
        else:
            cuts = []
        starts.append(np.array([0] + cuts, dtype=np.intp))
        stops.append(np.array(cuts + [nchunks], dtype=np.intp))

    return cartesian_product(starts), cartesian_product(stops)


@pytest.mark.slow
@given(idx_chunks_shape_st(), st.data())
@hypothesis.settings(max_examples=max_examples, deadline=None)
def test_zip_slab_submap(args, data):
    idx, chunks, shape = args
    starts, stops = data.draw(chunk_slabs_st(chunks, shape))

    source = np.arange(1, np.prod(shape) + 1, dtype=np.int32).reshape(shape)
    expect = source[idx]
    actual = np.zeros_like(expect)

    _, mappers = index_chunk_mappers(idx, chunks, shape)
    if mappers:
        for fromc_idx, toc_idx in zip(starts, stops):
            try:
                out, sub = zip_slab_submap(mappers, fromc_idx, toc_idx)
            except ChunkMapIndexError:
                continue
            assert not actual[out].any(), "overlapping out indices"
            actual[out] = source[sub]

    assert_array_equal(actual, expect)


@pytest.mark.slow
@given(idx_chunks_shape_st(max_ndim=1))
@hypothesis.settings(max_examples=max_examples, deadline=None)
def test_chunks_indexer(args):
    """Test IndexChunkMapper.chunks_indexer and IndexChunkMapper.whole_chunks_indexer"""
    idx, chunks, shape = args
    _, mappers = index_chunk_mappers(idx, chunks, shape)
    if not mappers:
        return  # Early exit for empty index
    assert len(shape) == len(chunks) == len(mappers) == 1

    dset_size = shape[0]
    chunk_size = chunks[0]
    mapper = mappers[0]

    source = np.arange(1, dset_size + 1, dtype=np.int32)
    expect = source[idx]
    actual = np.zeros_like(expect)

    all_chunks = np.arange(mapper.n_chunks, dtype=np.intp)
    sel_chunks = all_chunks[mapper.chunks_indexer()]
    whole_chunks = all_chunks[mapper.whole_chunks_indexer()]

    for i in all_chunks:
        if i not in sel_chunks:
            assert i not in whole_chunks
            with pytest.raises(ChunkMapIndexError):
                mapper.chunk_submap(i, i + 1, shift=True)
            continue

        # Must not raise ChunkMapIndexError
        out, sub = mapper.chunk_submap(i, i + 1, shift=True)
        chunk = source[i * chunk_size : (i + 1) * chunk_size]

        actual[() if out is DROP_AXIS else out] = chunk[sub]

        coverage = np.zeros_like(chunk)
        coverage[sub] = 1
        assert coverage.any(), "chunk selected by chunk_indexer() is not covered"
        if i in whole_chunks:
            assert coverage.all(), "whole chunk is partially covered"
        else:
            assert not coverage.all(), "partial chunk is wholly covered"

    assert_array_equal(actual, expect)


@given(st.data())
@hypothesis.settings(max_examples=500)
def test_chunk_indices_in_range(data):
    """Test IndexChunkMapper.chunk_indices_in_range"""

    # Generate a single IntegerArrayMapper with 1 point per chunk.
    # The function doesn't use anything other than the chunk_indices so testing on the
    # full hypothesis strategy of indices would be overkill.
    size = data.draw(st.integers(1, 4))
    idx, start, stop = data.draw(
        st.tuples(
            st.lists(elements=st.integers(0, size - 1), min_size=1, unique=True),
            st.integers(0, size + 1),
            st.integers(0, size + 1),
        )
    )
    _, (mapper,) = index_chunk_mappers(idx, (1,), (size,))

    try:
        actual = mapper.chunk_indices_in_range(start, stop)
    except ChunkMapIndexError:
        actual = np.array([], dtype=np.intp)

    expect = np.asarray(mapper.chunk_indices)
    # Note: this is O(n); chunk_indices_in_range() achieves the same in O(log(n))
    expect = expect[(expect >= start) & (expect < stop)]
    assert_array_equal(actual, expect, strict=True)


def test_cartesian_product():
    a = np.array([1, 2, 3], dtype=np.intp)
    b = np.array([4, 5], dtype=np.intp)
    c = np.array([6, 7], dtype=np.intp)

    expect = np.array(
        [
            [1, 4, 6],
            [1, 4, 7],
            [1, 5, 6],
            [1, 5, 7],
            [2, 4, 6],
            [2, 4, 7],
            [2, 5, 6],
            [2, 5, 7],
            [3, 4, 6],
            [3, 4, 7],
            [3, 5, 6],
            [3, 5, 7],
        ],
        dtype=np.intp,
    )

    actual = np.asarray(cartesian_product([a, b, c]))
    assert_array_equal(actual, expect, strict=True)


def test_cartesian_product_one_dim():
    a = np.array([1, 2, 3])

    expect = np.array([[1], [2], [3]], dtype=np.intp)
    actual = np.asarray(cartesian_product([a]))
    assert_array_equal(actual, expect, strict=True)


@pytest.mark.parametrize(
    "args",
    [
        ([1, 2, 3], [], [5, 6]),
        ([1, 2, 3], []),
        ([], [1, 2, 3]),
        ([], [], []),
        ([], []),
        ([],),
        (),
    ],
)
def test_cartesian_product_size_zero(args):
    expect = np.empty((0, len(args)), dtype=np.intp)
    args = [np.array(a, dtype=np.intp) for a in args]
    actual = np.asarray(cartesian_product(args))
    assert_array_equal(actual, expect, strict=True)
