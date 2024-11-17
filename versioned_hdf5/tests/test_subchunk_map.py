from __future__ import annotations

from typing import Any

import hypothesis
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_array_equal

from ..subchunk_map import SliceMapper, as_subchunk_map, index_chunk_mappers

max_examples = 10_000


def non_negative_step_slices_st(size: int):
    start = st.one_of(st.none(), st.integers(-size - 1, size + 1))
    stop = st.one_of(st.none(), st.integers(-size - 1, size + 1))
    # only non-negative steps (or None) are allowed
    step = st.one_of(st.none(), st.integers(1, size + 1))
    return st.builds(slice, start, stop, step)


def array_indices_st(size: int):
    return st.one_of(
        st.lists(st.integers(-size, max(0, size - 1)), max_size=size * 2),
        st.lists(st.booleans(), min_size=size, max_size=size),
    )


def scalar_indices_st(size: int):
    return st.integers(-size, size - 1) if size > 0 else st.nothing()


@st.composite
def basic_idx_st(draw, shape: tuple[int, ...]):
    """Hypothesis draw of slice and integer indexes"""
    nidx = draw(st.integers(0, len(shape)))
    idx_st = st.tuples(
        *(
            st.one_of(non_negative_step_slices_st(size), scalar_indices_st(size))
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

    Interleaving scalars and slices and array indices is not supported:
    https://github.com/Quansight-Labs/ndindex/issues/188
    """
    fancy_idx_axis = draw(st.integers(0, len(shape) - 1))
    nidx = draw(st.integers(fancy_idx_axis + 1, len(shape)))
    idx_st = st.tuples(
        *[non_negative_step_slices_st(shape[dim]) for dim in range(fancy_idx_axis)],
        array_indices_st(shape[fancy_idx_axis]),
        *[
            non_negative_step_slices_st(shape[dim])
            for dim in range(fancy_idx_axis + 1, nidx)
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


def test_invalid_indices():
    with pytest.raises(IndexError, match="too many indices"):
        index_chunk_mappers((0, 0), (4,), (2,))
    with pytest.raises(IndexError, match="out of bounds"):
        index_chunk_mappers((4,), (4,), (2,))
    with pytest.raises(IndexError, match="out of bounds"):
        index_chunk_mappers((-5,), (4,), (2,))
    with pytest.raises(IndexError, match="out of bounds"):
        index_chunk_mappers(([4],), (4,), (2,))
    with pytest.raises(IndexError, match="out of bounds"):
        index_chunk_mappers(([-5],), (4,), (2,))
    with pytest.raises(IndexError, match="boolean index did not match indexed array"):
        index_chunk_mappers(([True] * 3,), (4,), (2,))
    with pytest.raises(IndexError, match="boolean index did not match indexed array"):
        index_chunk_mappers(([True] * 5,), (4,), (2,))

    with pytest.raises(IndexError, match="valid indices"):
        index_chunk_mappers("foo", (4,), (2,))
    with pytest.raises(ValueError, match="step"):
        index_chunk_mappers(slice(None, None, 0), (4,), (2,))

    with pytest.raises(NotImplementedError, match="step"):
        index_chunk_mappers(slice(None, None, -1), (4,), (2,))
    with pytest.raises(NotImplementedError, match="None"):
        index_chunk_mappers(None, (4,), (2,))
    with pytest.raises(NotImplementedError, match="newaxis"):
        index_chunk_mappers(np.newaxis, (4,), (2,))
    with pytest.raises(NotImplementedError, match="Ellipsis"):
        index_chunk_mappers((..., 0), (4, 4), (2, 2))

    # Fancy indices are tentatively simplified to slices. However, it would be a
    # mistake to simplify [[0, 1], [0, 1]] to [:2, :2]!
    with pytest.raises(NotImplementedError, match="Multiple fancy indices"):
        index_chunk_mappers(([0, 1], [0, 1]), (4, 4), (2, 2))
    with pytest.raises(NotImplementedError, match="Multiple fancy indices"):
        index_chunk_mappers(([True, True], [True, True]), (2, 2), (2, 2))
    with pytest.raises(NotImplementedError, match="Multiple fancy indices"):
        index_chunk_mappers(([True, True], [0, 1]), (2, 2), (2, 2))
    with pytest.raises(NotImplementedError, match="1-dimensional"):
        index_chunk_mappers([[0, 1], [1, 0]], (4,), (2,))

    with pytest.raises(ValueError, match="chunk sizes"):
        index_chunk_mappers((), (4,), (0,))
    with pytest.raises(ValueError, match="shape"):
        index_chunk_mappers((), (-1,), (2,))
    with pytest.raises(ValueError, match="chunks"):
        index_chunk_mappers((), (4,), (2, 2))
