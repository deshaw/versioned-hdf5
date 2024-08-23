from collections import defaultdict

import hypothesis
import ndindex
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as stnp
from numpy.testing import assert_equal

from ..subchunk_map import as_subchunk_map


def non_negative_step_slices(size):
    start = st.one_of(st.integers(min_value=-size, max_value=size - 1), st.none())
    stop = st.one_of(st.integers(min_value=-size, max_value=size), st.none())
    # only non-negative steps (or None) are allowed
    step = st.one_of(st.integers(min_value=1, max_value=size), st.none())
    return st.builds(slice, start, stop, step)


@pytest.mark.slow
@given(st.data())
@hypothesis.settings(database=None, max_examples=10_000, deadline=None)
def test_as_subchunk_map(data):
    ndim = data.draw(st.integers(1, 4), label="ndim")
    shape = data.draw(st.tuples(*[st.integers(1, 100)] * ndim), label="shape")
    chunks = data.draw(st.tuples(*[st.integers(5, 20)] * ndim), label="chunks")
    idx = ndindex.Tuple(
        *[
            data.draw(non_negative_step_slices(shape[dim]), label=f"idx{dim}")
            for dim in range(ndim)
        ]
    )

    _check_as_subchunk_map(chunks, idx, shape)


@pytest.mark.slow
@given(st.data())
@hypothesis.settings(database=None, max_examples=10_000, deadline=None)
def test_as_subchunk_map_fancy_idx(data):
    ndim = data.draw(st.integers(1, 4), label="ndim")
    shape = data.draw(st.tuples(*[st.integers(1, 100)] * ndim), label="shape")
    chunks = data.draw(st.tuples(*[st.integers(5, 20)] * ndim), label="chunks")
    fancy_idx_axis = data.draw(st.integers(0, ndim - 1), label="fancy_idx_axis")
    fancy_idx = data.draw(
        stnp.arrays(
            np.intp,
            st.integers(0, shape[fancy_idx_axis] - 1),
            elements=st.integers(0, shape[fancy_idx_axis] - 1),
            unique=True,
        ),
        label="fancy_idx",
    )
    idx = ndindex.Tuple(
        *[
            data.draw(non_negative_step_slices(shape[dim]), label=f"idx{dim}")
            for dim in range(fancy_idx_axis)
        ],
        fancy_idx,
        *[
            data.draw(non_negative_step_slices(shape[dim]), label=f"idx{dim}")
            for dim in range(fancy_idx_axis + 1, ndim)
        ],
    )

    _check_as_subchunk_map(chunks, idx, shape)


@pytest.mark.slow
@given(st.data())
@hypothesis.settings(database=None, max_examples=10_000, deadline=None)
def test_as_subchunk_map_mask(data):
    ndim = data.draw(st.integers(1, 4), label="ndim")
    shape = data.draw(st.tuples(*[st.integers(1, 100)] * ndim), label="shape")
    chunks = data.draw(st.tuples(*[st.integers(5, 20)] * ndim), label="chunks")
    mask_idx_axis = data.draw(st.integers(0, ndim - 1), label="mask_idx_axis")
    mask_idx = data.draw(
        stnp.arrays(np.bool_, shape[mask_idx_axis], elements=st.booleans()),
        label="mask_idx",
    )
    idx = ndindex.Tuple(
        *[
            data.draw(non_negative_step_slices(shape[dim]), label=f"idx{dim}")
            for dim in range(mask_idx_axis)
        ],
        mask_idx,
        *[
            data.draw(non_negative_step_slices(shape[dim]), label=f"idx{dim}")
            for dim in range(mask_idx_axis + 1, ndim)
        ],
    )

    _check_as_subchunk_map(chunks, idx, shape)


def _check_as_subchunk_map(chunks, idx, shape):
    idx = idx.reduce(shape)
    if not isinstance(idx, ndindex.Tuple):
        idx = ndindex.Tuple(idx)
    chunk_size = ndindex.ChunkSize(chunks)

    as_subchunk_map_dict = defaultdict(list)
    for chunk, arr_subidx, chunk_subidx in as_subchunk_map(chunk_size, idx, shape):
        as_subchunk_map_dict[chunk].append((arr_subidx, chunk_subidx))
    as_subchunks_dict = defaultdict(list)
    for chunk in chunk_size.as_subchunks(idx, shape):
        arr_subidx = chunk.as_subindex(idx).raw
        chunk_subidx = idx.as_subindex(chunk).raw
        as_subchunks_dict[chunk].append((arr_subidx, chunk_subidx))
    assert list(as_subchunk_map_dict.keys()) == list(as_subchunks_dict.keys())
    for chunk in as_subchunk_map_dict:
        assert len(as_subchunk_map_dict[chunk]) == len(as_subchunks_dict[chunk]) == 1
        arr_subidx_1, chunk_subidx1 = as_subchunk_map_dict[chunk][0]
        arr_subidx_2, chunk_subidx2 = as_subchunks_dict[chunk][0]

        assert len(arr_subidx_1) == len(arr_subidx_2)
        for ix1, ix2 in zip(arr_subidx_1, arr_subidx_2):
            if isinstance(ix1, np.ndarray):
                assert_equal(ix1, ix2)
            else:
                assert ix1 == ix2

        assert len(chunk_subidx1) == len(chunk_subidx2)
        for ix1, ix2 in zip(chunk_subidx1, chunk_subidx2):
            if isinstance(ix1, np.ndarray):
                assert_equal(ix1, ix2)
            else:
                assert ix1 == ix2


def test_empty_index():
    # test we correctly handle empty index
    _check_as_subchunk_map((5,), ndindex.Slice(1, 1), (2,))
