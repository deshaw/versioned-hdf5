from __future__ import annotations

from typing import Any, Literal

import hypothesis
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_array_equal

from ..staged_changes import StagedChangesArray
from .test_subchunk_map import idx_st, shape_chunks_st

max_examples = 10_000


def action_st(shape: tuple[int, ...], max_size: int = 20):
    resize_st = st.tuples(*(st.integers(0, max_size) for _ in shape))
    return st.one_of(
        st.tuples(st.just("getitem"), idx_st(shape)),
        st.tuples(st.just("setitem"), idx_st(shape)),
        st.tuples(st.just("resize"), resize_st),
    )


@st.composite
def staged_array_st(
    draw, max_ndim: int = 4, max_size: int = 20, max_actions: int = 6
) -> tuple[
    Literal["full", "from_array"],  # how to create the base array
    tuple[int, ...],  # initial shape
    tuple[int, ...],  # chunk size
    list[tuple[Literal["getitem", "setitem", "resize"], Any]],  # 0 or more actions
]:
    base, (shape, chunks), n_actions = draw(
        st.tuples(
            st.one_of(st.just("full"), st.just("from_array")),
            shape_chunks_st(max_ndim=max_ndim, min_size=0, max_size=max_size),
            st.integers(0, max_actions),
        )
    )

    orig_shape = shape
    actions = []
    for _ in range(n_actions):
        label, arg = draw(action_st(shape, max_size=max_size))
        actions.append((label, arg))
        if label == "resize":
            shape = arg

    return base, orig_shape, chunks, actions


@pytest.mark.slow
@given(staged_array_st())
@hypothesis.settings(max_examples=max_examples, deadline=None)
def test_staged_array(args):
    base, shape, chunks, actions = args

    rng = np.random.default_rng(0)
    fill_value = 42

    if base == "full":
        base = np.full(shape, fill_value, dtype="u4")
        arr = StagedChangesArray.full(shape, chunks, fill_value, dtype=base.dtype)
        assert arr.n_base_slabs == 0
    elif base == "from_array":
        base = rng.integers(2**32, size=shape, dtype="u4")
        # copy base to detect bugs where the StagedChangesArray
        # accidentally writes back to the base slabs
        arr = StagedChangesArray.from_array(base.copy(), chunks, fill_value)
        if base.size == 0:
            assert arr.n_base_slabs == 0
        else:
            assert arr.n_base_slabs > 0
    else:
        raise AssertionError("unreachable")

    assert arr.dtype == base.dtype
    assert arr.fill_value.shape == ()
    assert arr.fill_value == fill_value
    assert arr.fill_value.dtype == base.dtype
    assert arr.slabs[0].dtype == base.dtype
    assert arr.slabs[0].shape == chunks
    assert arr.itemsize == 4
    assert arr.size == np.prod(shape)
    assert arr.nbytes == np.prod(shape) * 4
    assert arr.shape == shape
    assert len(arr) == shape[0]
    assert arr.ndim == len(shape)
    assert arr.chunk_size == chunks
    assert len(arr.n_chunks) == len(shape)
    for n, s, c in zip(arr.n_chunks, shape, chunks):
        assert n == s // c + (s % c > 0)
    assert arr.n_staged_slabs == 0
    assert arr.n_base_slabs + 1 == arr.n_slabs == len(arr.slabs)
    assert not arr.has_changes

    expect = base.copy()

    for label, arg in actions:
        if label == "getitem":
            assert_array_equal(arr[arg], expect[arg], strict=True)

        elif label == "setitem":
            value = rng.integers(2**32, size=expect[arg].shape, dtype="u4")
            expect[arg] = value
            arr[arg] = value
            assert_array_equal(arr, expect, strict=True)

        elif label == "resize":
            # Can't use np.resize(), which works differently as it reflows the data
            new = np.full(arg, fill_value, dtype="u4")
            common_idx = tuple(slice(min(o, n)) for o, n in zip(arr.shape, arg))
            new[common_idx] = expect[common_idx]
            expect = new
            arr.resize(arg)
            assert arr.shape == arg
            assert_array_equal(arr, expect, strict=True)

        else:
            raise AssertionError("unreachable")

    # Test has_changes property.
    # Edge cases would be complicated to test:
    # - a __setitem__ with empty index or a resize with the same shape won't
    #   flip has_changes
    # - a round-trip where an array is created empty, resized, filled, wiped by
    #   resize(0), and then # resized to the original shape will have has_changes=True
    #   even if identical to the original.
    if expect.shape != base.shape or (expect != base).any():
        assert arr.has_changes
    elif all(label == "getitem" for label, _ in actions):
        assert not arr.has_changes

    # One final __getitem__ of everything
    assert_array_equal(arr, expect, strict=True)

    # Reconstruct the array starting from the base + changes
    final = np.full(expect.shape, fill_value, dtype=base.dtype)
    shapes = [base.shape] + [arg for label, arg in actions if label == "resize"]
    common_idx = tuple(slice(min(sizes)) for sizes in zip(*shapes))
    final[common_idx] = base[common_idx]
    for value_idx, _, chunk in arr.changes():
        if not isinstance(chunk, tuple):
            assert chunk.dtype == base.dtype
            final[value_idx] = chunk
    assert_array_equal(final, expect, strict=True)

    # Test __iter__
    assert_array_equal(list(arr), list(expect), strict=True)


# TODO weird dtypes
# TODO astype()
# TODO refill()
# TODO copy()
# TODO load()
# TODO __repr__
# TODO invalid parameters
