import itertools
from typing import Iterable, Iterator

import cython
import numpy as np
from cython import Py_ssize_t
from ndindex import (
    BooleanArray,
    ChunkSize,
    Integer,
    IntegerArray,
    Slice,
    Tuple,
    ndindex,
)


@cython.cfunc
def _ceiling(a: Py_ssize_t, b: Py_ssize_t) -> Py_ssize_t:
    """
    Returns ceil(a/b)
    """
    return -(-a // b)


@cython.cfunc
def _smallest(x: Py_ssize_t, a: Py_ssize_t, m: Py_ssize_t) -> Py_ssize_t:
    """
    Find the smallest integer y >= x where y = a + k*m for whole k's
    Assumes 0 <= a <= x and m >= 1.

    a                  x    y
    | <-- m --> | <-- m --> |
    """
    n: Py_ssize_t = _ceiling(x - a, m)
    return a + n * m


@cython.cfunc
def _subindex_chunk_slice(
    c_start: Py_ssize_t,
    c_stop: Py_ssize_t,
    i_start: Py_ssize_t,
    i_stop: Py_ssize_t,
    i_step: Py_ssize_t,
) -> slice:
    common: Py_ssize_t = i_start % i_step

    start: Py_ssize_t = max(c_start, i_start)
    # Get the smallest lcm multiple of common that is >= start
    start = _smallest(start, common, i_step)
    # Finally, we need to shift start so that it is relative to index
    start = (start - i_start) // i_step

    stop: Py_ssize_t = min(c_stop, i_stop)
    stop = _ceiling(stop - i_start, i_step) if stop > i_start else 0

    return slice(start, stop, 1)


@cython.cfunc
def _subindex_slice_chunk(
    s_start: Py_ssize_t,
    s_stop: Py_ssize_t,
    s_step: Py_ssize_t,
    c_start: Py_ssize_t,
    c_stop: Py_ssize_t,
) -> slice:
    common: Py_ssize_t = s_start % s_step

    start: Py_ssize_t = max(s_start, c_start)
    # Get the smallest step multiple of common that is >= start
    start = _smallest(start, common, s_step)
    # Finally, we need to shift start so that it is relative to index
    start -= c_start

    stop: Py_ssize_t = max(0, min(s_stop, c_stop) - c_start)

    # This is the same, in the special case we're in, to
    #     return Slice(start, stop, s_step).reduce(d).raw
    # It's reimplemented here for speed.
    # assert 0 <= start < stop <= d
    step: Py_ssize_t = s_step
    if start + step >= stop:
        stop, step = start + 1, 1  # Indexes 1 element
    else:
        stop -= (stop - start - 1) % step
        if stop - start == 1:
            step = 1  # Indexes 1 element
    return slice(start, stop, step)


def as_subchunk_map(
    chunk_size: ChunkSize, idx, shape: tuple[int, ...]
) -> Iterator[
    tuple[
        tuple[Slice, ...],
        tuple[slice | np.ndarray | tuple[()], ...],
        tuple[slice | np.ndarray | tuple[()] | int, ...],
    ]
]:
    """
    Computes the chunk selection assignment. In particular, given a `chunk_size`
    it returns triple (chunk_slices, arr_subidxs, chunk_subidxs) such that for a
    chunked Dataset `ds` we can translate selections like

    >> ds[idx]

    into selecting from the individual chunks of `ds` as

    >> arr = np.ndarray(output_shape)
    >> for chunk, arr_idx_raw, index_raw in as_subchunk_map(ds.chunk_size, idx, ds.shape):
    ..     arr[arr_idx_raw] = ds.data_dict[chunk][index_raw]

    Similarly, assignments like

    >> ds[idx] = arr

    can be translated into

    >> for chunk, arr_idx_raw, index_raw in as_subchunk_map(ds.chunk_size, idx, ds.shape):
    ..     ds.data_dict[chunk][index_raw] = arr[arr_idx_raw]

    :param chunk_size: the `ChunkSize` of the Dataset
    :param idx: the "index" to read from / write to the Dataset
    :param shape: the shape of the Dataset
    :return: a generator of `(chunk, arr_idx_raw, index_raw)` tuples
    """
    assert isinstance(chunk_size, ChunkSize)
    if isinstance(idx, Tuple):
        pass
    elif isinstance(idx, tuple):
        idx = Tuple(*idx)
    else:
        idx = Tuple(ndindex(idx))
    assert isinstance(shape, tuple)

    if any(dim < 0 for dim in shape):
        raise ValueError("shape dimensions must be non-negative")

    if len(shape) != len(chunk_size):
        raise ValueError("chunks dimensions must equal the array dimensions")

    if idx.isempty(shape):
        # abort early for empty index
        return

    idx_len: Py_ssize_t = len(idx.args)

    prefix_chunk_size = chunk_size[:idx_len]
    prefix_shape = shape[:idx_len]

    suffix_chunk_size = chunk_size[idx_len:]
    suffix_shape = shape[idx_len:]

    chunk_subindexes = []

    n: Py_ssize_t
    d: Py_ssize_t
    s: Py_ssize_t
    i: Slice | IntegerArray | BooleanArray | Integer
    chunk_idxs: Iterable[Py_ssize_t]
    chunk_idx: Py_ssize_t
    chunk_start: Py_ssize_t
    chunk_stop: Py_ssize_t

    # Process the prefix of the axes which idx selects on
    for n, i, d in zip(prefix_chunk_size, idx.args, prefix_shape):
        i = i.reduce((d,))

        # Compute chunk_idxs, e.g., chunk_idxs == [2, 4] for chunk sizes (100, 1000)
        # would correspond to chunk (slice(200, 300), slice(4000, 5000)).
        chunk_subindexes_for_axis: list = []
        if isinstance(i, Slice):
            if i.step <= 0:
                raise NotImplementedError(f"Slice step must be positive not {i.step}")

            start: Py_ssize_t = i.start
            stop: Py_ssize_t = i.stop
            step: Py_ssize_t = i.step

            if step > n:
                chunk_idxs = (
                    (start + k * step) // n
                    for k in range((stop - start + step - 1) // step)
                )
            else:
                chunk_idxs = range(start // n, (stop + n - 1) // n)

            for chunk_idx in chunk_idxs:
                chunk_start = chunk_idx * n
                chunk_stop = min((chunk_idx + 1) * n, d)

                chunk_subindexes_for_axis.append(
                    (
                        Slice(chunk_start, chunk_stop, 1),
                        _subindex_chunk_slice(
                            chunk_start, chunk_stop, start, stop, step
                        ),
                        _subindex_slice_chunk(
                            start, stop, step, chunk_start, chunk_stop
                        ),
                    )
                )
        elif isinstance(i, IntegerArray):
            assert i.ndim == 1
            chunk_idxs = np.unique(i.array // n)

            for chunk_idx in chunk_idxs:
                chunk_start = chunk_idx * n
                chunk_stop = min((chunk_idx + 1) * n, d)
                i_chunk_mask = (chunk_start <= i.array) & (i.array < chunk_stop)
                chunk_subindexes_for_axis.append(
                    (
                        Slice(chunk_start, chunk_stop, 1),
                        i_chunk_mask,
                        i.array[i_chunk_mask] - chunk_start,
                    )
                )
        elif isinstance(i, BooleanArray):
            if i.ndim != 1:
                raise NotImplementedError("boolean mask index must be 1-dimensional")
            if i.shape != (d,):
                raise IndexError(
                    f"boolean index did not match indexed array; dimension is {d}, "
                    f"but corresponding boolean dimension is {i.shape[0]}"
                )

            # pad i.array to be a multiple of n and group into chunks
            mask = np.pad(
                i.array, (0, n - (d % n)), "constant", constant_values=(False,)
            )
            mask = mask.reshape((mask.shape[0] // n, n))

            # count how many elements were selected in each chunk
            chunk_selected_counts = np.sum(mask, axis=1, dtype=np.intp)

            # compute offsets based on selected counts which will be used to build the masks for each chunk
            chunk_selected_offsets = np.zeros(
                len(chunk_selected_counts) + 1, dtype=np.intp
            )
            chunk_selected_offsets[1:] = np.cumsum(chunk_selected_counts)

            # chunk_idxs for the chunks which are not empty
            chunk_idxs = np.flatnonzero(chunk_selected_counts)

            for chunk_idx in chunk_idxs:
                chunk_start = chunk_idx * n
                chunk_stop = min((chunk_idx + 1) * n, d)
                chunk_subindexes_for_axis.append(
                    (
                        Slice(chunk_start, chunk_stop, 1),
                        np.concatenate(
                            [
                                np.zeros(chunk_selected_offsets[chunk_idx], dtype=bool),
                                np.ones(chunk_selected_counts[chunk_idx], dtype=bool),
                                np.zeros(
                                    chunk_selected_offsets[-1]
                                    - chunk_selected_offsets[chunk_idx + 1],
                                    dtype=bool,
                                ),
                            ]
                        ),
                        np.flatnonzero(i.array[chunk_start:chunk_stop]),
                    )
                )
        elif isinstance(i, Integer):
            i_raw: Py_ssize_t = i.raw
            chunk_idx = i_raw // n
            chunk_start = chunk_idx * n
            chunk_stop = min((chunk_idx + 1) * n, d)
            chunk_subindexes_for_axis.append(
                (
                    Slice(chunk_start, chunk_stop, 1),
                    (),
                    i_raw - chunk_start,
                )
            )
        else:
            raise NotImplementedError(f"index type {type(i)} not supported")

        chunk_subindexes.append(chunk_subindexes_for_axis)

    # Handle the remaining suffix axes on which we did not select, we still need to break
    # them up into chunks.
    for n, d in zip(suffix_chunk_size, suffix_shape):
        chunk_slices = (
            Slice(chunk_idx * n, min((chunk_idx + 1) * n, d), 1)
            for chunk_idx in range((d + n - 1) // n)
        )
        chunk_subindexes.append(
            [(chunk_slice, chunk_slice.raw, ()) for chunk_slice in chunk_slices]
        )

    # Now combine the chunk_slices and subindexes for each dimension into tuples
    # across all dimensions.
    for p in itertools.product(*chunk_subindexes):
        chunk_slices, arr_subidxs, chunk_subidxs = zip(*p)

        # skip dimensions which were sliced away
        arr_subidxs = tuple(
            arr_subidx
            for arr_subidx in arr_subidxs
            if not isinstance(arr_subidx, tuple) or arr_subidx != ()
        )

        # skip suffix dimensions
        chunk_subidxs = chunk_subidxs[:idx_len]

        yield Tuple(*chunk_slices), arr_subidxs, chunk_subidxs
