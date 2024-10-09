# Note: this entire module is compiled by cython with wraparound=False
# See meson.build for details

from __future__ import annotations

import abc
import enum
import itertools
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import cython
import numpy as np
from cython import Py_ssize_t, bint
from ndindex import (
    BooleanArray,
    ChunkSize,
    Integer,
    IntegerArray,
    Slice,
    Tuple,
    ndindex,
)
from numpy.typing import NDArray

# Temporary hack to work around pytest issue: if a pure-pyton Cython module is first
# imported by `pytest .` and it contains doctests, it will be imported in pure python
# instead of its compiled form. This fails in CI, as Cython is not a runtime dependency,
# and regardless would cause a lot of type-checking normally performed by Cython to be
# skipped so it is to be avoided.
from . import hyperspace  # noqa: F401

if TYPE_CHECKING:
    # TODO import from typing and remove quotes (requires Python 3.10)
    # TODO use type <name> = ... (requires Python 3.12)
    from typing_extensions import TypeAlias

    AnySlicer: TypeAlias = "slice | NDArray[np.intp] | NDArray[np.bool] | int"
    AnySlicerND: TypeAlias = tuple[AnySlicer, ...]


class DropAxis(enum.Enum):
    _drop_axis = 0


# Returned instead of an AnySlicer. Signals that the axis should be removed when
# aggregated into an AnySlicerND.
DROP_AXIS = DropAxis._drop_axis


@cython.ccall
@cython.exceptval(check=False)
def ceil_a_over_b(a: Py_ssize_t, b: Py_ssize_t) -> Py_ssize_t:
    """Returns ceil(a/b). Assumes a >= 0 and b > 0.

    Note
    ----
    This module is compiled with the cython.cdivision flag. This causes behaviour to
    change if a and b have opposite signs and you try debugging the module in pure
    python, without compiling it. This function blindly assumes that a and b are always
    the same sign.
    """
    return a // b + (a % b > 0)


@cython.cfunc
@cython.exceptval(check=False)
def _smallest(x: Py_ssize_t, a: Py_ssize_t, m: Py_ssize_t) -> Py_ssize_t:
    """Find the smallest integer y >= x where y = a + k*m for whole k's
    Assumes 0 <= a <= x and m >= 1.

    a                  x    y
    | <-- m --> | <-- m --> |
    """
    return a + ceil_a_over_b(x - a, m) * m


class ChunkMapIndexError(IndexError):
    """Raised by IndexChunkMapper.chunk_submap() if the requested range of chunk indices
    does not intersect with IndexChunkMapper.chunk_indices.

    We define a custom exception instead of IndexError as raising this is a normal
    occourrence that will be caught quite a long way down the line and we don't want to
    silence an accidental, genuine IndexError.
    """


@cython.cclass
class IndexChunkMapper:
    """Abstract class that manipulates a numpy fancy index along a single axis of a
    chunked array

    Parameters
    ----------
    chunk_indices:
        Array of indices of all the chunks involved in the selection along the axis
    chunk_size:
        Size of each chunk, in points, along the axis
    dset_size:
        Size of the whole array, in points, along the axis
    """

    chunk_indices: Py_ssize_t[:]
    chunk_size: Py_ssize_t
    dset_size: Py_ssize_t
    n_chunks: Py_ssize_t
    last_chunk_size: Py_ssize_t

    def __init__(
        self,
        chunk_indices: Py_ssize_t[:],
        chunk_size: Py_ssize_t,
        dset_size: Py_ssize_t,
    ):
        self.chunk_indices = chunk_indices
        self.chunk_size = chunk_size
        self.dset_size = dset_size
        self.n_chunks = ceil_a_over_b(dset_size, chunk_size)
        self.last_chunk_size = (dset_size % chunk_size) or chunk_size

    @cython.cfunc
    def _chunk_start_stop(
        self, a: Py_ssize_t, b: Py_ssize_t
    ) -> tuple[Py_ssize_t, Py_ssize_t]:
        """Return the index of the first element of the chunk indexed by a and
        the index after the last element that comes before the chunk indexed by b.

        In other words, a slice of chunks
            slice(a, b)
        contains points
            slice(*chunk_start_stop(a, b))
        """
        indices = self.chunk_indices
        idx_len = len(indices)
        if idx_len == 0 or b <= indices[0] or a > indices[idx_len - 1]:
            raise ChunkMapIndexError()

        start = a * self.chunk_size
        stop = min(b * self.chunk_size, self.dset_size)
        return start, stop

    @cython.ccall
    @abc.abstractmethod
    def chunk_submap(
        self,
        chunk_start_idx: Py_ssize_t,
        chunk_stop_idx: Py_ssize_t,
        shift: bint,
    ) -> tuple[AnySlicer | DropAxis, AnySlicer]:
        """Given a range of chunk indices, return a tuple of

        - the slicer selecting the points within the sliced array
          (the return value for __getitem__, the value parameter for __setitem__)
        - the slicer selecting the points within the input chunks.
          If shift=True, this is relative to the start of the input chunks;
          If False, it is relative to the wider array.

        In other words, in the simplified case of slicing across axis=0 only:

        __getitem__:
            out, sub = mapper.chunk_submap(a, b, True)
            return_value[out] = arr[a * chunk_size:b * chunk_size][sub]

            out, sub = mapper.chunk_submap(a, b, False)
            return_value[out] = arr[sub]

        __setitem__:
            out, sub = mapper.chunk_submap(a, b, True)
            arr[a * chunk_size:b * chunk_size][sub] = value_param[out]

            out, sub = mapper.chunk_submap(a, b, False)
            arr[sub] = value_param[out]

        Note that when shift=False `sub` is not necessarily the same as `out`;
        consider for example a slice with step>1: `out` will be a slice with step=1
        while `sub` will have the same step as the input slice.

        Raises
        ------
        ChunkMapIndexError
            If there is no intersection between
            [chunk_start, chunk_stop[ and self.chunk_indices

        See Also
        --------
        zip_chunk_submap
        zip_slab_submap
        """

    @cython.cfunc
    def chunk_submap_compat(
        self, chunk_idx: Py_ssize_t
    ) -> tuple[Slice, AnySlicer | DropAxis, AnySlicer]:
        """Temporary compatibility layer with legacy as_chunk_submap(),
        adding the index of the data_dict to the return value.
        """
        start = chunk_idx * self.chunk_size
        stop = min(start + self.chunk_size, self.dset_size)
        out, sub = self.chunk_submap(chunk_idx, chunk_idx + 1, True)
        return Slice(start, stop, 1), out, sub

    @cython.ccall
    def chunk_indices_in_range(self, a: Py_ssize_t, b: Py_ssize_t) -> Py_ssize_t[:]:
        """Return the subset of chunk_indices in [a, b[

        Raises
        ------
        ChunkMapIndexError
            If there is no intersection
        """
        start_idx: Py_ssize_t
        stop_idx: Py_ssize_t
        start_idx, stop_idx = np.searchsorted(self.chunk_indices, [a, b], side="left")
        if stop_idx <= start_idx:
            raise ChunkMapIndexError()
        return self.chunk_indices[start_idx:stop_idx]

    @cython.ccall
    def chunks_indexer(self):  # -> slice | NDArray[np.intp]:
        """Return a numpy basic or advanced index, to be applied along the matching axis
        to an array with one point per chunk, that returns all chunks involved in the
        selection without altering the shape of the array.
        """
        return np.asarray(self.chunk_indices)

    @cython.ccall
    @abc.abstractmethod
    def whole_chunks_indexer(self):  # -> slice | NDArray[np.intp]:
        """Return a subset of chunks_indexer that selects only the chunks where all
        points of the chunk are included in the selection.

        e.g. if the index of this mapper is [True, True, False, False, True, False]
        and self.chunk_size=2, then:

        - self.chunks_indexer -> [0, 2]
        - self.whole_chunks_indexer -> [0]
        """


@cython.cclass
class SliceMapper(IndexChunkMapper):
    start: Py_ssize_t
    stop: Py_ssize_t
    step: Py_ssize_t

    def __init__(
        self,
        idx: slice,
        chunk_size: Py_ssize_t,
        dset_size: Py_ssize_t,
    ):
        self.start = idx.start
        self.stop = idx.stop
        self.step = idx.step

        if self.step <= 0:
            raise NotImplementedError(f"Slice step must be positive not {self.step}")

        if self.step > chunk_size:
            n = (self.stop - self.start + self.step - 1) // self.step
            chunk_indices = (self.start + np.arange(n) * self.step) // chunk_size
        else:
            chunk_start = self.start // chunk_size
            chunk_stop = (self.stop + chunk_size - 1) // chunk_size
            chunk_indices = np.arange(chunk_start, chunk_stop)

        super().__init__(chunk_indices, chunk_size, dset_size)

    @cython.ccall
    def chunk_submap(
        self, chunk_start_idx: Py_ssize_t, chunk_stop_idx: Py_ssize_t, shift: bint
    ) -> tuple[slice, slice]:
        start, stop = self._chunk_start_stop(chunk_start_idx, chunk_stop_idx)
        return (
            self._subindex_chunk_slice(start, stop),
            self._subindex_slice_chunk(start, stop, shift=shift),
        )

    @cython.cfunc
    def _subindex_chunk_slice(self, c_start: Py_ssize_t, c_stop: Py_ssize_t) -> slice:
        """Given a slice(self.start, self.stop, self.step) indexing an axis of
        an array, return the slice of the output array (on __getitem__) or value
        parameter array (on __setitem__) along the same axis that is targeted by the
        same slice after it's been clipped to select only the data within the range
        [c_start, c_stop[

        In other words:

        a = _subindex_chunk_slice(c_start, c_stop)
        b = _subindex_slice_chunk(c_start, c_stop)
        For __getitem__: out[a] = cache_chunk[b]
        For __setitem__: cache_chunk[b] = value[a]
        """
        s_start = self.start
        s_stop = self.stop
        s_step = self.step

        start = max(c_start, s_start)
        # Get the smallest lcm multiple of common that is >= start
        start = _smallest(start, s_start % s_step, s_step)
        # Finally, we need to shift start so that it is relative to index
        start = (start - s_start) // s_step

        stop = min(c_stop, s_stop)
        stop = ceil_a_over_b(stop - s_start, s_step) if stop > s_start else 0

        if start >= stop:
            raise ChunkMapIndexError()
        return slice(start, stop, 1)

    @cython.cfunc
    def _subindex_slice_chunk(
        self, c_start: Py_ssize_t, c_stop: Py_ssize_t, shift: bint
    ) -> slice:
        """Given a slice(s_start, s_stop, s_step) indexing an axis of
        an array, return a slice that's been clipped to select only
        the data within the range [c_start, c_stop[ and shifted back
        by s_start.

        See examples on _subindex_chunk_slice
        """
        s_start = self.start
        s_stop = self.stop
        s_step = self.step

        start = max(s_start, c_start)
        # Get the smallest step multiple of common that is >= start
        start = _smallest(start, s_start % s_step, s_step)
        # Finally, we need to shift start so that it is relative to index
        start -= c_start

        stop = max(0, min(s_stop, c_stop) - c_start)

        # This is the same, in the special case we're in, to
        #     return Slice(start, stop, s_step).reduce(d).raw
        # It's reimplemented here for speed.
        # assert 0 <= start < stop <= d
        step = s_step
        if start + step >= stop:
            stop, step = start + 1, 1  # Indexes 1 element
        else:
            stop -= (stop - start - 1) % step

        if not shift:
            start += c_start
            stop += c_start

        return slice(start, stop, step)

    @cython.ccall
    def chunks_indexer(self):  # -> slice | NDArray[np.intp]:
        indices = self.chunk_indices
        idx_len = len(indices)

        if idx_len == 0:
            return slice(0, 0, 1)
        elif self.step > self.chunk_size:
            return np.asarray(indices)
        else:
            chunk_start = indices[0]
            chunk_stop = indices[idx_len - 1] + 1
            return slice(int(chunk_start), int(chunk_stop), 1)

    @cython.ccall
    def whole_chunks_indexer(self):  # -> slice | NDArray[np.intp]:
        if self.chunk_size == 1:
            # All chunks are wholly selected
            return self.chunks_indexer()

        indices = self.chunk_indices
        idx_len = len(indices)
        if idx_len == 0:
            return slice(0, 0, 1)
        last_idx = indices[idx_len - 1]

        if self.step > 1:
            if self.last_chunk_size == 1 and last_idx == self.n_chunks - 1:
                # Last chunk contains exactly one point, so it's wholly covered.
                # For all other chunks, chunk_size > 1 and step > 1 so the selection
                # will never cover a whole chunk

                return slice(last_idx, self.n_chunks, 1)
            else:
                return slice(0, 0, 1)

        # step==1. The first and last chunk may be partially selected;
        # the rest are always wholly selected.
        chunk_start = indices[0]
        if self.start % self.chunk_size != 0:
            # First chunk is partially selected
            chunk_start += 1

        chunk_stop = last_idx  # excluded
        if self.stop == self.dset_size or self.stop % self.chunk_size == 0:
            # Last chunk is wholly selected
            chunk_stop += 1

        return slice(int(chunk_start), int(chunk_stop), 1)


@cython.cclass
class IntegerArrayMapper(IndexChunkMapper):
    idx: NDArray[np.intp]

    def __init__(
        self,
        idx: NDArray[np.intp],
        chunk_size: Py_ssize_t,
        dset_size: Py_ssize_t,
    ):
        if idx.ndim != 1:
            raise NotImplementedError("array index must be 1-dimensional")
        self.idx = idx
        chunk_indices = np.unique(idx // chunk_size)
        super().__init__(chunk_indices, chunk_size, dset_size)

    @cython.ccall
    def chunk_submap(
        self,
        chunk_start_idx: Py_ssize_t,
        chunk_stop_idx: Py_ssize_t,
        shift: bint,
    ) -> tuple[NDArray[np.bool], NDArray[np.intp]]:
        start, stop = self._chunk_start_stop(chunk_start_idx, chunk_stop_idx)

        mask = (start <= self.idx) & (self.idx < stop)
        sub = self.idx[mask]
        if len(sub) == 0:
            raise ChunkMapIndexError()
        if shift:
            sub -= start

        return mask, sub

    @cython.cfunc
    def _chunk_sizes_in_chunk_indices(self):  # -> NDArray[np.intp] | int:
        """Return the number of points taken from each chunk within self.chunk_indices.
        All but the last chunk always contain self.chunk_size points.
        """
        indices = self.chunk_indices
        idx_len = len(indices)
        if idx_len == 0:
            return self.chunk_size

        if indices[idx_len - 1] < self.dset_size // self.chunk_size:
            # Not the last chunk, or last chunk but exactly divisible by chunk_size
            return self.chunk_size

        out = np.full_like(indices, fill_value=self.chunk_size)
        out[idx_len - 1] = self.last_chunk_size
        return out

    @cython.ccall
    def whole_chunks_indexer(self):
        # Don't double count when the same index is picked twice
        idx_unique = np.unique(self.idx)
        _, counts = np.unique(idx_unique // self.chunk_size, return_counts=True)
        indices = np.asarray(self.chunk_indices)
        return indices[counts == self._chunk_sizes_in_chunk_indices()]


@cython.cclass
class BooleanArrayMapper(IndexChunkMapper):
    idx: NDArray[np.bool]
    _chunk_selected_counts: Py_ssize_t[:]
    _chunk_selected_offsets: Py_ssize_t[:]

    def __init__(
        self,
        idx: NDArray[np.bool],
        chunk_size: Py_ssize_t,
        dset_size: Py_ssize_t,
    ):
        if idx.ndim != 1:
            raise NotImplementedError("boolean mask index must be 1-dimensional")
        if idx.shape != (dset_size,):
            raise IndexError(
                f"boolean index did not match indexed array; dimension is {dset_size}, "
                f"but corresponding boolean dimension is {idx.shape[0]}"
            )

        self.idx = idx

        # pad i.array to be a multiple of n and group into chunks
        mask = np.pad(
            idx,
            (0, chunk_size - (dset_size % chunk_size)),
            "constant",
            constant_values=(False,),
        )
        mask = mask.reshape((mask.shape[0] // chunk_size, chunk_size))

        # count how many elements were selected in each chunk
        self._chunk_selected_counts = np.sum(mask, axis=1, dtype=np.intp)
        counts = self._chunk_selected_counts
        ncounts = len(counts)

        # compute offsets based on selected counts which will be used to build
        # the masks for each chunk
        offsets = np.empty(ncounts + 1, dtype=np.intp)
        offsets[0] = 0
        offsets[1:] = np.cumsum(counts)
        self._chunk_selected_offsets = offsets

        chunk_indices = np.flatnonzero(counts)
        super().__init__(chunk_indices, chunk_size, dset_size)

    @cython.ccall
    def chunk_submap(
        self,
        chunk_start_idx: Py_ssize_t,
        chunk_stop_idx: Py_ssize_t,
        shift: bint,
    ) -> tuple[slice, NDArray[np.intp]]:
        start, stop = self._chunk_start_stop(chunk_start_idx, chunk_stop_idx)

        sub = np.flatnonzero(self.idx[start:stop])
        if len(sub) == 0:
            raise ChunkMapIndexError()
        if not shift:
            sub += start

        offsets = self._chunk_selected_offsets
        counts = self._chunk_selected_counts
        n_before = offsets[chunk_start_idx]
        n_select = 0
        for i in range(chunk_start_idx, chunk_stop_idx):
            n_select += counts[i]

        return (
            slice(n_before, n_before + n_select, 1),
            sub,
        )

    @cython.ccall
    def whole_chunks_indexer(self):
        counts = np.asarray(self._chunk_selected_counts)
        if self.last_chunk_size == self.chunk_size:
            return np.flatnonzero(counts == self.chunk_size)

        chunk_sizes = np.full(self.n_chunks, fill_value=self.chunk_size, dtype=np.intp)
        chunk_sizes[self.n_chunks - 1] = self.last_chunk_size
        return np.flatnonzero(counts == chunk_sizes)


@cython.cclass
class IntegerMapper(IndexChunkMapper):
    idx: Py_ssize_t

    def __init__(
        self,
        idx: Py_ssize_t,
        chunk_size: Py_ssize_t,
        dset_size: Py_ssize_t,
    ):
        assert 0 <= idx < dset_size
        self.idx = idx
        chunk_indices = np.array([idx // chunk_size], dtype=np.intp)
        super().__init__(chunk_indices, chunk_size, dset_size)

    @cython.ccall
    def chunk_submap(
        self,
        chunk_start_idx: Py_ssize_t,
        chunk_stop_idx: Py_ssize_t,
        shift: bint,
    ) -> tuple[DropAxis, int]:
        # Raise if we're selecting an index out of bounds
        start, _ = self._chunk_start_stop(chunk_start_idx, chunk_stop_idx)
        return DROP_AXIS, self.idx - start if shift else self.idx

    @cython.ccall
    def chunks_indexer(self):
        # a[i] would change the shape
        # a[[i]] (the default without overriding this method) would return a copy
        # a[i:i+1] returns a view, which is faster
        i = self.chunk_indices[0]
        return slice(i, i + 1, 1)

    @cython.ccall
    def whole_chunks_indexer(self):
        if self.chunk_size == 1:
            return self.chunks_indexer()

        # If the index is in the last chunk and the last chunk is size 1, then it's
        # wholly selected. In any other case, it's partially selected.
        partial = self.dset_size % self.chunk_size
        if partial != 1:
            return slice(0, 0, 1)
        if self.chunk_indices[0] != self.dset_size // self.chunk_size:
            return slice(0, 0, 1)
        return self.chunks_indexer()


@cython.cclass
class EverythingMapper(IndexChunkMapper):
    """Select all points along an axis [:]"""

    def __init__(self, chunk_size: Py_ssize_t, dset_size: Py_ssize_t):
        n_chunks = ceil_a_over_b(dset_size, chunk_size)
        super().__init__(np.arange(n_chunks, dtype=np.intp), chunk_size, dset_size)

    @cython.ccall
    def chunk_submap(
        self,
        chunk_start_idx: Py_ssize_t,
        chunk_stop_idx: Py_ssize_t,
        shift: bint,
    ) -> tuple[slice, slice]:
        start, stop = self._chunk_start_stop(chunk_start_idx, chunk_stop_idx)
        out = slice(start, stop, 1)
        sub = slice(0, stop - start, 1) if shift else out
        return out, sub

    @cython.ccall
    def chunks_indexer(self):
        return slice(0, self.n_chunks, 1)

    @cython.ccall
    def whole_chunks_indexer(self):
        return slice(0, self.n_chunks, 1)


def index_chunk_mappers(
    idx: Any,
    chunk_size: tuple[int, ...] | ChunkSize,
    shape: tuple[int, ...],
) -> tuple[Tuple, list[IndexChunkMapper]]:
    """Preprocess a numpy fancy index used in __getitem__ or __setitem__

    Returns
    -------
    - ndindex.Tuple with the preprocessed index
    - list of IndexChunkMapper objects, one per axis
      (including those omitted in the index)
    """
    assert isinstance(chunk_size, (tuple, ChunkSize))
    if not all(c > 0 for c in chunk_size):
        raise ValueError("chunk sizes must be structly positive")

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
        return idx, []

    idx_len = len(idx.args)

    prefix_chunk_size = chunk_size[:idx_len]
    prefix_shape = shape[:idx_len]

    suffix_chunk_size = chunk_size[idx_len:]
    suffix_shape = shape[idx_len:]

    n: Py_ssize_t
    d: Py_ssize_t
    mappers = []

    # Process the prefix of the axes which idx selects on
    for i, n, d in zip(idx.args, prefix_chunk_size, prefix_shape):
        i = i.reduce((d,))

        # Compute chunk_idxs, e.g., chunk_idxs == [2, 4] for chunk sizes (100, 1000)
        # would correspond to chunk (slice(200, 300), slice(4000, 5000)).
        mapper: IndexChunkMapper

        if isinstance(i, Slice):
            mapper = SliceMapper(i.raw, n, d)
        elif isinstance(i, IntegerArray):
            mapper = IntegerArrayMapper(i.raw, n, d)
        elif isinstance(i, BooleanArray):
            mapper = BooleanArrayMapper(i.raw, n, d)
        elif isinstance(i, Integer):
            mapper = IntegerMapper(i.raw, n, d)
        else:
            raise NotImplementedError(f"index type {type(i)} not supported")
        mappers.append(mapper)

    # Handle the remaining suffix axes on which we did not select, we still need to
    # break them up into chunks.
    for n, d in zip(suffix_chunk_size, suffix_shape):
        mapper = EverythingMapper(n, d)
        mappers.append(mapper)

    return idx, mappers


def as_subchunk_map(
    chunk_size: tuple[int, ...] | ChunkSize,
    idx: Any,
    shape: tuple[int, ...],
) -> Iterator[
    tuple[
        Tuple,
        AnySlicerND,
        AnySlicerND,
    ]
]:
    """Computes the chunk selection assignment. In particular, given a `chunk_size`
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
    idx, mappers = index_chunk_mappers(idx, chunk_size, shape)
    if not mappers:
        return
    idx_len = len(idx.args)

    mapper: IndexChunkMapper  # noqa: F842
    chunk_subindexes = [
        [mapper.chunk_submap_compat(chunk_idx) for chunk_idx in mapper.chunk_indices]
        for mapper in mappers
    ]

    # Now combine the chunk_slices and subindexes for each dimension into tuples
    # across all dimensions.
    for p in itertools.product(*chunk_subindexes):
        chunk_slices, arr_subidxs, chunk_subidxs = zip(*p)

        # skip dimensions which were sliced away
        arr_subidxs = tuple(
            arr_subidx for arr_subidx in arr_subidxs if arr_subidx is not DROP_AXIS
        )
        # skip suffix dimensions
        chunk_subidxs = chunk_subidxs[:idx_len]

        yield Tuple(*chunk_slices), arr_subidxs, chunk_subidxs


@cython.ccall
def zip_chunk_submap(
    mappers: list[IndexChunkMapper], chunk_idx: Py_ssize_t[:]
) -> tuple[AnySlicerND, AnySlicerND]:
    """Call IndexChunkMapper.chunk_submap() along every axis
    and return the resulting indices to slice a single chunk

    Parameters
    ----------
    mappers:
        Output of index_chunk_mappers()
    chunk_idx:
        Chunk index

    Returns
    -------
    out
        The n-dimensional numpy index selecting the points within the return value of
        __getitem__ or within the value parameter of __setitem__
    sub
        The n-dimensional numpy index selecting the points within the chunk,
        with all addresses shifted to the start of the chunk.

    In other words::
        chunk = arr[tuple(slice(i*c, (i+1)*c) for i, c in zip(chunk_idx, chunk_size)]
        return_value[out] = chunk[sub]  # __getitem__
        chunk[sub] = value_param[out]  # __setitem__

    Raises
    ------
    ChunkMapIndexError
        If the chunk index is not among those selected by IndexChunkMapper.chunk_indices

    See Also
    --------
    zip_slab_submap
    IndexChunkMapper.chunk_submap

    Examples
    --------
    >>> _, mappers = index_chunk_mappers(
    ...     (slice(15, 35, 2), 19), chunk_size=(10, 10), shape=(100, 100)
    ... )
    >>> zip_chunk_submap(mappers, np.asarray((3, 1), dtype=np.intp))
    ((slice(8, 10, 1),), (slice(1, 4, 2), 9))
    """
    out_idx = []
    sub_idx = []

    for i in range(len(chunk_idx)):
        mapper: IndexChunkMapper = mappers[i]
        a = chunk_idx[i]
        out, sub = mapper.chunk_submap(a, a + 1, shift=True)

        # skip axes which were sliced by a scalar
        if out is not DROP_AXIS:
            out_idx.append(out)
        sub_idx.append(sub)

    return tuple(out_idx), tuple(sub_idx)


@cython.ccall
def zip_slab_submap(
    mappers: list[IndexChunkMapper],
    fromc_idx: Py_ssize_t[:],
    toc_idx: Py_ssize_t[:],
) -> tuple[AnySlicerND, AnySlicerND]:
    """Call IndexChunkMapper.chunk_submap() along every axis
    and return the resulting indices to slice the whole array
    within the limits of a hyperrectangular selection of chunks.

    Parameters
    ----------
    mappers:
        Output of index_chunk_mappers()
    fromc_idx:
        Top-left corner, included, of the hyperrectangle of chunks
    toc_idx:
        Bottom-right corner, excluded, of the hyperrectangle of chunks to process.

    Returns
    -------
    out
        The n-dimensional numpy index selecting the points within the return value of
        __getitem__ or within the value parameter of __setitem__
    sub
        The n-dimensional numpy index selecting the input points within the whole array,
        relative to the beginning of the entire array itself (not the selection).

    In other words::
        return_value[sub] = arr[out]  # __getitem__
        arr[out] = value_param[sub]  # __setitem__

    Raises
    ------
    ChunkMapIndexError
        If there is no intersection between the index stored in the mappers and the
        chunks in the range [fromc_idx, toc_idx[

    See Also
    --------
    zip_chunk_submap
    IndexChunkMapper.chunk_submap

    Examples
    --------
    >>> _, mappers = index_chunk_mappers(
    ...     (slice(15, 35, 2), 19), chunk_size=(10, 10), shape=(100, 100)
    ... )
    >>> zip_slab_submap(mappers, np.asarray((2, 1), np.intp), np.asarray((4, 4), np.intp))
    ((slice(3, 10, 1),), (slice(21, 34, 2), 19))

    Notes
    -----
    This is a separate function from zip_chunk_submap, despite the two having very
    similar code, because Cython does not optimize well input parameters that can have
    multiple types (such as toc_idx: Py_ssize_t[:] | None).
    """
    out_idx = []
    sub_idx = []

    for i in range(len(fromc_idx)):
        mapper: IndexChunkMapper = mappers[i]
        a = fromc_idx[i]
        b = toc_idx[i]
        # Raises ChunkMapIndexError if there is no intersection
        out, sub = mapper.chunk_submap(a, b, shift=False)

        # skip axes which were sliced by a scalar
        if out is not DROP_AXIS:
            out_idx.append(out)
        sub_idx.append(sub)

    return tuple(out_idx), tuple(sub_idx)


@cython.ccall
def cartesian_product(views: list[Py_ssize_t[:]]) -> Py_ssize_t[:, :]:
    """Cartesian product of 1D views of indices

    Same as np.array(list(itertools.product(*arrays)))

    Adapted from https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points

    >>> np.asarray(cartesian_product([np.array([1, 2]), np.array([3, 4])]))
    array([[1, 3],
           [1, 4],
           [2, 3],
           [2, 4]])
    """
    arrays = [np.asarray(v) for v in views]
    la = len(arrays)
    if not la:
        return np.empty((0, 0), dtype=np.intp)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=np.intp)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)
