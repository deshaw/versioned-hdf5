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
from cython import bint
from ndindex import ChunkSize, Slice, Tuple, ndindex
from numpy.typing import NDArray

from .cytools import (
    ceil_a_over_b,
    count2stop,
    np_hsize_t,
    smallest_step_after,
    stop2count,
)
from .tools import asarray

if TYPE_CHECKING:  # pragma: nocover
    # TODO import from typing and remove quotes (requires Python 3.10)
    # TODO use type <name> = ... (requires Python 3.12)
    from typing_extensions import TypeAlias

    AnySlicer: TypeAlias = "slice | NDArray[np_hsize_t] | int"
    AnySlicerND: TypeAlias = tuple[AnySlicer, ...]

if cython.compiled:  # pragma: nocover
    from cython.cimports.versioned_hdf5.cytools import (  # type: ignore
        ceil_a_over_b,
        count2stop,
        hsize_t,
        smallest_step_after,
        stop2count,
    )

if cython.compiled:  # pragma: nocover
    from cython.cimports.versioned_hdf5.cytools import (  # type: ignore
        ceil_a_over_b,
        smallest_step_after,
    )


class DropAxis(enum.Enum):
    _drop_axis = 0


# Returned instead of an AnySlicer. Signals that the axis should be removed when
# aggregated into an AnySlicerND.
DROP_AXIS = DropAxis._drop_axis


# This is an abstract class, which should inherit from abc.ABC
# or have metaclass=abc.ABCMeta. Neither are supported by Cython though.
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

    chunk_indices: hsize_t[:]
    dset_size: hsize_t
    chunk_size: hsize_t
    n_chunks: hsize_t
    last_chunk_size: hsize_t

    def __init__(
        self,
        chunk_indices: hsize_t[:],
        dset_size: hsize_t,
        chunk_size: hsize_t,
    ):
        self.chunk_indices = chunk_indices
        self.dset_size = dset_size
        self.chunk_size = chunk_size
        self.n_chunks = ceil_a_over_b(dset_size, chunk_size)
        self.last_chunk_size = (dset_size % chunk_size) or chunk_size

    @cython.cfunc
    @cython.nogil
    @cython.exceptval(check=False)
    def _chunk_start_stop(self, chunk_idx: hsize_t) -> tuple[hsize_t, hsize_t]:
        """Return the range of points [a, b[ of the chunk indexed by chunk_idx"""
        if not cython.compiled:
            chunk_idx = int(chunk_idx)  # np.uint64 + int -> np.float64

        start = chunk_idx * self.chunk_size
        stop = min(start + self.chunk_size, self.dset_size)
        return start, stop

    @cython.ccall
    @abc.abstractmethod
    def chunk_submap(
        self, chunk_idx: hsize_t
    ) -> tuple[Slice, AnySlicer | DropAxis, AnySlicer]:
        """Given a chunk index, return a tuple of

        data_dict key
            key of the data_dict (see build_data_dict())
        value_subidx
            the slicer selecting the points within the sliced array
            (the return value for __getitem__, the value parameter for __setitem__)
        chunk_subidx
            the slicer selecting the points within the input chunks.

        In other words, in the simplified one-dimensional case:

            _, value_subidx, chunk_subidx = mapper.chunk_submap(i)
            chunk_view = base_arr[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
            return_value[value_subidx] = chunk_view[chunk_subidx]  # __getitem__
            chunk_view[chunk_subidx] = value_param[value_subidx]  # __setitem__
        """

    @cython.ccall
    @abc.abstractmethod
    def chunks_indexer(self):  # -> slice | NDArray[np_hsize_t]:
        """Return a numpy basic or advanced index, to be applied along the matching axis
        to an array with one point per chunk, that returns all chunks involved in the
        selection without altering the shape of the array.
        """

    @cython.ccall
    @abc.abstractmethod
    def whole_chunks_idxidx(self):  # -> slice | NDArray[np.intp]:
        """Return an index that further refines the output of chunks_indexer and
        selects only the chunks where all points of the chunk are included in the
        selection.

        e.g. if the index of this mapper is [True, False, False, False, True, True]
        and self.chunk_size=2, then:

        - self.chunks_indexer() -> [0, 2]
        - self.whole_chunks_idxidx() -> [1]  # relative to the output of chunks_indexer

        In other words:

        >> all_chunk_indices = np.arange(self.n_chunks)
        >> selected_chunk_indices = all_chunk_indices[self.chunks_indexer()]
        >> whole_chunk_indices = selected_chunk_indices[self.whole_chunks_idxidx()]
        """

    @property
    def value_view_idx(self) -> slice | None:
        """Return an index, to be applied to the matching axis of the __getitem__
        return value or the __setitem__ value parameter, that will create a view with
        has the same dimensionality as the base array.

        This allows passing the view to read_many_slices().

        Example
        -------
        >>> idx = (slice(3), 1, slice(None))
        >>> chunks = (4, 4, 4)
        >>> shape = (10, 10, 10)
        >>> ndindex_idx, mappers = index_chunk_mappers(idx, shape, chunks)
        >>> value = np.empty(ndindex_idx.newshape(shape))  # e.g. for __getitem__
        >>> value_view = value[tuple(mapper.value_view_idx for mapper in mappers)]
        >>> value.shape
        (3, 10)
        >>> value_view.shape
        (3, 1, 10)
        >>> value_view.base is value
        True
        """
        return slice(None)


@cython.cclass
class SliceMapper(IndexChunkMapper):
    """IndexChunkMapper for slices"""

    start: hsize_t = cython.declare("hsize_t", visibility="readonly")
    stop: hsize_t = cython.declare("hsize_t", visibility="readonly")
    step: hsize_t = cython.declare("hsize_t", visibility="readonly")

    def __init__(
        self,
        idx: slice,
        dset_size: hsize_t,
        chunk_size: hsize_t,
    ):
        self.start = idx.start
        self.stop = idx.stop
        self.step = idx.step

        if self.step <= 0:
            raise NotImplementedError(f"Slice step must be positive not {self.step}")

        if self.step > chunk_size:
            n = (self.stop - self.start + self.step - 1) // self.step
            chunk_indices = (
                self.start + np.arange(n, dtype=np_hsize_t) * self.step
            ) // chunk_size
        else:
            chunk_start = self.start // chunk_size
            chunk_stop = (self.stop + chunk_size - 1) // chunk_size
            chunk_indices = np.arange(chunk_start, chunk_stop, dtype=np_hsize_t)

        super().__init__(chunk_indices, dset_size, chunk_size)

    @cython.ccall
    def chunk_submap(self, chunk_idx: hsize_t) -> tuple[Slice, slice, slice]:
        chunk_start, chunk_stop = self._chunk_start_stop(chunk_idx)
        sel_start = self.start
        sel_stop = self.stop
        sel_step = self.step

        abs_start = max(chunk_start, sel_start)
        # Get the smallest lcm multiple of common that is >= start
        abs_start = smallest_step_after(abs_start, sel_start % sel_step, sel_step)

        # shift start so that it is relative to index
        value_sub_start = (abs_start - sel_start) // sel_step
        value_sub_stop = ceil_a_over_b(min(chunk_stop, sel_stop) - sel_start, sel_step)

        chunk_sub_start = abs_start - chunk_start
        count = value_sub_stop - value_sub_start
        chunk_sub_stop = count2stop(chunk_sub_start, count, sel_step)

        return (
            Slice(chunk_start, chunk_stop, 1),
            slice(value_sub_start, value_sub_stop, 1),
            slice(chunk_sub_start, chunk_sub_stop, sel_step),
        )

    @cython.ccall
    def chunks_indexer(self):  # -> slice | NDArray[np_hsize_t]:
        indices = self.chunk_indices
        if self.step > self.chunk_size:
            return _maybe_array_idx_to_slice(np.asarray(indices))
        else:
            idx_len = len(indices)
            assert idx_len > 0  # Empty slices are filtered out by index_chunk_mappers()
            chunk_start = indices[0]
            chunk_stop = indices[idx_len - 1] + 1
            return slice(int(chunk_start), int(chunk_stop), 1)

    @cython.ccall
    def whole_chunks_idxidx(self):
        indices = self.chunk_indices
        idx_len = len(indices)
        assert idx_len > 0  # Empty slices are filtered out by index_chunk_mappers()

        if self.chunk_size == 1:
            # All chunks are wholly selected
            return slice(0, idx_len, 1)

        if self.step > 1:
            if self.last_chunk_size == 1 and indices[idx_len - 1] == self.n_chunks - 1:
                # Last chunk contains exactly one point, so it's wholly covered.
                # For all other chunks, chunk_size > 1 and step > 1 so the selection
                # will never cover a whole chunk
                return slice(idx_len - 1, idx_len, 1)
            else:
                return slice(0, 0, 1)

        # step==1. The first and last chunk may be partially selected;
        # the rest are always wholly selected.
        idxidx_start = int(self.start % self.chunk_size != 0)
        idxidx_stop = idx_len
        if self.stop < self.dset_size and self.stop % self.chunk_size > 0:
            idxidx_stop -= 1
        return slice(idxidx_start, idxidx_stop, 1)


@cython.cclass
class IntegerMapper(IndexChunkMapper):
    """IndexChunkMapper for scalar integer indices"""

    idx: hsize_t = cython.declare("hsize_t", visibility="readonly")

    def __init__(self, idx: hsize_t, dset_size: hsize_t, chunk_size: hsize_t):
        assert 0 <= idx < dset_size
        self.idx = idx
        chunk_indices = np.array([idx // chunk_size], dtype=np_hsize_t)
        super().__init__(chunk_indices, dset_size, chunk_size)

    @cython.ccall
    def chunk_submap(self, chunk_idx: hsize_t) -> tuple[Slice, DropAxis, int]:
        chunk_start, chunk_stop = self._chunk_start_stop(chunk_idx)
        chunk_sub_idx = self.idx - chunk_start
        return Slice(chunk_start, chunk_stop, 1), DROP_AXIS, chunk_sub_idx

    @cython.ccall
    def chunks_indexer(self):
        # a[i] would change the shape
        # a[[i]] (the default without overriding this method) would return a copy
        # a[i:i+1] returns a view, which is faster
        i = int(self.chunk_indices[0])
        return slice(i, i + 1, 1)

    @cython.ccall
    def whole_chunks_idxidx(self):
        if self.chunk_size == 1:
            return slice(0, 1, 1)

        # If the index is in the last chunk and the last chunk is size 1, then it's
        # wholly selected. In any other case, it's partially selected.
        partial = self.dset_size % self.chunk_size
        if partial != 1:
            return slice(0, 0, 1)
        if self.chunk_indices[0] != self.dset_size // self.chunk_size:
            return slice(0, 0, 1)
        return slice(0, 1, 1)

    @property
    def value_view_idx(self) -> slice | None:
        return None


@cython.cclass
class IntegerArrayMapper(IndexChunkMapper):
    """IndexChunkMapper for one-dimensional fancy integer array indices.
    This is also used for boolean indices (preprocessed with np.flatnonzero()).
    """

    idx: NDArray[np_hsize_t] = cython.declare(object, visibility="readonly")
    is_ascending: bint = cython.declare(bint, visibility="readonly")

    def __init__(
        self,
        idx: NDArray[np_hsize_t],
        dset_size: hsize_t,
        chunk_size: hsize_t,
    ):
        self.idx = idx

        self.is_ascending = False
        idx_v: hsize_t[:] = idx
        prev = idx_v[0]
        for i in range(1, len(idx_v)):
            if idx_v[i] < prev:
                break
            prev = idx_v[i]
        else:
            self.is_ascending = True

        chunk_indices = np.unique(idx // chunk_size)
        super().__init__(chunk_indices, dset_size, chunk_size)

    @cython.ccall
    def chunk_submap(
        self, chunk_idx: hsize_t
    ) -> tuple[Slice, NDArray[np_hsize_t] | slice, NDArray[np_hsize_t] | slice]:
        chunk_start, chunk_stop = self._chunk_start_stop(chunk_idx)

        if self.is_ascending:
            # O(n*logn)
            start_idx, stop_idx = np.searchsorted(self.idx, [chunk_start, chunk_stop])
            mask = slice(int(start_idx), int(stop_idx), 1)
        # TODO optimize monotonic descending
        else:
            # O(n^2)
            mask = (chunk_start <= self.idx) & (self.idx < chunk_stop)
            mask = _maybe_array_idx_to_slice(mask)

        return (
            Slice(chunk_start, chunk_stop, 1),
            mask,
            _maybe_array_idx_to_slice(self.idx[mask] - chunk_start),
        )

    @cython.cfunc
    def _chunk_sizes_in_chunk_indices(self):  # -> NDArray[np_hsize_t] | int:
        """Return the number of points taken from each chunk within self.chunk_indices.
        All but the last chunk always contain self.chunk_size points.
        """
        indices = self.chunk_indices
        idx_len = len(indices)
        assert idx_len > 0  # Empty slices are filtered out by index_chunk_mappers()

        if indices[idx_len - 1] < self.dset_size // self.chunk_size:
            # Not the last chunk, or last chunk but exactly divisible by chunk_size
            return self.chunk_size

        out = np.full_like(indices, fill_value=self.chunk_size)
        out[idx_len - 1] = self.last_chunk_size
        return out

    @cython.ccall
    def chunks_indexer(self):
        indices = np.asarray(self.chunk_indices)
        return _maybe_array_idx_to_slice(indices)

    @cython.ccall
    def whole_chunks_idxidx(self):
        # Don't double count when the same index is picked twice
        idx_unique = np.unique(self.idx)
        _, counts = np.unique(idx_unique // self.chunk_size, return_counts=True)
        return _maybe_array_idx_to_slice(counts == self._chunk_sizes_in_chunk_indices())


def index_chunk_mappers(
    idx: Any,
    shape: tuple[int, ...],
    chunk_size: tuple[int, ...] | ChunkSize,
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

    d: hsize_t
    n: hsize_t
    mappers = []

    # Process the prefix of the axes which idx selects on
    for i, d, n in zip(idx.args, shape[:idx_len], chunk_size[:idx_len]):
        i = i.reduce((d,))
        mappers.append(_index_to_mapper(i.raw, d, n))

    # Handle the remaining suffix axes on which we did not select, we still need to
    # break them up into chunks.
    for d, n in zip(shape[idx_len:], chunk_size[idx_len:]):
        mappers.append(SliceMapper(slice(0, d, 1), d, n))

    return idx, mappers


@cython.cfunc
def _index_to_mapper(idx, dset_size: hsize_t, chunk_size: hsize_t) -> IndexChunkMapper:
    """Convert a one-dimensional index, preprocessed by ndindex, to a mapper"""
    if isinstance(idx, int):
        return IntegerMapper(idx, dset_size, chunk_size)

    if isinstance(idx, np.ndarray):
        if idx.ndim != 1:
            raise NotImplementedError("array index must be 1-dimensional")
        idx = _maybe_array_idx_to_slice(idx)
        if isinstance(idx, np.ndarray):
            return IntegerArrayMapper(idx, dset_size, chunk_size)

    if isinstance(idx, slice):
        return SliceMapper(idx, dset_size, chunk_size)

    raise NotImplementedError(f"index type {type(idx)} not supported")


@cython.cfunc
def _maybe_array_idx_to_slice(idx: Any):  # -> NDArray[np_hsize_t] | slice:
    """Attempt to convert an integer or boolean array index to a slice"""
    # boolean array indices can be trivially expressed as an integer array
    if idx.dtype == bool:
        idx = np.flatnonzero(idx)

    # Don't copy when converting from np.intp to uint64 on 64-bit platforms
    idx = asarray(idx, np_hsize_t)

    if not idx.flags.writeable:
        # Cython doesn't support read-only views in pure Python mode,
        # so we're forced to copy to prevent a crash in the next line.
        idx = idx.copy()

    idx_v: hsize_t[:] = idx
    idx_len = len(idx_v)
    if idx_len == 0:
        return slice(0, 0, 1)

    start = idx_v[0]
    if idx_len == 1:
        return slice(int(start), int(start + 1), 1)
    if idx_v[1] <= start:  # step <1
        return idx
    stop = idx_v[idx_len - 1] + 1
    step = idx_v[1] - start
    if idx_len == 2:
        return slice(int(start), int(stop), int(step))

    if stop2count(start, stop, step) == idx_len:
        j = start + step + step
        for i in range(2, idx_len):
            if idx_v[i] != j:
                break
            j += step
        else:
            return slice(int(start), int(stop), int(step))

    return idx


def as_subchunk_map(
    idx: Any,
    shape: tuple[int, ...],
    chunk_size: tuple[int, ...] | ChunkSize,
) -> Iterator[
    tuple[
        Tuple,
        AnySlicerND,
        AnySlicerND,
    ]
]:
    """Computes the chunk selection assignment. In particular, given a `chunk_size`
    it returns triple (chunk_idx, value_sub_idx, chunk_sub_idx) such that for a
    chunked Dataset `ds` we can translate selections like

    >> value = ds[idx]

    into selecting from the individual chunks of `ds` as

    >> value = np.empty(output_shape)
    >> for chunk_idx, value_sub_idx, chunk_sub_idx in as_subchunk_map(
    ..     idx, ds.shape, ds.chunk_size
    .. ):
    ..     value[value_sub_idx] = ds.data_dict[chunk_idx][chunk_sub_idx]

    Similarly, assignments like

    >> ds[idx] = value

    can be translated into

    >> for chunk_idx, value_sub_idx, chunk_sub_idx in as_subchunk_map(
    ..     idx, ds.shape, ds.chunk_size
    .. ):
    ..     ds.data_dict[chunk_idx][chunk_sub_idx] = value[value_sub_idx]

    :param idx: the "index" to read from / write to the Dataset
    :param shape: the shape of the Dataset
    :param chunk_size: the `ChunkSize` of the Dataset
    :return: a generator of `(chunk_idx, value_sub_idx, chunk_sub_idx)` tuples
    """
    idx, mappers = index_chunk_mappers(idx, shape, chunk_size)
    if not mappers:
        return
    idx_len = len(idx.args)

    mapper: IndexChunkMapper  # noqa: F842
    chunk_subindexes = [
        [mapper.chunk_submap(chunk_idx) for chunk_idx in mapper.chunk_indices]
        for mapper in mappers
    ]

    # Now combine the chunk_slices and subindexes for each dimension into tuples
    # across all dimensions.
    for p in itertools.product(*chunk_subindexes):
        chunk_slices, value_subidxs, chunk_subidxs = zip(*p)

        # skip dimensions which were sliced away
        value_subidxs = tuple(
            value_subidx
            for value_subidx in value_subidxs
            if value_subidx is not DROP_AXIS
        )
        # skip suffix dimensions
        chunk_subidxs = chunk_subidxs[:idx_len]

        yield Tuple(*chunk_slices), value_subidxs, chunk_subidxs
