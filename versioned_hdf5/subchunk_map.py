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

# Use same data type for indexing as in libhdf5 C.
# This matters on 32-bit platforms, where ssize_t is 32 bit and would be incapable of
# indexing hdf5 datasets on disk wider than 2**31
from cython import bint, ssize_t
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
    def read_many_slices_params(
        self,
    ) -> tuple[NDArray[np_hsize_t], NDArray[np_hsize_t] | None]:
        """Return the parameters for read_many_slices() for all chunks of the selection.

        Returns
        -------
        Tuple of two 2D arrays, (slices, chunk_to_slices), or (slices, None).
        The presence of chunk_to_slices changes between basic and fancy indexes.

        **Basic indexing**

        For basic indexing and in simple secial cases of fancy indexes, there is a
        1:1 correlation between selected chunks and slices to tranfer. In this

        len(slices) == len(self.chunk_indices).
        chunk_to_slices is None

        Each row represents a slice to transfer the data for the matching chunk on
        chunk_indices.

        To retrieve the absolute chunk index and the slice for the i-th chunk of the
        selection:

        >> chunk_idx = mapper.chunk_indices[i]
        >> slices, _ = mapper.read_many_slices_params()
        >> slice_i = slices[i, :]


        **Fancy indexing**

        Each chunk is transferred by 1 or more slices (1:N).

        len(slices) > len(self.chunk_indices)
        len(chunk_to_slices) == len(self.chunk_indices)

        chunk_to_slices is a 1D array with as many points as chunk_indices plus one,
        with each point representing the start of the slices view for that chunk. The
        slices for a chunk end where the slices for the next chunk begin.

        The first and last point are there just for convenience; this is always true:
            chunk_to_slices[0] == 0
            chunk_to_slices[-1] == len(slices)

        To retrieve the absolute chunk index and the slices for the i-th chunk of the
        selection:

        >> chunk_idx = mapper.chunk_indices[i]
        >> slices, chunk_to_slices = mapper.read_many_slices_params()
        >> slices_i = slices[chunk_to_slices[i]:chunk_to_slices[i+1]]

        Note: in any n-dimensional index there is always at most one fancy index.


        **The slices array**

        The slices array features one row per slice and exactly five columns:

        chunk_sub_start
            The start of the slice, relative to the first point of the chunk
        value_sub_start
            The start of the slice, relative to the whole __getitem__ return value or
            the whole __setitem__ value parameter. 0 for scalar indices.
        count
            The number of points selected by the slice
        chunk_sub_stride
            How many points to skip within the chunks between each selected point,
            a.k.a. step
        value_sub_stride
            How many points to skip within the __getitem__ return value or the
            __setitem__ value parameter between each selected point.
            This is always 1 except for the case of out-of-order fancy indexes.

        See the enum subchunk_map.pxd::ReadManySlicesColumn.

        The mapping from the columns of the slices view to the parameters of
        read_many_slices() is as follows:

        __getitem__ (from slab to output value)
            src_start  = chunk_sub_start + (slab_offset if axis==0 else 0)
            dst_start  = value_sub_start
            count      = count
            src_stride = chunk_sub_stride
            dst_stride = value_sub_stride

        __setitem__ (from input value to slab)
            src_start  = value_sub_start
            dst_start  = chunk_sub_start + (slab_offset if axis==0 else 0)
            count      = count
            src_stride = value_sub_stride
            dst_stride = chunk_sub_stride

        slab-to-slab transfer
            src_start  = chunk_sub_start + (src_slab_offset if axis==0 else 0)
            dst_start  = chunk_sub_start + (dst_slab_offset if axis==0 else 0)
            count      = count
            src_stride = chunk_sub_stride
            dst_stride = chunk_sub_stride
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
    def whole_chunks_indexer(self):  # -> slice | NDArray[np_hsize_t]:
        """Return a subset of chunks_indexer that selects only the chunks where all
        points of the chunk are included in the selection.

        e.g. if the index of this mapper is [True, True, False, False, True, False]
        and self.chunk_size=2, then:

        - self.chunks_indexer -> [0, 2]
        - self.whole_chunks_indexer -> [0]
        """

    @property
    def value_view_idx(self) -> slice | None:
        """Return an index, to be applied to the matching axis of the __getitem__
        return value or the __setitem__ value parameter, that will create a view with
        has the same dimensionality as the base array.

        This allows passing the view to read_many_slices().

        Example
        -------
        >>> idx = (slice(3, 5), 1, slice(None))
        >>> chunks = (4, 4, 4)
        >>> shape = (10, 10, 10)
        >>> ndindex_idx, mappers = index_chunk_mappers(idx, chunks, shape)
        >>> value = np.empty(ndindex_idx.newshape(shape))  # e.g. for __getitem__
        >>> value_view = value[tuple(mapper.value_view_idx for mapper in mappers)]
        >>> value.shape
        (2, 10)
        >>> value_view.shape
        (2, 1, 10)
        >>> value_view.base is value
        True
        """
        return slice(None)


@cython.cclass
class BasicChunkMapper(IndexChunkMapper):
    """Abstract IndexChunkMapper for numpy basic indexing (slices and integers)"""

    @cython.ccall
    def read_many_slices_params(self) -> tuple[NDArray[np_hsize_t], None]:
        n_sel_chunks = len(self.chunk_indices)
        slices = np.empty((n_sel_chunks, 5), dtype=np_hsize_t)
        slices_v: hsize_t[:, :] = slices

        with cython.nogil:
            for idxidx in range(n_sel_chunks):
                chunk_idx = self.chunk_indices[idxidx]
                (
                    chunk_sub_start,
                    value_sub_start,
                    count,
                    chunk_sub_stride,
                ) = self._read_many_slices_param(chunk_idx)
                # For reference, see subchunk_map.pxd::ReadManySlicesColumn
                slices_v[idxidx, 0] = chunk_sub_start
                slices_v[idxidx, 1] = value_sub_start
                slices_v[idxidx, 2] = count
                slices_v[idxidx, 3] = chunk_sub_stride
                slices_v[idxidx, 4] = 1  # value_sub_stride

        return slices, None

    @cython.cfunc
    @cython.nogil
    @cython.exceptval(check=False)
    @abc.abstractmethod
    def _read_many_slices_param(
        self,
        chunk_idx: hsize_t,
    ) -> tuple[hsize_t, hsize_t, hsize_t, hsize_t]:
        """Return the parameters to read_many_slices() for a single chunk.

        Returns tuple of:
        - chunk_sub_start
        - value_sub_start
        - count
        - chunk_sub_stride
        """


@cython.cclass
class SliceMapper(BasicChunkMapper):
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

    @cython.cfunc
    @cython.nogil
    @cython.exceptval(check=False)
    def _read_many_slices_param(
        self,
        chunk_idx: hsize_t,
    ) -> tuple[hsize_t, hsize_t, hsize_t, hsize_t]:
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

        count = value_sub_stop - value_sub_start
        chunk_sub_start = abs_start - chunk_start
        chunk_sub_stride = sel_step

        return chunk_sub_start, value_sub_start, count, chunk_sub_stride

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
    def whole_chunks_indexer(self):  # -> slice | NDArray[np_hsize_t]:
        if self.chunk_size == 1:
            # All chunks are wholly selected
            return self.chunks_indexer()

        indices = self.chunk_indices
        idx_len = len(indices)
        assert idx_len > 0  # Empty slices are filtered out by index_chunk_mappers()
        last_idx = indices[idx_len - 1]

        if self.step > 1:
            if self.last_chunk_size == 1 and last_idx == self.n_chunks - 1:
                # Last chunk contains exactly one point, so it's wholly covered.
                # For all other chunks, chunk_size > 1 and step > 1 so the selection
                # will never cover a whole chunk
                return slice(int(last_idx), self.n_chunks, 1)
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
class IntegerMapper(BasicChunkMapper):
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

    @cython.cfunc
    @cython.nogil
    @cython.exceptval(check=False)
    def _read_many_slices_param(
        self,
        chunk_idx: hsize_t,
    ) -> tuple[hsize_t, hsize_t, hsize_t, hsize_t]:
        chunk_start = chunk_idx * self.chunk_size
        return (
            self.idx - chunk_start,  # chunk_sub_start
            0,  # value_sub_start
            1,  # count
            1,  # chunk_sub_stride
        )

    @cython.ccall
    def chunks_indexer(self):
        # a[i] would change the shape
        # a[[i]] (the default without overriding this method) would return a copy
        # a[i:i+1] returns a view, which is faster
        i = int(self.chunk_indices[0])
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

    @property
    def value_view_idx(self) -> slice | None:
        return None


@cython.cclass
class EverythingMapper(BasicChunkMapper):
    """Select all points along an axis [:].

    This is functionally identical to SliceMapper(slice(None), chunk_size, dset_size),
    special-cased here for simplicity and speed.
    """

    def __init__(self, dset_size: hsize_t, chunk_size: hsize_t):
        n_chunks = ceil_a_over_b(dset_size, chunk_size)
        super().__init__(np.arange(n_chunks, dtype=np_hsize_t), dset_size, chunk_size)

    @cython.ccall
    def chunk_submap(self, chunk_idx: hsize_t) -> tuple[Slice, slice, slice]:
        chunk_start, chunk_stop = self._chunk_start_stop(chunk_idx)
        return (
            Slice(chunk_start, chunk_stop, 1),
            slice(chunk_start, chunk_stop, 1),
            slice(0, chunk_stop - chunk_start, 1),
        )

    @cython.cfunc
    @cython.nogil
    @cython.exceptval(check=False)
    def _read_many_slices_param(
        self, chunk_idx: hsize_t
    ) -> tuple[hsize_t, hsize_t, hsize_t, hsize_t]:
        chunk_start, chunk_stop = self._chunk_start_stop(chunk_idx)

        return (
            0,  # chunk_sub_start
            chunk_start,  # value_sub_start
            chunk_stop - chunk_start,  # count
            1,  # chunk_sub_stride
        )

    @cython.ccall
    def chunks_indexer(self):
        return slice(0, self.n_chunks, 1)

    @cython.ccall
    def whole_chunks_indexer(self):
        return slice(0, self.n_chunks, 1)


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

    @cython.ccall
    def read_many_slices_params(
        self,
    ) -> tuple[NDArray[np_hsize_t], NDArray[np_hsize_t] | None]:
        n_sel_chunks = len(self.chunk_indices)
        max_rows = len(self.idx)

        slices = np.empty((max_rows, 5), dtype=np_hsize_t)
        slices_v: hsize_t[:, :] = slices

        assert max_rows >= n_sel_chunks
        if max_rows > n_sel_chunks:
            chunks_to_slices = np.empty(n_sel_chunks + 1, dtype=np_hsize_t)
            chunk_to_slices_v: hsize_t[:] = chunks_to_slices
        else:
            # Because we know we are selecting at least one point per chunk of
            # chunk_indices, then if there are exactly as many chunks as there are
            # points we know that each point falls in a separate chunk and there will be
            # exactly one slice(idx[i], idx[i] + 1, 1) per chunk, so there is no need
            # for the 1:N mapping table chunk_to_slices.
            chunks_to_slices = None

        if self.is_ascending:
            starts_stops = np.empty((n_sel_chunks, 2), dtype=np_hsize_t)
            starts_stops_v: hsize_t[:, :] = starts_stops
            for chunk_idxidx in range(n_sel_chunks):
                chunk_idx = self.chunk_indices[chunk_idxidx]
                chunk_start, chunk_stop = self._chunk_start_stop(chunk_idx)
                starts_stops_v[chunk_idxidx, 0] = chunk_start
                starts_stops_v[chunk_idxidx, 1] = chunk_stop

            # This should be faster than calling searchsorted n_sel_chunks times
            # TODO test performance and consider rewriting this in cython, with the
            # extra assumption that starts_stops is sorted,  which allows reducing
            # big-O complexity from O(n*logn) to O(n).
            starts_stops_idxidx: ssize_t[:, :] = np.searchsorted(self.idx, starts_stops)
            idx_v: hsize_t[:] = self.idx
            value_full_idx_v: hsize_t[:] = np.arange(max_rows, dtype=np_hsize_t)

        value_sub_idx: hsize_t[:]
        chunk_sub_idx: hsize_t[:]

        slices_cursor = 0
        for chunk_idxidx in range(n_sel_chunks):
            chunk_idx = self.chunk_indices[chunk_idxidx]
            chunk_start, chunk_stop = self._chunk_start_stop(chunk_idx)

            if self.is_ascending:
                # O(n), plus O(n ~ n*logn) for the searchsorted above
                start_idx = starts_stops_idxidx[chunk_idxidx, 0]
                stop_idx = starts_stops_idxidx[chunk_idxidx, 1]
                value_sub_idx = value_full_idx_v[start_idx:stop_idx]
                chunk_sub_idx = idx_v[start_idx:stop_idx]
            # TODO optimize monotonic descending
            else:
                # O(n^2)
                mask = (chunk_start <= self.idx) & (self.idx < chunk_stop)
                # Don't copy when converting from np.intp to uint64 on 64-bit platforms
                value_sub_idx = asarray(np.flatnonzero(mask), np_hsize_t)
                chunk_sub_idx = self.idx[value_sub_idx]

            if max_rows > n_sel_chunks:
                chunk_to_slices_v[chunk_idxidx] = slices_cursor

            next_slices_cursor = _fancy_idx_to_slices(
                value_sub_idx, chunk_sub_idx, slices_v, slices_cursor
            )
            for i in range(slices_cursor, next_slices_cursor):
                slices_v[i, 0] -= chunk_start
            slices_cursor = next_slices_cursor

        if max_rows > n_sel_chunks:
            if slices_cursor == n_sel_chunks:
                # e.g. idx=[0, 2, 3], chunk_size=2, dset_size=4
                chunks_to_slices = None
            else:
                chunk_to_slices_v[n_sel_chunks] = slices_cursor
            # This is a view that wastes some memory. No point calling realloc() as this
            # object is volatile and will be dereferenced at the end of the current
            # __getitem__ / __setitem__ call.
            slices = slices[:slices_cursor]
        else:
            assert slices_cursor == n_sel_chunks
        return slices, chunks_to_slices

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
    def whole_chunks_indexer(self):
        # Don't double count when the same index is picked twice
        idx_unique = np.unique(self.idx)
        _, counts = np.unique(idx_unique // self.chunk_size, return_counts=True)
        indices = np.asarray(self.chunk_indices)
        indices = indices[counts == self._chunk_sizes_in_chunk_indices()]
        return _maybe_array_idx_to_slice(indices)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def _fancy_idx_to_slices(
    value_sub_idx: hsize_t[:],
    chunk_sub_idx: hsize_t[:],
    slices: hsize_t[:, :],
    cursor: hsize_t,
) -> hsize_t:
    """Break down a one-dimensional integer array into a (greedy) minimal list of
    slices. Populate input of read_many_slices().

    Parameters
    ----------
    value_sub_idx:
        integer array index selecting the __getitem__ output value or __setitem__
        parameter value for the chunk
    chunk_sub_idx:
        integer array index selecting the points within the chunk
    slices:
        Return value of IndexChunkMapper._init_many_slices_params()
    cursor:
        Current row of slices

    Returns
    -------
    Next row of slices
    """
    npoints = len(value_sub_idx)

    i = 0
    while i < npoints:
        value_sub_start = value_sub_idx[i]
        chunk_sub_start = chunk_sub_idx[i]
        count = 1
        value_sub_stride = 1  # This is 1 for monotonically increasing indices
        chunk_sub_stride = 1

        while True:
            if i == npoints - 1:
                break

            v0 = value_sub_idx[i]
            v1 = value_sub_idx[i + 1]
            c0 = chunk_sub_idx[i]
            c1 = chunk_sub_idx[i + 1]

            if v1 <= v0 or c1 <= c0:
                break  # step < 1

            if count == 1:
                # Recognize a stride pattern
                value_sub_stride = v1 - v0
                chunk_sub_stride = c1 - c0

                # Handle special case where having stride>1 now would break
                # a contiguous block with stride=1 immediately afterwards
                if i < npoints - 2:
                    v2 = value_sub_idx[i + 2]
                    if value_sub_stride > 1 and v2 == v1 + 1:
                        value_sub_stride = 1
                        chunk_sub_stride = 1
                        break

                    c2 = chunk_sub_idx[i + 2]
                    if chunk_sub_stride > 1 and c2 == c1 + 1:
                        value_sub_stride = 1
                        chunk_sub_stride = 1
                        break

            else:
                if v1 - v0 != value_sub_stride:
                    break
                if c1 - c0 != chunk_sub_stride:
                    break

            count += 1
            i += 1
        i += 1

        # For reference, see subchunk_map.pxd::ReadManySlicesColumn
        slices[cursor, 0] = chunk_sub_start
        slices[cursor, 1] = value_sub_start
        slices[cursor, 2] = count
        slices[cursor, 3] = chunk_sub_stride
        slices[cursor, 4] = value_sub_stride
        cursor += 1

    return cursor


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
        mappers.append(EverythingMapper(d, n))

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
        if idx == slice(0, dset_size, 1):
            return EverythingMapper(dset_size, chunk_size)
        else:
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


class TransferType(enum.Enum):
    getitem = 0
    setitem = 1
    slab_to_slab = 2


@cython.ccall
def read_many_slices_params_nd(
    transfer_type: TransferType,
    mappers: list[IndexChunkMapper],
    chunk_idxidx: hsize_t[:, :],
    src_slab_offsets: hsize_t[:],
    dst_slab_offsets: hsize_t[:],
) -> hsize_t[:, :, :]:
    """Assemble the parameters for read_many_slices() for a multi-dimensional index
    for a specific transfer type between two arrays.

    Parameters
    ----------
    transfer_type:
        The type of transfer to perform:

        TransferType.getitem
            The src array of read_many_slices() is an individual slab
            The dst array is the return value of __getitem__
            src_slab_offsets must have as many rows as chunk_idxidx
        TransferType.setitem
            The src array is the value parameter of __setitem__
            The dst array is an individual slab
            dst_slab_offsets must have as many rows as chunk_idxidx
        TransferType.slab_to_slab
            Both src and dst arrays are slabs

    mappers:
        List of IndexChunkMapper objects, one per axis of the index, as returned by
        index_chunk_mappers()
    chunk_idxidx:
        2D array of shape (nchunks, ndim) where each row represents a chunk to be
        transferred and point contains the index of IndexChunkMapper.chunk_indices for
        the axis matching the column. In other words,

            mappers[j].chunk_indices[chunk_idxidx[i, j]] * mappers[j].chunk_size

        is the address along axis j of the top-left corner in the virtual array
        of the i-th chunk to be transferred.

    src_slab_offsets:
        1D array of shape (nchunks, ) with the offset of the slab along axis 0, to be
        added to addresses relative to the start of the chunk in the src slab.
        Ignored when transfer_type==TransferType.setitem.
    dst_slab_offsets:
        1D array of shape (nchunks, ) with the offset of the slab along axis 0, to be
        added to addresses relative to the start of the chunk in the dst slab.
        Ignored when transfer_type==TransferType.getitem.

    Returns
    -------
    3D array of indices of shape (nslices, 5, ndim), ready to be passed to
    read_many_slices().

    There is one row per slice.
    In absence of fancy indexing, this is the same as the number of chunks.

    Axis 1 contains exactly 5 columns:

    - src_start
    - dst_start
    - count
    - src_stride
    - dst_stride

    See slicetools.pyx::read_many_slices() for documentation on these parameters.
    See subchunk_map.pxd::ReadManySlicesNDColumn for reference on the column indices.

    The final column represents the axis. Note how axis must be last, and thus
    internally C-contiguous, in order to allow read_many_slices() to use the index as-is
    without any further processing.
    """
    # Hack around lack of cython enums in pure python mode
    transfer_type_int: cython.int = transfer_type.value
    TRANSFER_TYPE_GETITEM: cython.int = TransferType.getitem.value
    TRANSFER_TYPE_SETITEM: cython.int = TransferType.setitem.value

    nchunks = chunk_idxidx.shape[0]
    ndim = chunk_idxidx.shape[1]

    # Store the slices in a C array (in the stack) of views (in the heap).
    # In order to keep the reference counter of the views, which are Python objects,
    # above zero for the duration of the function, we also need to reference them from a
    # Python object.
    slices_refs = []
    # Can't define module-level constants in pure-python mode
    # https://github.com/numpy/numpy/blob/b0a52b76c2bebe5c237722589d13bf02affa9c43/numpy/core/include/numpy/ndarraytypes.h#L30
    # NPY_MAXDIMS = 32
    slices: hsize_t[:, :][32]  # type: ignore
    if not cython.compiled:
        slices = [None] * ndim

    # Generate the slices along each axis and calculate the flattened number of slices
    # to return. This is trivial in absence of a fancy index as there is a 1:1
    # relationship between chunks and slices.
    nslices: ssize_t = nchunks
    fancy_axis = -1
    for j in range(ndim):
        mapper: IndexChunkMapper = mappers[j]
        slices_j: hsize_t[:, :]
        slices_j, c2s = mapper.read_many_slices_params()
        slices[j] = slices_j
        slices_refs.append(slices_j)

        assert slices_j.shape[1] == 5
        if c2s is None:
            # Basic index, or fancy index that could be coerced into 1 slice per chunk
            assert slices_j.shape[0] == len(mapper.chunk_indices)
        else:
            # There is a fancy index on this axis. Note that fancy indices can return
            # c2s=None and effectively declass themselves to basic indices for the
            # purpose of this algorithm when they can be represented as exactly one
            # slice per chunk; e.g. idx=[0, 2, 3], chunk_size=2, dset_size=4.

            assert fancy_axis == -1  # There can be at most one fancy index
            fancy_axis = j
            chunks_to_slices: hsize_t[:] = c2s
            assert len(chunks_to_slices) == len(mapper.chunk_indices) + 1

            # Recalculate nslices
            nslices = 0
            for i in range(nchunks):
                k = chunk_idxidx[i, j]
                if not cython.compiled:
                    k = int(k)  # uint64_t + int -> float64
                fancy_count = chunks_to_slices[k + 1] - chunks_to_slices[k]
                if not cython.compiled:
                    fancy_count = int(fancy_count)  # uint64_t + int -> float64
                nslices += fancy_count

    # Allocate memory for return value
    ndslices: hsize_t[:, :, :] = np.empty((nslices, 5, ndim), dtype=np_hsize_t)

    # Perform pseudo-cartesian product of mapper.read_many_slices_params()[0]
    # for each axis. Unlike a true cartesian product, chunk_idxidx can skip chunks.
    #
    # A helpful legend to this seven-counters monstruosity,
    # with many apologies to the reader:
    #
    # i: row of chunks; the flat i-th chunk caught not only by the intersection of
    #    mappers[*].chunk_indices, but also by the per-chunk mask on slab_indices.
    # j: axis; column of chunks
    # k: the index on mappers[j].chunk_indices.
    #    mappers[j].chunk_indices[k] * chunk_size[j] is the address of the
    #    top-left corner of the chunk along axis j in the virtual dataset
    #
    # Additional counters only for when there's a fancy index, which causes a 1:N
    # relationship between chunks and slices:
    # fancy_k:   like k, but always pinned to the fancy index regardless of the
    #            current axis
    # fancy_cur: the current slice along the fancy index, within the current chunk k
    # outs:      the n-th n-dimensional slice to process with read_many_slices;
    #            cursor on axis 0 of the return value
    # inps:      the n-th 1-dimensional slice on slices[j].
    #            When j == fancy_axis, this equals to fans; on all other axes, this is a
    #            repetition of k for as many slices there are in the fancy index for the
    #            current chunk.
    with cython.nogil:
        for j in range(ndim):
            slices_j = slices[j]
            # For the mapping, refer to docstring of
            # IndexChunkMapper.read_many_slices_params()
            # For column indices, refer to subchunk_map.pxd::ReadManySlicesColumn
            if transfer_type_int == TRANSFER_TYPE_GETITEM:
                src_start = slices_j[:, 0]  # chunk_sub_start
                dst_start = slices_j[:, 1]  # value_sub_start
                count = slices_j[:, 2]  # count
                src_stride = slices_j[:, 3]  # chunk_sub_stride
                dst_stride = slices_j[:, 4]  # value_sub_stride
                add_src_offset: bint = j == 0
                add_dst_offset: bint = False

            elif transfer_type_int == TRANSFER_TYPE_SETITEM:
                src_start = slices_j[:, 1]  # value_sub_start
                dst_start = slices_j[:, 0]  # chunk_sub_start
                count = slices_j[:, 2]  # count
                src_stride = slices_j[:, 4]  # value_sub_stride
                dst_stride = slices_j[:, 3]  # chunk_sub_stride
                add_src_offset = False
                add_dst_offset = j == 0

            else:  # slab-to-slab
                src_start = slices_j[:, 0]  # chunk_sub_start
                dst_start = slices_j[:, 0]  # chunk_sub_start
                count = slices_j[:, 2]  # count
                src_stride = slices_j[:, 3]  # chunk_sub_stride
                dst_stride = slices_j[:, 3]  # chunk_sub_stride
                add_src_offset = j == 0
                add_dst_offset = j == 0

            assert not add_src_offset or len(src_slab_offsets) == nchunks
            assert not add_dst_offset or len(dst_slab_offsets) == nchunks

            if fancy_axis == -1:
                # 1 chunk : 1 slice
                for i in range(nchunks):
                    k = chunk_idxidx[i, j]
                    src_offset = src_slab_offsets[i] if add_src_offset else 0
                    dst_offset = dst_slab_offsets[i] if add_dst_offset else 0

                    # For reference, see subchunk_map.pxd::ReadManySlicesNDColumn
                    ndslices[i, 0, j] = src_start[k] + src_offset
                    ndslices[i, 1, j] = dst_start[k] + dst_offset
                    ndslices[i, 2, j] = count[k]
                    ndslices[i, 3, j] = src_stride[k]
                    ndslices[i, 4, j] = dst_stride[k]

            else:
                # 1 chunk : n slices for the fancy index
                # 1 chunk : 1 slice on other axes, which must be replicated
                # to match the fancy index
                outs: ssize_t = 0
                for i in range(nchunks):
                    if j != fancy_axis:
                        k = chunk_idxidx[i, j]
                    src_offset = src_slab_offsets[i] if add_src_offset else 0
                    dst_offset = dst_slab_offsets[i] if add_dst_offset else 0

                    fancy_k = chunk_idxidx[i, fancy_axis]
                    if not cython.compiled:
                        fancy_k = int(fancy_k)  # uint64_t + int -> float64
                    fancy_start = chunks_to_slices[fancy_k]
                    fancy_stop = chunks_to_slices[fancy_k + 1]
                    if not cython.compiled:
                        fancy_start = int(fancy_start)
                        fancy_stop = int(fancy_stop)

                    for fancy_cur in range(fancy_start, fancy_stop):
                        inps = fancy_cur if j == fancy_axis else k
                        # For reference, see subchunk_map.pxd::ReadManySlicesNDColumn
                        ndslices[outs, 0, j] = src_start[inps] + src_offset
                        ndslices[outs, 1, j] = dst_start[inps] + dst_offset
                        ndslices[outs, 2, j] = count[inps]
                        ndslices[outs, 3, j] = src_stride[inps]
                        ndslices[outs, 4, j] = dst_stride[inps]
                        outs += 1

                assert outs == nslices

    return ndslices


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
