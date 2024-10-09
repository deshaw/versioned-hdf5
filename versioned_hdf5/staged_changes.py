# Note: this entire module is compiled by cython with wraparound=False
# See meson.build for details

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import cython
import numpy as np
from cython import Py_ssize_t, bint
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .hyperspace import empty_view, fill_hyperspace, view_from_tuple
from .subchunk_map import (
    ChunkMapIndexError,
    IndexChunkMapper,
    cartesian_product,
    ceil_a_over_b,
    index_chunk_mappers,
    zip_chunk_submap,
    zip_slab_submap,
)

if TYPE_CHECKING:
    from .subchunk_map import AnySlicerND

if cython.compiled:
    from cython.cimports.versioned_hdf5.hyperspace import (  # type: ignore
        empty_view,
        fill_hyperspace,
        view_from_tuple,
    )
    from cython.cimports.versioned_hdf5.subchunk_map import (  # type: ignore
        IndexChunkMapper,
        cartesian_product,
        ceil_a_over_b,
        zip_chunk_submap,
        zip_slab_submap,
    )

T = TypeVar("T", bound=np.generic)


class StagedChangesArray(Generic[T]):
    """A basic array-like that wraps around an underlying read-only array-like.
    All changes to the data or the shape are stored in memory in chunks.

    **Performance assumptions**

    - Reading a whole chunk from the underlying data is not more expensive
      than reading individual elements from the chunk
    - Reading multiple contiguous chunks with a single getitem call is
      cheaper than cycling through the chunks and reading them individually
    - Reading two chunks that are contiguous on the innermost axis is typically
      cheaper than reading two chunks that are contiguous on an outer axis

    High level documentation on how the class works internally: :doc:`staged_changes`.
    """

    #: __getitem__ bound method of underlying array.
    #: Must support numpy fancy indexing.
    #:
    #: Note
    #: ----
    #: Returned arrays can be written back to unless they are views of another array.
    #: Set the writeable flag to False before returning them to prevent this.
    base_getitem: Callable[[Any], NDArray[T]]

    #: shape of the base array, e.g. before any calls to resize()
    base_shape: tuple[int, ...]

    #: current shape of the StagedChangesArray, e.g. downstream of resize()
    shape: tuple[int, ...]

    #: size of the tiles that will be modified at once. A write to
    #: less than a whole chunk will cause the remainder of the chunk
    #: to be read from the underlying array.
    chunk_size: tuple[int, ...]

    #: fill value for the array. This is a zero-dimensional array.
    fill_value: NDArray[T]

    #: array with same number of dimesions as base, containing 1 point per chunk.
    #: 0 = not modified; -1 = fill_value; 1+ = modified and stored at the matching
    #: index in chunk_values
    _chunk_states: NDArray[np.intp] | None

    #: Modified chunks, as indexed by chunk_states. Index 0 is unused and always None.
    #: Other elements can be None if the chunk has been removed after a resize().
    chunk_values: list[NDArray[T] | None]

    #: Flag that indicates that the array has been modified.
    _has_changes: bool

    #: Flag that indicates that base_getitem performs a transformation of the data
    #: in the base array, so all chunks will be returned by changes() even if they
    #: are marked as not modified in chunk_states.
    _base_getitem_transform: bool

    __slots__ = tuple(__annotations__)

    def __init__(
        self,
        base_getitem: Callable[[Any], NDArray[T]],
        shape: tuple[int, ...],
        chunk_size: tuple[int, ...],
        dtype: DTypeLike | None = None,
        fill_value: Any | None = None,
    ):
        if any(s < 0 for s in shape):
            raise ValueError("shape dimensions must be non-negative")
        if any(c <= 0 for c in chunk_size):
            raise ValueError("chunk sizes must be strictly positive")

        self.base_getitem = base_getitem
        self.shape = shape
        self.base_shape = shape
        self.chunk_size = chunk_size
        self._chunk_states = None
        self.chunk_values = [None]
        self._has_changes = False
        self._base_getitem_transform = False

        if fill_value is None:
            # Unlike 0.0, this works for weird dtypes such as np.void
            self.fill_value = np.zeros(1, dtype=dtype).reshape(())
        else:
            self.fill_value = np.asarray(fill_value, dtype)
            if self.fill_value.ndim != 0:
                raise ValueError("fill_value must be a scalar")

    @property
    def dtype(self) -> np.dtype[T]:
        return self.fill_value.dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        """Size in bytes of this array if it were completely loaded. Actual used memory
        can be less, as only modified chunks are stored in memory and even then they may
        be CoW copies of another array.
        """
        return self.size * self.fill_value.nbytes

    @property
    def nchunks(self) -> tuple[int, ...]:
        """Number of chunks on each axis"""
        return tuple(ceil_a_over_b(s, c) for s, c in zip(self.shape, self.chunk_size))

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self) -> NDArray[T]:
        return self[()]

    @property
    def has_changes(self) -> bool:
        """Return True if any chunks have been modified or if a lazy transformation
        took place; False otherwise.
        """
        return (
            self._has_changes
            or self._base_getitem_transform
            or self.shape != self.base_shape
        )

    @property
    def chunk_states(self) -> NDArray[np.intp]:
        """Lazily create chunk_states on first call to __setitem__ or resize()"""
        if self._chunk_states is None:
            self._chunk_states = np.zeros(self.nchunks, dtype=np.intp)
        return self._chunk_states

    def __repr__(self) -> str:
        return (
            f"StagedChangesArray<shape={self.shape}, chunk_size={self.chunk_size}, "
            f"dtype={self.dtype}, fill_value={self.fill_value.item()}, "
            f"{len(self.chunk_values) - 1} modified chunks>\n"
            f"chunk_states:\n{self.chunk_states}"
        )

    def _changes_plan(self, *, load_base: bool = False) -> ChangesPlan:
        return ChangesPlan(
            chunk_size=self.chunk_size,
            shape=self.shape,
            base_shape=self.base_shape,
            chunk_states=self.chunk_states,
            load_base=load_base or self._base_getitem_transform,
        )

    def _getitem_plan(self, idx: Any) -> GetItemPlan:
        return GetItemPlan(
            idx,
            chunk_size=self.chunk_size,
            shape=self.shape,
            chunk_states=self.chunk_states,
        )

    def _setitem_plan(self, idx: Any) -> SetItemPlan:
        return SetItemPlan(
            idx,
            chunk_size=self.chunk_size,
            shape=self.shape,
            chunk_states=self.chunk_states,
            chunk_values_len=len(self.chunk_values),
        )

    def _resize_plan(self, shape: tuple[int, ...]) -> ResizePlan:
        return ResizePlan(
            chunk_size=self.chunk_size,
            old_shape=self.shape,
            new_shape=shape,
            chunk_states=self.chunk_states,
            chunk_values_len=len(self.chunk_values),
        )

    def changes(
        self,
        *,
        load_base: bool = False,
    ) -> Iterator[tuple[tuple[slice, ...], NDArray[T] | None]]:
        """Yield all the changed chunks so far, as tuples of

        - slice index in the base array
        - chunk value, or None if the chunk has been removed after a resize()

        This lets you update the base array:

        >> for idx, value in staged_array.changes():
        ..     if value is not None:
        ..         base[idx] = value

        Note
        ----
        If a chunk is created full of the fill_value by resize() it will not be yielded
        by this method.

        Parameters
        ----------
        load_base: bool, optional
            If True, load all chunks from the base array, even if they haven't been
            modified.
        """
        if not self.has_changes and not load_base:
            return

        plan = self._changes_plan(load_base=load_base)
        if not load_base:
            assert not plan.loads

        for base_idx in plan.deleted:
            yield base_idx, None

        for base_idx, values_idx in plan.modified:
            chunk = self.chunk_values[values_idx]
            assert chunk is not None
            yield base_idx, chunk

        for input_idx, sub_indices, base_indices in plan.loads:
            # TODO in case of a change of dtype, this could potentially load the whole
            # base array into memory at once. We could break this down by defining a
            # maximum number of chunks to load at once.
            slab = self.base_getitem(input_idx)
            for sub_idx, base_idx in zip(sub_indices, base_indices):
                yield base_idx, slab[sub_idx]

    def __getitem__(self, idx: Any) -> NDArray[T]:
        """Get a slice of data from the array. This reads from the staged chunks
        in memory when available and from the base array otherwise.
        """
        if not self.has_changes and self.base_getitem is not DUMMY_GETITEM:
            return self.base_getitem(idx)

        plan = self._getitem_plan(idx)

        if not plan.modified and not plan.full and len(plan.base) == 1:
            # Either empty selection or the selection doesn't include any changed
            # chunks. We can skip a deep-copy.
            # It's important not to call self.base_getitem(idx), as the array might
            # have been shrunk but the idx may slice it up to the old shape.
            _, sub_idx = plan.base[0]
            return self.base_getitem(sub_idx)

        if len(plan.modified) == 1 and not plan.full and not plan.base:
            # Access a single chunk in memory. We can skip a deep-copy.
            values_idx, _, sub_idx = plan.modified[0]
            chunk = self.chunk_values[values_idx]
            assert chunk is not None
            return chunk[sub_idx]

        out = np.empty(plan.shape, dtype=self.dtype)

        for values_idx, out_idx, sub_idx in plan.modified:
            chunk = self.chunk_values[values_idx]
            assert chunk is not None
            out[out_idx] = chunk[sub_idx]

        for out_idx in plan.full:
            out[out_idx] = self.fill_value

        for out_idx, sub_idx in plan.base:
            out[out_idx] = self.base_getitem(sub_idx)

        return out

    def __setitem__(self, idx: Any, value: ArrayLike) -> None:
        """Break the given value into chunks and store it in memory.
        Do not modify the base array.

        This function may read data from the base array to fill chunks that are
        only partially covered by the index.

        If base_getitem() raises an exception at any point, the state of the
        StagedChangesArray is preserved coherent.

        Note
        ----
        This method preserves views of the value array and assumes it is OK to write
        back to it on later calls to __setitem__. If this is not desirable, you need to
        set the writeable flag to False on the value array before passing it to
        __setitem__.
        """
        plan = self._setitem_plan(idx)
        if not plan.mutates:
            return

        # Preprocess value parameter
        value = np.asarray(value)
        if plan.shape != value.shape:
            # This makes the value read-only
            value = np.broadcast_to(value, plan.shape)
        # If dtype is mismatch, this deep-copies the value and makes it writeable again.
        # It's better than just calling `asarray(value, dtype=self.dtype)` earlier
        # in the use case where we want to do both a broadcast and a dtype change,
        # so that now the slab is writeable.
        value = value.astype(self.dtype, copy=False)
        value = cast(NDArray[T], value)

        # Don't append directly to self.chunk_values so that
        # we don't corrupt the state if base_getitem() raises
        new_values: list[NDArray[T]] = []
        for input_idx, sub_indices in plan.loads:
            slab = self.base_getitem(input_idx)
            # Ensure we don't write back to the base array.
            # We will update all these chunks immediately afterwards,
            # so we can deep-copy straight away to speed things up.
            if slab.base is not None or not slab.flags.writeable:
                slab = slab.copy()
            # Note that multiple chunks will be views of the same slab
            new_values.extend(slab[sub_idx] for sub_idx in sub_indices)
        self.chunk_values.extend(new_values)

        self.chunk_values.extend(
            np.empty(shape, dtype=self.dtype) for shape in plan.append_empty
        )
        self.chunk_values.extend(
            np.full(shape, self.fill_value) for shape in plan.append_full
        )
        self.chunk_values.extend(
            # Note: these are views of the __setitem__ value array.
            value[out]
            for out in plan.append_direct
        )

        for idx, shape in plan.replace_empty:
            self.chunk_values[idx] = np.empty(shape, dtype=self.dtype)
        for idx, out in plan.replace_direct:
            # Note: these are views of the __setitem__ value array.
            self.chunk_values[idx] = value[out]

        for idx, out, sub in plan.updates:
            chunk = self.chunk_values[idx]
            assert chunk is not None
            if not chunk.flags.writeable:
                # Duplicate Copy-on-Write (CoW) chunk
                self.chunk_values[idx] = chunk = chunk.copy()
            chunk[sub] = value[out]

        self._has_changes = True
        self._chunk_states = plan.chunk_states

    def resize(self, shape: tuple[int, ...]) -> None:
        """Change the array shape in place and fill new elements with self.fill_value.

        Edge chunks which are not exactly divisible by chunk size are loaded in memory
        and partially filled with fill_value.
        In-memory chunks that are no longer needed are deleted. In order to retain the
        indices of chunk_values, they are replaced with None.
        """
        if self.shape == shape:
            return

        plan = self._resize_plan(shape)

        # Don't append directly to self.chunk_values so that
        # we don't corrupt the state if base_getitem() raises
        new_values: list[NDArray[T]] = []
        for input_idx, sub_indices in plan.loads:
            slab = self.base_getitem(input_idx)
            # Ensure we don't write back to base array
            if slab.base is not None:
                slab.flags.writeable = False
            # Note that multiple chunks will be views of the same slab
            new_values.extend(slab[sub_idx] for sub_idx in sub_indices)
        self.chunk_values += new_values

        for idx, axis, new_size in plan.updates:
            chunk = self.chunk_values[idx]
            assert chunk is not None
            self.chunk_values[idx] = _resize_array_along_axis(
                chunk, axis, new_size, fill_value=self.fill_value
            )

        for idx in plan.deletes:
            self.chunk_values[idx] = None

        self.shape = shape
        self._chunk_states = plan.chunk_states

    def copy(self) -> StagedChangesArray[T]:
        """Return a Copy-on-Write (CoW) copy of self. Chunks are duplicated only when
        they are modified in either the original array or the copy.
        """
        # Crucially, this does not deep-copy the base array if
        # self.base_getitem is a bound method
        out = copy.copy(self)

        if self._chunk_states is not None:
            out._chunk_states = self._chunk_states.copy()

        for chunk in self.chunk_values:
            if chunk is not None:
                chunk.flags.writeable = False
        out.chunk_values = self.chunk_values[:]
        return out

    def astype(self, dtype: DTypeLike, casting: Any = "unsafe") -> StagedChangesArray:
        """Return a new StagedChangesArray with a different dtype.

        Chunks that are already in memory are eagerly converted to the new dtype.
        Chunks that are not in memory are lazily converted upon calling changes().
        """
        if self.dtype == dtype:
            return self.copy()

        if self.base_getitem is DUMMY_GETITEM:
            new_getitem = DUMMY_GETITEM
        else:
            prev_getitem = self.base_getitem

            def new_getitem(idx: Any) -> NDArray:
                return prev_getitem(idx).astype(dtype, casting=casting)

        out = StagedChangesArray(
            base_getitem=new_getitem,
            shape=self.shape,
            chunk_size=self.chunk_size,
            dtype=dtype,
            fill_value=self.fill_value,
        )
        out.base_shape = self.base_shape
        out._base_getitem_transform = self.base_getitem is not DUMMY_GETITEM
        out._chunk_states = self.chunk_states.copy()

        out.chunk_values = []
        for chunk in self.chunk_values:
            if chunk is not None:
                chunk = chunk.astype(dtype, casting=casting)
                out._has_changes = True
            out.chunk_values.append(chunk)

        return out

    def refill(self, fill_value: Any) -> StagedChangesArray[T]:
        """Create a CoW copy of self with changed fill_value.

        This eagerly updates all modified chunks that are wholly or partially filled
        with the old fill_value and lazily refills the chunks that haven't been loaded
        from the base array yet as they get loaded.

        The new array will have its changes() method return all chunks, even those that
        are not marked as modified.
        """
        fill_value = np.asarray(fill_value, self.dtype)
        if fill_value.ndim != 0:
            raise ValueError("fill_value must be a scalar")

        out = self.copy()
        if fill_value == self.fill_value:
            return out

        out.fill_value = fill_value

        if self.base_getitem is not DUMMY_GETITEM:
            prev_fill_value = self.fill_value
            prev_getitem = self.base_getitem

            # TODO it would be useful to use a variant of this function for changes(),
            # to avoid yielding chunks without any points equal to the old fill_value.
            def new_getitem(idx: Any) -> NDArray[T]:
                slab = prev_getitem(idx)
                mask = slab == prev_fill_value
                if mask.any():
                    if slab.base is not None or not slab.flags.writeable:
                        slab = slab.copy()
                    slab[mask] = fill_value
                return slab

            out.base_getitem = new_getitem
            out._base_getitem_transform = True

        for i, chunk in enumerate(out.chunk_values):
            if chunk is not None:
                mask = chunk == self.fill_value
                if mask.any():  # Keep CoW chunks if they are not modified
                    out.chunk_values[i] = chunk = chunk.copy()
                    chunk[mask] = fill_value
                    out._has_changes = True

        return out

    @staticmethod
    def full(
        shape: tuple[int, ...],
        chunk_size: tuple[int, ...],
        dtype: DTypeLike | None = None,
        fill_value: Any | None = None,
    ) -> StagedChangesArray:
        """Create a new StagedChangesArray with all chunks already in memory and
        full of fill_value.
        It won't consume any significant amounts of memory until it's modified.
        """
        out = StagedChangesArray(
            base_getitem=DUMMY_GETITEM,
            shape=shape,
            chunk_size=chunk_size,
            dtype=dtype,
            fill_value=fill_value,
        )
        out._chunk_states = np.full(out.nchunks, -1, dtype=np.intp)
        return out


def DUMMY_GETITEM(_):
    """A special base_getitem that will never be called. It is used whenever a
    StagedChangedArray is built by full().
    """
    raise AssertionError("unreachable")  # pragma: nocover


@cython.cclass
@dataclass(init=False)
class GetItemPlan:
    """Instructions to execute StagedChangesArray.__getitem__"""

    #: Shape of the array returned by __getitem__
    shape: tuple[int, ...]

    #: Modified chunks already in memory, to be copied to the output array.
    #:
    #: List of tuples of:
    #: - index of StagedChangesArray.chunk_values
    #: - index to slice the output array
    #: - index to slice the chunk_values element
    modified: list[tuple[int, AnySlicerND, AnySlicerND]]

    #: Areas of the output array that must be filled with the fill_value
    full: list[AnySlicerND]

    #: Chunks that are not in memory and must be copied directly from the base array.
    #:
    #: List of tuples of:
    #: - index to slice the output array
    #: - index to slice the base array
    base: list[tuple[AnySlicerND, AnySlicerND]]

    def __init__(
        self,
        idx: Any,
        chunk_size: tuple[int, ...],
        shape: tuple[int, ...],
        chunk_states: NDArray[np.intp],
    ):
        """Generate instructions to execute StagedChangesArray.__getitem__

        Parameters
        ----------
        idx:
            Arbitrary numpy fancy index passed as a parameter to __getitem__

        All other parameters are the matching attributes of StagedChangesArray
        """
        idx, mappers = index_chunk_mappers(idx, chunk_size, shape)
        self.shape = idx.newshape(shape)
        self.modified = []
        self.full = []
        self.base = []

        if not mappers:
            # Empty selection
            return

        mod_chunks_indices: Py_ssize_t[:, :]
        mod_chunks_indices, any_unmodified = _modified_chunks_in_selection(
            chunk_states, mappers
        )
        ndim = len(mappers)
        n_mod_chunks = len(mod_chunks_indices)
        if n_mod_chunks == 0:
            # No modified or full chunks within the selection
            self.base.append(((), idx.raw))
            return

        # Copy modified chunks from in-memory cache
        for i in range(n_mod_chunks):
            states_idx = mod_chunks_indices[i, :ndim]
            values_idx = mod_chunks_indices[i, ndim]
            # This should never raise ChunkMapIndexError
            # as we already filtered with _modified_chunks_in_selection()
            out_idx, sub_idx = zip_chunk_submap(mappers, states_idx)
            if values_idx == -1:
                self.full.append(out_idx)
            else:
                self.modified.append((values_idx, out_idx, sub_idx))

        if not any_unmodified:
            return

        # Copy all unnmodified chunks from the underlying array, minimising
        # the number of reads, between two modified chunks.
        #
        # Example
        # -------
        # We want to get all of the chunks of a 3D array.
        # There are two modified chunks at [3, 4, 5] and [3, 7, 8]
        # 1. Copy the modified chunk at [3, 4, 5]
        # 2. Copy the modified chunk at [3, 7, 8]
        # 3. Copy from the underlying array:
        #    - [:3,  :,   :]
        #    - [ 3,  :4,  :]
        #    - [ 3,   4, :5]
        # 4. Copy from the underlying array:
        #    - [ 3,   4, 6:]
        #    - [ 3, 5:7,  :]
        #    - [ 3,   7, :8]
        # 5. Copy from the underlying array:
        #    - [ 3,   7, 8:]
        #    - [ 3,   8:, :]
        #    - [4:,    :, :]

        obstacles = mod_chunks_indices[:, :ndim]
        hyperrectangles = fill_hyperspace(obstacles, chunk_states.shape)
        for i in range(len(hyperrectangles)):
            fromc_idx = hyperrectangles[i, :ndim]
            toc_idx = hyperrectangles[i, ndim:]
            try:
                out_idx, sub_idx = zip_slab_submap(mappers, fromc_idx, toc_idx)
                self.base.append((out_idx, sub_idx))
            except ChunkMapIndexError:
                pass

    @property
    def head(self) -> str:
        return (
            f"GetItemPlan<shape={self.shape}, "
            f"{len(self.modified)} modified chunks, {len(self.full)} full chunks, "
            f"{len(self.base)} direct reads>"
        )

    def __repr__(self) -> str:
        s = self.head
        fmt = _fmt_fancy_index
        for values_idx, out_idx, sub_idx in self.modified:
            s += f"\n  res[{fmt(out_idx)}] = chunk_values[{values_idx}][{fmt(sub_idx)}]"
        for out_idx in self.full:
            s += f"\n  res[{fmt(out_idx)}] = fill_value"
        for out_idx, sub_idx in self.base:
            s += f"\n  res[{fmt(out_idx)}] = base[{fmt(sub_idx)}]"

        return s


@cython.cclass
class LoadSlabPlan:
    """Instructions to load a contiguous section of the base array into
    StagedChangedArray.chunk_values.

    This is a helper class, generated and consumed on the fly by SetItemPlan,
    ResizePlan and ChangesPlan.

    This is designed under the assumption that loading individual chunks from the base
    array is more expensive than loading at once larger slabs that cover multiple
    chunks.

    How to apply:

    >> v = staged_array.base_getitem(plan.input_idx())
    >> staged_array.chunk_states[plan.chunk_states_idx()] = (
    ..   plan.chunk_states_values() + len(staged_array.chunk_values)
    .. )
    >> staged_array.chunk_values.extend(v[idx] for idx in plan.sub_indices(shift=True))
    """

    #: Chunk index of the top-left corner to be loaded from the base array, included
    fromc_idx: Py_ssize_t[:]
    #: Chunk index of the bottom-right corner to be loaded from the base array, excluded
    toc_idx: Py_ssize_t[:]
    #: StagedChangesArray.chunk_size
    chunk_size: Py_ssize_t[:]
    ndim: Py_ssize_t

    def __init__(
        self,
        fromc_idx: Py_ssize_t[:],
        toc_idx: Py_ssize_t[:],
        chunk_size: Py_ssize_t[:],
    ):
        self.fromc_idx = fromc_idx
        self.toc_idx = toc_idx
        self.chunk_size = chunk_size
        self.ndim = len(chunk_size)

    @cython.cfunc
    def input_idx(self) -> tuple[slice, ...]:
        """Index to slice the base array with"""
        out = []
        for i in range(self.ndim):
            start = self.fromc_idx[i]
            stop = self.toc_idx[i]
            size = self.chunk_size[i]
            out.append(slice(start * size, stop * size, 1))
        return tuple(out)

    @cython.cfunc
    def sub_indices(self, shift: bint) -> list[tuple[slice, ...]]:
        """Return list of n-dimensional indices:
        If shift is True, to slice the base[input_idx] into chunks;
        If shift is False, to slice base for each chunk.
        """
        n_singles = 0
        slices_along_axes = []
        for i in range(self.ndim):
            start = self.fromc_idx[i]
            stop = self.toc_idx[i]

            if shift and stop == start + 1:
                # Single chunk along this axis
                n_singles += 1
                slices_along_axes.append([slice(None)])
                continue

            step = self.chunk_size[i]

            slices_i = []
            for i in range(stop - start):
                s = (i if shift else i + start) * step
                slices_i.append(slice(s, s + step, 1))
            slices_along_axes.append(slices_i)

        if n_singles == self.ndim:
            return [()]  # Single chunk

        return list(itertools.product(*slices_along_axes))

    @cython.cfunc
    def chunk_states_idx(self) -> tuple[slice, ...]:
        """N-dimensional index to select the target area within
        StagedChangesArray.chunk_states.
        """
        out = []
        for i in range(self.ndim):
            out.append(slice(self.fromc_idx[i], self.toc_idx[i], 1))
        return tuple(out)

    @cython.cfunc
    def chunk_states_values(self, offset: Py_ssize_t):  # -> NDArray[np.intp]:
        """Return array to fill StagedChangesArray.chunk_states with.

        Parameters
        ----------
        offset:
            Previous length of StagedChangesArray.chunk_values
        """
        size = 1
        shape = []
        for i in range(self.ndim):
            size_i = self.toc_idx[i] - self.fromc_idx[i]
            size *= size_i
            shape.append(size_i)

        return np.arange(offset, size + offset, dtype=np.intp).reshape(*shape)

    @staticmethod
    def generate(
        chunk_states: NDArray[np.intp],
        mappers: list[IndexChunkMapper],
        partial: bool,
    ) -> Iterator[LoadSlabPlan]:
        """Helper of SetItemPlan, ResizePlan, and LoadPlan that generates plans to load
        all chunks that are not yet in the cache so they must be loaded from the base
        array first.

        Parameters
        ----------
        chunk_states:
            StagedChangesArray.chunk_states
        mappers:
            For __setitem__(), output of index_chunk_mappers()
            For ChangesPlan(load_base=True), [EveythingMapper, ...]
            For resize(), [EveythingMapper, ...] for the new shape
        partial:
            True
                Only load chunks that are partially selected by the mappers. This is
                used by StagedChangesArray.__setitem__(), which won't need the previous
                chunks' contents for those chunks that are wholly selected, and
                StageChangesArray.resize(), which only needs to load the edge chunks
                that are not exactly divisible by the chunk size.
            False
                Load all chunks that are covered by the selection.
                This is used by ChangesPlan(load_base=True).
        """
        mod_chunks_indices: Py_ssize_t[:, :]
        try:
            mod_chunks_indices, any_unmodified = _modified_chunks_in_selection(
                chunk_states, mappers, plus_whole=partial
            )
        except ChunkMapIndexError:
            # No chunks are partially selected by the mappers
            return

        if not any_unmodified:
            return

        ndim = len(mappers)
        mapper: IndexChunkMapper
        chunk_size = empty_view(ndim)
        for i in range(ndim):
            mapper = mappers[i]
            chunk_size[i] = mapper.chunk_size

        # Discard chunk_values indices
        obstacles = mod_chunks_indices[:, :ndim]
        hyperrectangles = fill_hyperspace(obstacles, chunk_states.shape)
        for i in range(len(hyperrectangles)):
            fromc_idx_per_axis = []
            toc_idx_per_axis = []
            try:
                for j in range(ndim):
                    mapper = mappers[j]
                    aij = hyperrectangles[i, j]
                    bij = hyperrectangles[i, j + ndim]
                    contg = _contiguous_ranges(mapper.chunk_indices_in_range(aij, bij))
                    fromc_idx_per_axis.append(contg[:, 0])
                    toc_idx_per_axis.append(contg[:, 1])
            except ChunkMapIndexError:
                continue

            fromc_idx_cart = cartesian_product(fromc_idx_per_axis)
            toc_idx_cart = cartesian_product(toc_idx_per_axis)
            for i in range(len(fromc_idx_cart)):
                fromc_idx_i = fromc_idx_cart[i, :]
                toc_idx_i = toc_idx_cart[i, :]
                yield LoadSlabPlan(fromc_idx_i, toc_idx_i, chunk_size)


def _repr_loads(
    plans: list[
        tuple[
            tuple[slice, ...],
            list[tuple[slice, ...]],
        ]
    ],
    chunk_values_len: int,
) -> str:
    """Build section of SetItemPlan.__repr__ or ResizePlan.__repr__
    for their respective loads attribute.

    Parameters
    ----------
    plans:
        SetItemPlan.loads or ResizePlan.loads.
        This is a list of tuples:

        - index to slice the base array with
        - list of indices to slice the returned slab into chunks
    chunk_values_len:
        Size of chunk_values *after* the plans have been executed

    See Also
    --------
    SetItemPlan.__repr__
    ResizePlan.__repr__
    ChangesPlan.__repr__
    """
    fmt = _fmt_fancy_index
    values_idx = chunk_values_len - sum(len(sub_indices) for _, sub_indices in plans)
    i = 0
    s = ""
    for input_idx, sub_indices in plans:
        if len(sub_indices) == 1:
            s += "\n  chunk_values.append("
            s += f"base[{fmt(input_idx)}])  # [{values_idx}]"
            values_idx += 1
        else:
            s += f"\n  v{i} = base[{fmt(input_idx)}]"
            for sub_idx in sub_indices:
                s += f"\n  chunk_values.append(v{i}[{fmt(sub_idx)}])  # [{values_idx}]"
                values_idx += 1
            i += 1

    assert values_idx == chunk_values_len
    return s


@cython.cclass
@dataclass(init=False)
class SetItemPlan:
    """Instructions to execute StagedChangesArray.__setitem__"""

    #: Shape the value parameter must be broadcasted to
    shape: tuple[int, ...]

    #: Load chunks from the base array and append them to
    #: StagedChangesArray.chunk_values before doing anything else.
    #: These are chunks that are partially selected by the index.
    #: This is a list of tuples of:
    #: - index to slice the base array with
    #: - list of indices to slice the returned slab into chunks
    loads: list[tuple[tuple[slice, ...], list[tuple[slice, ...]]]]

    #: Append new chunks to StagedChangesArray.chunk_values and leave them empty.
    #: This must happen after the `loads` operation above.
    #: This is a list of shapes.
    append_empty: list[tuple[int, ...]]

    #: Append new chunks to StagedChangesArray.chunk_values and fill them with
    #: fill_value.
    #: This must happen after the `loads` and `append_empty` operations above.
    #: This is a list of shapes.
    append_full: list[tuple[int, ...]]

    #: Append new chunks to StagedChangesArray.chunk_values directly,
    #: potentially as views of the __setitem__ value parameter.
    #: This must happen after the `loads`, `append_empty`, and `append_full`
    #: operations above.
    #: This is a list indices to slice the value parameter to __setitem__, always
    #: guaranteed to generate exactly one chunk.
    append_direct: list[AnySlicerND]

    #: Replace chunks that are already in StagedChangesArray.chunk_values with new
    #: empty chunks of the given shape.
    replace_empty: list[tuple[int, tuple[int, ...]]]

    #: Replace chunks that are already in StagedChangesArray.chunk_values with slices
    #: of the value parameter to __setitem__.
    replace_direct: list[tuple[int, AnySlicerND]]

    #: Update chunks that are already in StagedChangesArray.chunk_values.
    #: Note that this includes chunks that were just created by `loads`, `append_full`,
    #: `append_empty`, and `replace_empty` above.
    #:
    #: List of tuples of:
    #: - index of StagedChangesArray.chunk_values
    #: - index to slice the value parameter to __setitem__
    #: - index to slice the chunk_values element
    updates: list[tuple[int, AnySlicerND, AnySlicerND]]

    #: chunk_states after the __setitem__ operation
    chunk_states: NDArray[np.intp]

    def __init__(
        self,
        idx: Any,
        chunk_size: tuple[int, ...],
        shape: tuple[int, ...],
        chunk_states: NDArray[np.intp],
        chunk_values_len: Py_ssize_t,
    ):
        """Generate instructions to execute StagedChangesArray.__setitem__.

        Parameters
        ----------
        idx:
            Arbitrary numpy fancy index passed as a parameter to __setitem__
        chunk_size:
            StagedChangesArray.chunk_size
        shape:
            StagedChangesArray.shape, e.g. shape of the target array that
            is being updated
        chunk_states:
            StagedChangesArray.chunk_states
        chunk_values_len:
            len(StagedChangesArray.chunk_values)
        """
        self.loads = []
        self.append_empty = []
        self.append_full = []
        self.append_direct = []
        self.replace_empty = []
        self.replace_direct = []
        self.updates = []

        idx, mappers = index_chunk_mappers(idx, chunk_size, shape)
        self.shape = idx.newshape(shape)
        if not mappers:
            # Empty selection
            self.chunk_states = chunk_states
            return

        # Note: this is not just for the sake of ideological purity of not writing back
        # to the parameters; it also allows the __getitem__ call to the base array to
        # raise for any reason without corrupting the state.
        # The cost of this copy is modest (0.3ms for 1M chunks).
        self.chunk_states = chunk_states = chunk_states.copy()

        load_plan: LoadSlabPlan
        for load_plan in LoadSlabPlan.generate(chunk_states, mappers, partial=True):
            self.loads.append(
                (load_plan.input_idx(), load_plan.sub_indices(shift=True))
            )
            v = load_plan.chunk_states_values(chunk_values_len)
            chunk_states[load_plan.chunk_states_idx()] = v
            chunk_values_len += v.size

        # Get the list of all chunks that are in memory.
        # This includes the ones we are going to loaded above with the LoadSlabPlan.
        mod_chunks_indices: Py_ssize_t[:, :]
        mod_chunks_indices, any_unmodified = _modified_chunks_in_selection(
            chunk_states, mappers
        )
        n_mod_chunks = len(mod_chunks_indices)
        ndim = len(mappers)

        append_empty_indices: list[Py_ssize_t[:]] = []
        append_full_indices: list[Py_ssize_t[:]] = []
        append_direct_indices: list[Py_ssize_t[:]] = []

        whole_chunk_tester = WholeChunkTester(mappers)
        chunk_shape_factory = ChunkShapeFactory(mappers)

        for i in range(n_mod_chunks):
            states_idx = mod_chunks_indices[i, :ndim]
            values_idx = mod_chunks_indices[i, ndim]
            out_idx, sub_idx = zip_chunk_submap(mappers, states_idx)

            # Is the chunk completely covered by the selection?
            is_whole_chunk = whole_chunk_tester.is_whole_chunk(states_idx)

            if is_whole_chunk:
                # Optimize the trivial case where the chunk sub-index is a
                # slice(0, chunk_size[i], 1) along all axes.
                # In this case, store a view of the __setitem__ value instead of a
                # deep-copy.
                #
                # In more complex cases, it's not always possible to avoid a deep-copy
                # and reorder using both the sub and out indices.
                # Consider: a[[2, 0, 1, 4, 3]] = np.arange(5)
                # If chunk_size=3, chunk 1 could theoretically be coerced into a
                # [4:2:-1] slice view of the __setitem__ value, but chunk 0 must be
                # necessarily deep-copied.
                # For the sake of simplicity, when there are array indices we just
                # deep-copy everything.
                #
                # TODO it would be fairly straightforward to tamper with `out` to make
                # this work with integer indices filling a chunk with size 1.
                # See logic for DROP_AXIS in zip_chunk_submap.
                if all(
                    isinstance(sub_i, slice) and sub_i.step == 1 for sub_i in sub_idx
                ):
                    if values_idx > 0:
                        # Real chunk already in chunk_values, completely covered by
                        # selection. Replace it with a view.
                        self.replace_direct.append((values_idx, out_idx))
                    else:
                        # Full chunk, completely covered by selection.
                        # Append a new real chunk in its place.
                        append_direct_indices.append(states_idx)
                        self.append_direct.append((out_idx))

                elif values_idx > 0:
                    # Real chunk already in chunk_values, wholly covered by
                    # selection but the index is non-trivial. First replace it
                    # with an empty array and then update the empty array.
                    chunk_shape = chunk_shape_factory.chunk_shape(states_idx)
                    self.replace_empty.append((values_idx, chunk_shape))
                    self.updates.append((values_idx, out_idx, sub_idx))

                else:
                    # Full chunk wholly covered by selection, but the index is
                    # non-trivial. First append an empty chunk and then update it.
                    values_idx = chunk_values_len + len(append_empty_indices)
                    append_empty_indices.append(states_idx)
                    chunk_shape = chunk_shape_factory.chunk_shape(states_idx)
                    self.append_empty.append(chunk_shape)
                    self.updates.append((values_idx, out_idx, sub_idx))

            elif values_idx > 0:
                # Real chunk already in chunk_values, partially covered by selection.
                # This includes chunks we just loaded with LoadSlabPlan above.
                # Update it.
                self.updates.append((values_idx, out_idx, sub_idx))

            else:
                # Full chunk, partially covered by selection.
                # We must first generate it as a real chunk full of fill_value
                # an then update it.
                values_idx = (
                    chunk_values_len
                    + len(append_empty_indices)
                    + len(append_full_indices)
                )
                append_full_indices.append(states_idx)
                chunk_shape = chunk_shape_factory.chunk_shape(states_idx)
                self.append_full.append(chunk_shape)
                self.updates.append((values_idx, out_idx, sub_idx))

        if any_unmodified:
            # __setitem__ wholly covers chunks that aren't full of fillvalue.
            # Create new cache items.
            obstacles = mod_chunks_indices[:, :ndim]
            hyperrectangles = fill_hyperspace(obstacles, chunk_states.shape)
            for i in range(len(hyperrectangles)):
                c_indices = []
                try:
                    for j in range(ndim):
                        mapper = mappers[j]
                        aij = hyperrectangles[i, j]
                        bij = hyperrectangles[i, j + ndim]
                        c_indices.append(mapper.chunk_indices_in_range(aij, bij))
                except ChunkMapIndexError:
                    continue

                for chunk_idx in cartesian_product(c_indices):
                    out_idx, sub_idx = zip_chunk_submap(mappers, chunk_idx)

                    # Same logic as above
                    if all(
                        isinstance(sub_i, slice) and sub_i.step == 1
                        for sub_i in sub_idx
                    ):
                        # A trivial slice along all axes covers the whole chunk.
                        # Append a new chunk as a view of the __setitem__ value.
                        append_direct_indices.append(chunk_idx)
                        self.append_direct.append((out_idx))
                    else:
                        # A non-trivial index covers the whole chunk.                        # Append a new empty chunk and then fill it with the __setitem__
                        # value, manipulated by the out and sub indices.
                        values_idx = chunk_values_len + len(append_empty_indices)
                        append_empty_indices.append(chunk_idx)
                        chunk_shape = chunk_shape_factory.chunk_shape(chunk_idx)
                        self.append_empty.append(chunk_shape)
                        self.updates.append((values_idx, out_idx, sub_idx))

        appends = append_empty_indices + append_full_indices + append_direct_indices
        if appends:
            chunk_states_idx = tuple(np.array(appends).T)
            chunk_states_values = np.arange(
                chunk_values_len,
                chunk_values_len + len(appends),
                dtype=np.intp,
            )
            chunk_states[chunk_states_idx] = chunk_states_values

    @property
    def mutates(self) -> bool:
        """Return True if this plan alters the state of the StagedChangesArray in any
        way; False otherwise"""
        # loads, append_empty, append_full, and replace_empty are
        # always propaedeutic to updates
        return bool(self.updates or self.append_direct or self.replace_direct)

    @property
    def head(self) -> str:
        nloaded_chunks = sum(len(chunks) for _, chunks in self.loads)
        return (
            f"SetItemPlan<shape={self.shape}, "
            f"{len(self.loads)} loads from base into {nloaded_chunks} chunks, "
            f"{len(self.append_empty)} appends of empty chunks, "
            f"{len(self.append_full)} appends of full chunks, "
            f"{len(self.append_direct)} appends from __setitem__ value, "
            f"{len(self.replace_empty)} replaces with empty chunks, "
            f"{len(self.replace_direct)} replaces from __setitem__ value, "
            f"{len(self.updates)} updates>"
        )

    def __repr__(self) -> str:
        chunk_values_len = int(self.chunk_states.max()) + 1
        values_idx = (
            chunk_values_len
            - len(self.append_empty)
            - len(self.append_full)
            - len(self.append_direct)
        )

        s = self.head + _repr_loads(self.loads, values_idx)
        fmt = _fmt_fancy_index

        for shape in self.append_empty:
            s += f"\n  chunk_values.append(np.empty({shape})  # [{values_idx}]"
            values_idx += 1

        for shape in self.append_full:
            s += f"\n  chunk_values.append(np.full({shape}, fill_value)  # [{values_idx}]"
            values_idx += 1

        for out_idx in self.append_direct:
            s += f"\n  chunk_values.append(values[{fmt(out_idx)}])  # [{values_idx}]"
            values_idx += 1

        assert values_idx == chunk_values_len

        for values_idx, shape in self.replace_empty:
            s += f"\n  chunk_values[{values_idx}] = np.empty({shape})"

        for values_idx, out_idx in self.replace_direct:
            s += f"\n  chunk_values[{values_idx}] = values[{fmt(out_idx)}]"

        for values_idx, out_idx, sub_idx in self.updates:
            s += f"\n  chunk_values[{values_idx}][{fmt(sub_idx)}] = "
            s += f"values[{fmt(out_idx)}]"

        s += f"\nchunk_states:\n{self.chunk_states}"
        return s


@cython.cclass
class ChunkShapeFactory:
    """Quickly calculate the shape of each chunk in a StagedChangesArray"""

    mappers: list[IndexChunkMapper]
    ndim: Py_ssize_t
    fast_exit: bint
    chunk_size: tuple[int, ...]
    scratch: Py_ssize_t[:]

    def __init__(self, mappers: list[IndexChunkMapper]):
        self.ndim = len(mappers)
        self.mappers = mappers
        chunk_size = []
        self.fast_exit = True

        for i in range(self.ndim):
            mapper: IndexChunkMapper = mappers[i]
            chunk_size.append(mapper.chunk_size)
            if mapper.last_chunk_size != mapper.chunk_size:
                self.fast_exit = False

        self.chunk_size = tuple(chunk_size)
        if not self.fast_exit:
            self.scratch = empty_view(self.ndim)

    @cython.ccall
    def chunk_shape(self, chunk_idx: Py_ssize_t[:]) -> tuple[int, ...]:
        if self.fast_exit:
            return self.chunk_size

        from_scratch = False
        for i in range(self.ndim):
            mapper: IndexChunkMapper = self.mappers[i]
            if (
                mapper.last_chunk_size != mapper.chunk_size
                and chunk_idx[i] == mapper.n_chunks - 1
            ):
                from_scratch = True
                self.scratch[i] = mapper.last_chunk_size
            else:
                self.scratch[i] = mapper.chunk_size

        if from_scratch:
            # 10x faster than tuple(self.scratch)
            return tuple([v for v in self.scratch])
        else:
            # 4x faster than building a new tuple from scratch every time
            return self.chunk_size


@cython.cclass
@dataclass(init=False)
class ChangesPlan:
    """Instructions to execute StagedChangesArray.changes().

    If the dtype has changed, all chunks that haven't been loaded into memory must be
    loaded now.
    """

    #: Load all the chunks from the base array that aren't already in memory and
    #  append them to StagedChangesArray.chunk_values.
    #: This is a list of tuples, where each tuple contains:
    #: - index to slice the base array with
    #: - list of indices to slice the returned slab into chunks
    #: - matching list of indices of the chunks within the base array
    loads: list[
        tuple[
            tuple[slice, ...],
            list[tuple[slice, ...]],
            list[tuple[slice, ...]],
        ]
    ]

    #: All chunks that are in chunk_values
    #: List of tuple of
    #: - slice of the base array corresponding to the chunk
    #: - index of the chunk in chunk_values
    modified: list[tuple[tuple[slice, ...], int]]

    #: All chunks that have been deleted as the result of a resize operation.
    #: This is a list of tuple of slices of the base array corresponding to the chunk.
    #:
    #: Note that, if an edge chunk has changed shape, the chunk with the same index
    #: will appear both in `modified` and here.
    #: For example, consider original shape=(45, 10) and chunk_size=(10, 10).
    #: After calling resize((48, 10)), chunk (4, 0) will appear
    #: - in `modified`, as ((slice(40, 48), slice(0, 10))
    #: - in `deleted` , as ((slice(40, 45), slice(0, 10))
    #:
    #: Somewhat counter-intuitively, this means that this list can be non-empty even
    #: after a resize() operation that enlarged the array.
    deleted: list[tuple[slice, ...]]

    def __init__(
        self,
        chunk_size: tuple[int, ...],
        shape: tuple[int, ...],
        base_shape: tuple[int, ...],
        chunk_states: NDArray[np.intp],
        load_base: bint,
    ):
        """Generate instructions to execute StagedChangesArray.changes().

        Parameters
        ----------
        chunk_size:
            StagedChangesArray.chunk_size
        shape:
            StagedChangesArray.shape, e.g. current shape
        base_shape:
            StagedChangesArray.base_shape, e.g. before any resize operation
        chunk_states:
            StagedChangesArray.chunk_states
        load_base:
            If True, load all chunks from the base array that aren't already in memory
            If False, only return the chunks that are already in memory
        """
        self.loads = []
        self.modified = []
        self.deleted = []

        _, mappers = index_chunk_mappers((), chunk_size, shape)
        if mappers:  # size > 0
            if load_base:
                load_plan: LoadSlabPlan
                for load_plan in LoadSlabPlan.generate(
                    chunk_states, mappers, partial=False
                ):
                    self.loads.append(
                        (
                            load_plan.input_idx(),
                            load_plan.sub_indices(shift=True),
                            load_plan.sub_indices(shift=False),
                        )
                    )

            mod_chunks_indices: Py_ssize_t[:, :]
            mod_chunks_indices, _ = _modified_chunks_in_selection(
                chunk_states, mappers, include_full=False
            )
            n_mod_chunks = len(mod_chunks_indices)
            ndim = len(mappers)

            for i in range(n_mod_chunks):
                out_idx, _ = zip_chunk_submap(mappers, mod_chunks_indices[i, :ndim])
                values_idx = mod_chunks_indices[i, ndim]
                assert values_idx > 0
                self.modified.append((out_idx, values_idx))

        if base_shape != shape:
            slices, cut_points = zip(
                *[
                    _chunk_slices_along_axis_on_resize(o, n, c)
                    for o, n, c in zip(base_shape, shape, chunk_size)
                ]
            )
            for i in range(len(base_shape)):
                self.deleted.extend(
                    # cartesian product of
                    # all deleted slices along the current axis
                    # X all kept slices along all previous axes
                    # X all slices along all consecutive axes
                    itertools.product(
                        *(
                            s[cp:] if j == i else s[:cp] if j < i else s
                            for j, (s, cp) in enumerate(zip(slices, cut_points))
                        )
                    )
                )

    @property
    def head(self) -> str:
        nloaded_chunks = sum(len(chunks) for _, chunks, _ in self.loads)
        return (
            f"ChangesPlan<{len(self.deleted)} deleted chunks,"
            f" {len(self.modified)} chunks in memory,"
            f" {len(self.loads)} loads from base into {nloaded_chunks} chunks>"
        )

    def __repr__(self) -> str:
        s = self.head

        fmt = _fmt_fancy_index
        # Print deleted first for the sake of clarity, even if when base is a dict it
        # doesn't matter, as they may overlap modified chunks
        for base_idx in self.deleted:
            s += f"\n  del base[{fmt(base_idx)}]"
        for base_idx, values_idx in self.modified:
            s += f"\n  base[{fmt(base_idx)}] = chunk_values[{values_idx}]"

        # Build the loads section.
        # This is a variant of _repr_loads.
        i = 0
        for input_idx, sub_indices, base_indices in self.loads:
            assert len(sub_indices) == len(base_indices)

            if len(sub_indices) == 1:
                assert base_indices[0] == input_idx
                s += f"\n  out[{fmt(input_idx)}] = "
                s += f"base[{fmt(input_idx)}]"
            else:
                s += f"\n  v{i} = base[{fmt(input_idx)}]"
                for base_idx, sub_idx in zip(base_indices, sub_indices):
                    s += f"\n  out[{fmt(base_idx)}] = v{i}[{fmt(sub_idx)}]"
                i += 1

        return s


@cython.cfunc
def _chunk_slices_along_axis_on_resize(
    old_size: Py_ssize_t, new_size: Py_ssize_t, chunk_size: Py_ssize_t
) -> tuple[list[slice], int]:
    """Helper of ChangesPlan.

    Return the slices cutting the base array into chunks, along a single axis that may
    have been resized, as a tuple of:

    - All slices that cut the old shape into chunks along the axis
    - Chunk index starting from which the chunks have been dropped.
      This includes the chunk index of the last chunk if the chunk has been resized.
    """
    old_nchunks = ceil_a_over_b(old_size, chunk_size)
    new_nchunks = ceil_a_over_b(new_size, chunk_size)
    old_partial = old_size % chunk_size
    new_partial = new_size % chunk_size

    if new_nchunks > old_nchunks:
        if old_partial:
            n = old_nchunks - 1
        else:
            n = old_nchunks
    elif new_nchunks < old_nchunks:
        if new_partial:
            n = new_nchunks - 1
        else:
            n = new_nchunks
    elif new_size == old_size:
        n = old_nchunks
    else:
        n = old_nchunks - 1

    slices = []
    for i in range(old_nchunks):
        start = i * chunk_size
        stop = min(start + chunk_size, old_size)
        slices.append(slice(start, stop, 1))
    return slices, n


@cython.cclass
@dataclass(init=False)
class ResizePlan:
    """Instructions to execute StagedChangesArray.resize()"""

    #: Load edge chunks from the base array and append them to
    #: StagedChangesArray.chunk_values before doing anything else.
    #: This is necessary:
    #: - while enlarging, when the old shape does not divide exactly by the chunk size
    #: - while shrinking, when the new shape does not divide exactly by the chunk size
    #:
    #: This is a list of tuples, where each tuple contains
    #: - index to slice the base array with
    #: - list of indices to slice the returned slab into chunks
    loads: list[tuple[tuple[slice, ...], list[tuple[slice, ...]]]]

    #: Update chunks that are already in StagedChangesArray.chunk_values.
    #: Note that this includes chunks that were just loaded by `loads` above.
    #: Corner chunks may be updated more than once.
    #: List of tuples of:
    #: - index of StagedChangesArray.chunk_values
    #: - axis along which to resize
    #: - new size along axis
    updates: list[tuple[int, int, int]]

    #: Replace chunk_values with None at the given indices
    deletes: list[int]

    #: chunk_states after the resize operation
    chunk_states: NDArray[np.intp]

    def __init__(
        self,
        chunk_size: tuple[int, ...],
        old_shape: tuple[int, ...],
        new_shape: tuple[int, ...],
        chunk_states: NDArray[np.intp],
        chunk_values_len: Py_ssize_t,
    ):
        """Generate instructions to execute StagedChangesArray.resize().

        Parameters
        ----------
        chunk_size:
            StagedChangesArray.chunk_size
        old_shape:
            StagedChangesArray.shape before the resize operation
        new_shape:
            StagedChangesArray.shape after the resize operation
        chunk_states:
            StagedChangesArray.chunk_states
        chunk_values_len:
            len(StagedChangesArray.chunk_values)
        """
        self.loads = []
        self.updates = []
        self.deletes = []

        if len(new_shape) != len(old_shape):
            raise ValueError(
                "Number of dimensions in resize from {old_shape} to {new_shape}"
            )

        # Read comment in SetItemPlan.__init__
        self.chunk_states = chunk_states = chunk_states.copy()

        # Perform the resize one axis at a time
        # Go through all the shrinking axes first.
        shrinks = []
        enlarges = []

        axis: Py_ssize_t
        old_size: Py_ssize_t
        new_size: Py_ssize_t

        for axis, (old_size, new_size) in enumerate(zip(old_shape, new_shape)):
            if new_size < old_size:
                shrinks.append((axis, new_size))
            elif new_size > old_size:
                enlarges.append((axis, new_size))

        chunk_shape_view = view_from_tuple(chunk_size)
        old_shape_view = view_from_tuple(old_shape)

        for axis, new_size in shrinks + enlarges:
            chunk_values_len = self._resize_along_axis(
                axis, new_size, chunk_shape_view, old_shape_view, chunk_values_len
            )
            old_shape_view[axis] = new_size

    @cython.cfunc
    def _resize_along_axis(
        self,
        axis: Py_ssize_t,
        new_size: Py_ssize_t,
        chunk_shape: Py_ssize_t[:],
        old_shape: Py_ssize_t[:],
        chunk_values_len: Py_ssize_t,
    ) -> Py_ssize_t:
        """Resize along a single axis.

        Return the updated chunk_values_len.
        """
        ndim = len(chunk_shape)

        new_shape = empty_view(ndim)
        for i in range(ndim):
            new_shape[i] = new_size if i == axis else old_shape[i]

        chunk_size = chunk_shape[axis]
        old_nchunks = ceil_a_over_b(old_shape[axis], chunk_shape[axis])
        new_nchunks = ceil_a_over_b(new_size, chunk_shape[axis])
        old_partial = old_shape[axis] % chunk_shape[axis]
        new_partial = new_size % chunk_shape[axis]

        edge_chunk = -1
        if new_nchunks < old_nchunks:
            if new_partial != 0:
                # Edge chunks shrink from whole to partial
                edge_chunk = new_nchunks - 1
                edge_partial = new_partial
        elif new_nchunks > old_nchunks:
            if old_partial != 0:
                # Edge chunks enlarge from partial to whole
                edge_chunk = old_nchunks - 1
                edge_partial = chunk_size
        else:
            # same number of chunks, but the shape along the axis has changed,
            # so one of the following must happen to the edge chunk:
            # - shrink from whole to partial, or
            # - enlarge from partial to whole, or
            # - shrink or enlarge from partial to partial
            edge_chunk = new_nchunks - 1
            edge_partial = new_partial or chunk_size

        if edge_chunk != -1:
            # Edge chunks (either old or new) are not exactly divisible by
            # chunk_size. They must be
            # 1. loaded from base, if they're not already;
            # 2. resized;
            # 3. if they're being enlarged, partially filled with fill_value.
            edge_start = edge_chunk * chunk_size

            _, mappers = index_chunk_mappers(
                idx=(slice(None),) * axis + (slice(edge_start, edge_start + 1, 1),),
                chunk_size=tuple(chunk_shape),
                shape=tuple(
                    max(s, new_size) if i == axis else s
                    for i, s in enumerate(old_shape)
                ),
            )
            load_plan: LoadSlabPlan
            for load_plan in LoadSlabPlan.generate(
                self.chunk_states, mappers, partial=True
            ):
                self.loads.append(
                    (load_plan.input_idx(), load_plan.sub_indices(shift=True))
                )
                v = load_plan.chunk_states_values(chunk_values_len)
                self.chunk_states[load_plan.chunk_states_idx()] = v
                chunk_values_len += v.size

            # Now that we loaded the edge chunks, we can resize and fill them
            edge_chunk_states = cast(
                NDArray[np.intp], self.chunk_states.take(edge_chunk, axis=axis)
            )
            # Don't resize full chunks
            values_idx = edge_chunk_states[edge_chunk_states > 0]
            # assert (edge_chunk_states > 1).all()
            self.updates.extend(
                (values_idx, axis, edge_partial) for values_idx in values_idx.tolist()
            )

        if new_nchunks < old_nchunks:
            # Dereference unneeded chunks from chunk_values
            to_drop = self.chunk_states[
                tuple(
                    slice(new_nchunks, None) if i == axis else slice(None)
                    for i in range(ndim)
                )
            ]
            to_drop = to_drop[to_drop > 0]
            self.deletes.extend(to_drop.tolist())

        # Fill new chunks with fill_value
        self.chunk_states = _resize_array_along_axis(
            self.chunk_states, axis, new_nchunks, fill_value=-1
        )

        return chunk_values_len

    @property
    def head(self) -> str:
        nloaded_chunks = sum(len(chunks) for _, chunks in self.loads)
        return (
            f"ResizePlan<"
            f"{len(self.loads)} loads from base into {nloaded_chunks} chunks, "
            f"{len(self.updates)} updates, {len(self.deletes)} deletes>"
        )

    def __repr__(self) -> str:
        chunk_values_len = int(self.chunk_states.max()) + 1
        s = self.head + _repr_loads(self.loads, chunk_values_len)

        for values_idx, axis, size in self.updates:
            s += f"\n  resize chunk_values[{values_idx}] to {size=} along {axis=}"
        for values_idx in self.deletes:
            s += f"\n  chunk_values[{values_idx}] = None"

        s += f"\nchunk_states:\n{self.chunk_states}"
        return s


@cython.cfunc
def _resize_array_along_axis(
    arr,  #:  NDArray[T],
    axis: Py_ssize_t,
    new_size: Py_ssize_t,
    fill_value: T,
):  # -> NDArray[T]:
    """Either shrink or enlarge an array along the right edge of a single axis.

    Parameters
    ----------
    a:
        The array to be transformed
    axis:
        The axis along which to resize
    new_size:
        The new size of the array along the axis
    fill_value:
        The value to fill new elements with, if enlarging the array
    """
    i: Py_ssize_t  # noqa: F841
    ndim: Py_ssize_t = arr.ndim
    old_size: Py_ssize_t = arr.shape[axis]
    delta = new_size - old_size

    if delta < 0:
        idx = tuple(
            slice(None, new_size) if i == axis else slice(None) for i in range(axis + 1)
        )
        return arr[idx]
    else:
        pad_with = tuple((0, delta if i == axis else 0) for i in range(ndim))
        return np.pad(arr, pad_with, mode="constant", constant_values=fill_value)


@cython.ccall
def _modified_chunks_in_selection(
    chunk_states: NDArray[np.intp],
    mappers: list[IndexChunkMapper],
    *,
    include_full: bint = True,
    plus_whole: bint = False,
) -> tuple[NDArray[np.intp], bint]:
    """Find all the modified chunks within a selection, in order of appearance
    within the flattened chunk_states.

    Parameters
    ----------
    chunk_states:
        StagedChangesArray.chunk_states.

        - Unmodified chunks are set to 0
        - Full chunks are set to -1
        - Modified chunks are positive integers that index
          StagedChangesArray.chunk_values (index 0 is unused).
    mappers:
        Output of index_chunk_mappers()
    include_full: optional
        If True, also return chunks that are full of the fill_value.
        The returned index of chunk_values will be -1 and you need to make sure you
        don't accidentally read from the end of the list.
        If False, omit them.
    plus_whole: optional
        If False (default), return all the chunks selected by the mappers that are
        marked as modified in chunk_states.

        If True, additionally return all the chunks that are unmodified, but that are
        wholly selected by the mappers along all axes.

        In other words: generate a negative map whose ~complement generated by
        fill_hyperspace() selects all chunks that are partially covered by the
        selection but are not in memory yet, so if we're going to partially overwrite
        them with __setitem__ or resize() we need to load them from the base array
        first.

    Returns
    -------
    Tuple of:

    - 2D array where each row represents a modified chunk, with the first ndim columns
      being the chunk indices in chunk_states and the rightmost column being the index
      in chunk_values.
    - boolean that's True if there are any gaps of unmodified chunks, False if there's
      none.

    Raises
    ------
    ChunkMapIndexError
        If plus_whole=True and all chunks are wholly selected
        (so loading partial chunks into memory is not necessary).

    **Example**

    >>> index = (slice(0, 20), slice(15, 45))
    >>> chunk_size = (10, 10)
    >>> shape = (30, 60)
    >>> chunk_states = np.array(
    ...   [[0, 0, 1, 4, 0, 0],
    ...    [0, 0, 0, 0, 5, 0],
    ...    [0, 0, 2, 0, 3, 0]]
    ... )
    >>> _, mappers = index_chunk_mappers(index, chunk_size, shape)
    >>> tuple(np.asarray(m.chunk_indices) for m in mappers)
    (array([0, 1]), array([1, 2, 3, 4]))
    >>> tuple(m.chunks_indexer() for m in mappers)
    (slice(0, 2, 1), slice(1, 5, 1))
    >>> tuple(m.whole_chunks_indexer() for m in mappers)
    (slice(0, 2, 1), slice(2, 4, 1))
    >>> _modified_chunks_in_selection(chunk_states, mappers)
    (array([[0, 2, 1],
           [0, 3, 4],
           [1, 4, 5]]), True)
    >>> _modified_chunks_in_selection(chunk_states, mappers, plus_whole=True)
    (array([[0, 2, 1],
           [0, 3, 4],
           [1, 2, 0],
           [1, 3, 0],
           [1, 4, 5]]), True)

    chunk_indices        = ([0, 1], [1, 2, 3, 4])
    whole chunks_indices = ([0, 1], [   2, 3   ])

    chunk_states    selection    plus_whole=False    plus_whole=True
    001400          .pppp.       ..14..              ..14..
    000050          .pwwp.       ....5.              ..005.
    002030          ......       ......              ......

                    (p=partial,  (0, 2) | 1          (0, 2) | 1
                     w=whole)    (0, 3) | 4          (0, 3) | 4
                                 (1, 4) | 5          (1, 2) | 0
                                                     (1, 3) | 0
                                                     (1, 4) | 5
    """
    axis: Py_ssize_t
    mapper: IndexChunkMapper

    if plus_whole:
        whole_indexers = []
        npartial: Py_ssize_t = 0
        for mapper in mappers:
            idx = mapper.chunks_indexer()
            widx = mapper.whole_chunks_indexer()
            whole_indexers.append(widx)
            if isinstance(idx, slice):
                npartial += not isinstance(widx, slice) or widx != idx
            else:
                assert isinstance(idx, np.ndarray)
                npartial += not isinstance(widx, np.ndarray) or len(widx) < len(idx)

        if npartial == 0:
            raise ChunkMapIndexError()  # All chunks are wholly selected

        # Add chunks that are wholly selected along all axes to the mask
        wholes = np.zeros_like(chunk_states)
        for axis, widx in enumerate(whole_indexers):
            widx_nd: list = [slice(None)] * wholes.ndim
            widx_nd[axis] = widx
            wholes[tuple(widx_nd)] += 1

    # Slice chunk_states
    indexers = tuple([mapper.chunks_indexer() for mapper in mappers])
    indexers = _independent_ndindex(indexers)
    states = chunk_states[indexers]
    assert states.ndim == chunk_states.ndim == len(mappers)

    if include_full:
        mask = states != 0
    else:
        mask = states > 0

    if plus_whole:
        wholes = wholes[indexers]
        assert wholes.shape == mask.shape
        mask |= wholes == len(mappers)

    idxidx = np.nonzero(mask)

    states_indices = [
        np.asarray(mapper.chunk_indices)[idxidx_i]
        for mapper, idxidx_i in zip(mappers, idxidx)
    ]
    values_indices = chunk_states[tuple(states_indices)]
    return (
        np.stack(states_indices + [values_indices], axis=1),
        values_indices.size < mask.size,
    )


@cython.cfunc
def _contiguous_ranges(indices: Py_ssize_t[:]) -> Py_ssize_t[:, :]:
    """Given an array of indices, return the edges [a, b[ of the contiguous chunk ranges
    within it, with a's on column 0 and b's on column 1.

    >>> np.asarray(_contiguous_ranges(np.array([3, 7, 8])))
    array([[3, 4], [7, 9]])
    """
    # indices of chunks right before a jump
    discontiguities: Py_ssize_t[:] = np.flatnonzero(np.diff(indices) - 1)

    ni = len(indices)
    ndi = len(discontiguities)
    out: Py_ssize_t[:, :] = np.empty((ndi + 1, 2), dtype=np.intp)

    for i in range(ndi + 1):
        out[i, 0] = indices[discontiguities[i - 1] + 1 if i != 0 else 0]
        out[i, 1] = indices[discontiguities[i] if i != ndi else ni - 1] + 1

    return out


@cython.cclass
class WholeChunkTester:
    """Given one mapper per axis, call their whole_chunk_indexer() method, digest its
    output, and provide a fast method to test whether a chunk is wholly selected along
    all axes or not.
    """

    all_partial: bint
    ndim: Py_ssize_t
    starts: Py_ssize_t[:]
    stops: Py_ssize_t[:]
    ranges_dims: Py_ssize_t[:]
    ranges: list[Py_ssize_t[:, :]]

    def __init__(self, mappers: list[IndexChunkMapper]):
        self.all_partial = False
        self.ndim = len(mappers)
        self.starts = empty_view(self.ndim)
        self.stops = empty_view(self.ndim)
        self.ranges_dims = empty_view(self.ndim)
        self.ranges = []
        nranges = 0

        for i in range(self.ndim):
            mapper: IndexChunkMapper = mappers[i]
            indexer = mapper.whole_chunks_indexer()
            if isinstance(indexer, slice):
                assert indexer.step == 1
                start: Py_ssize_t = indexer.start
                stop: Py_ssize_t = indexer.stop
                assert start >= 0
                if stop <= start:
                    self.all_partial = True
                    return
                self.starts[i] = start
                self.stops[i] = stop
            else:
                assert isinstance(indexer, np.ndarray) and indexer.dtype == np.intp
                indexer_len = len(indexer)
                if indexer_len == 0:
                    self.all_partial = True
                    return
                elif indexer_len > 2:
                    self.ranges_dims[nranges] = i
                    nranges += 1
                    self.ranges.append(_contiguous_ranges(indexer))

        if nranges < self.ndim:
            self.ranges_dims[nranges] = -1

    @cython.ccall
    def is_whole_chunk(self, idx: Py_ssize_t[:]) -> bint:
        """Return True if a chunk is wholly selected along all axes by all mappers;
        False otherwise
        """
        # If there's at least one empty slice or empty array index, early exit
        if self.all_partial:
            return False

        # Then go through the start and stop of each slice and index (O(1))
        for i in range(self.ndim):
            if idx[i] < self.starts[i] or idx[i] >= self.stops[i]:
                return False

        # Finally perform a bisection search through the array indices, if any
        # ( O(logn) and with Python interaction)
        for i in range(self.ndim):
            j = self.ranges_dims[i]
            if j == -1:
                break
            ranges: Py_ssize_t[:, :] = self.ranges[i]
            k: Py_ssize_t = np.searchsorted(ranges[:, 0], idx[j], side="right")
            assert 0 <= k < len(ranges)  # Thanks to the (starts, stops) check before
            # idx[j] is either within a contiguous range or between contiguous ranges
            if idx[j] >= ranges[k, 1]:
                return False

        return True


@cython.ccall
def _independent_ndindex(idx: tuple) -> tuple:
    """Given an n-dimensional array and a tuple index where each element could be a flat
    array-like, convert it to an index that numpy understands as 'for each axis, take
    these indices' - which is what most users would intuitively expect.

    Example
    >>> a = np.array([[ 0, 10, 20],
    ...               [30, 40, 50],
    ...               [60, 70, 80]])
    >>> a[[1, 2], [0, 1]]
    array([30, 70])
    >>> a[_independent_ndindex(([1, 2], [0, 1]))]
    array([[30, 40],
           [60, 70]])
    """
    arr_indices = []
    for i, idx_i in enumerate(idx):
        if isinstance(idx_i, slice):
            continue
        idx_i = np.asarray(idx_i)
        if idx_i.ndim == 0:
            continue
        arr_indices.append((i, idx_i))

    ndim = len(arr_indices)
    if ndim < 2:
        return idx

    out = list(idx)
    for j, (i, idx_i) in zip(range(ndim - 1, 0, -1), arr_indices[: ndim - 1]):
        out[i] = idx_i[(...,) + (None,) * j]
    return tuple(out)


def _fmt_slice(s: slice) -> str:
    start = "" if s.start is None else s.start
    stop = "" if s.stop is None else s.stop
    step = "" if s.step in (1, None) else f":{s.step}"
    return f"{start}:{stop}{step}"


def _fmt_fancy_index(idx: Any) -> str:
    if isinstance(idx, tuple):
        if idx == ():
            return "()"
    else:
        idx = (idx,)

    return ", ".join(_fmt_slice(i) if isinstance(i, slice) else str(i) for i in idx)
