# Note: this entire module is compiled by cython with wraparound=False
# See meson.build for details

# Read docs/staged_changes.rst for high-level documentation.

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Iterator, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, cast

import cython
import numpy as np
from cython import bint, ssize_t
from numpy.typing import ArrayLike, DTypeLike, NDArray

from versioned_hdf5.cytools import ceil_a_over_b, count2stop, np_hsize_t
from versioned_hdf5.slicetools import read_many_slices
from versioned_hdf5.subchunk_map import (
    EntireChunksMapper,
    IndexChunkMapper,
    TransferType,
    index_chunk_mappers,
    read_many_slices_params_nd,
)
from versioned_hdf5.tools import asarray, format_ndindex, ix_with_slices
from versioned_hdf5.typing_ import ArrayProtocol, MutableArrayProtocol

if cython.compiled:  # pragma: nocover
    from cython.cimports.versioned_hdf5.cytools import (  # type: ignore
        ceil_a_over_b,
        count2stop,
        hsize_t,
    )
    from cython.cimports.versioned_hdf5.subchunk_map import (  # type: ignore
        IndexChunkMapper,
        read_many_slices_params_nd,
    )


T = TypeVar("T", bound=np.generic)


class StagedChangesArray(MutableMapping[Any, T]):
    """Writeable numpy array-like, a.k.a. virtual array, which wraps around a
    sequence of read-only array-likes of chunks, known as the base slabs.

    The base slabs must be concatenations of chunks along axis 0, typically with shape
    (n*chunk_size[0], *chunk_size[1:]), where n is the number of chunks on the slab.

    All changes to the data or the shape are stored in memory into additional slabs
    backed by plain numpy arrays, which are appended after the full slab and the base
    slabs. Nothing writes back to the base slabs.

    In order to build a StagedChangesArray, in addition to the base slabs you must
    provide a mapping from each chunk of the virtual array to
    - the index in the sequence of base slabs, off by 1: element 0 in the sequence of
      base slabs is indexed as 1 etc. Index 0 refers to a special slab, a.k.a. the full
      slab, which is created internally and contains exactly 1 chunk full of the
      fill_value.
    - the offset along axis 0 of the referenced base slab that contains the chunk.
      Chunks full of the fill_value must always have offset 0.

    slabs = [full slab] + base_slabs + staged_slabs

    These two mappings are two arrays of integers, slab_indices and slab_offsets, with
    the same dimensionality as the virtual array and one point per chunk.

    e.g.::

        chunk_size[0] = 10

        slab_indices      slab_offsets  slabs[1]              slabs[2]
        (0 = fill_value)

        1 1 0             30 10  0       0:10 (unreferenced)   0:10 (unreferenced)
        0 2 0              0 20  0      10:20                 10:20
        0 2 0              0 10  0      20:30 (unreferenced)  20:30
                                        30:40

        virtual array (* = chunk completely covered in fill_value)

        slabs[1][30:40]  slabs[1][10:20]  slabs[0][0:10]*
        slabs[0][ 0:10]* slabs[2][20:30]  slabs[0][0:10]*
        slabs[0][ 0:10]* slabs[2][10:30]  slabs[0][0:10]*

    The base slabs may not cover the whole surface of the virtual array, which thus
    becomes a sparse array with missing chunks filled with a fill_value. Chunks are
    individually dense and setting a single point in a chunk materializes the whole
    chunk, so this differs from e.g. sparse.COO which is sparse at the single point
    level.

    Likewise, the slab_indices and slab_offsets mapping may point to the same chunk
    multiple times for multiple virtual slabs. This allows effectively implementing
    copy-on-write (CoW) system.

    High level documentation on how the class works internally: :doc:`staged_changes`.

    Parameters
    ----------
    shape:
        The shape of the presented array
    chunk_size:
        The shape of each chunk
    base_slabs:
        Sequence of zero or more read-only numpy-like objects containing the baseline
        data, with the chunks concatenated along axis 0. They will typically be shaped
        (n*chunk_size[0], *chunk_size[1:]), with n sufficiently large to accomodate all
        unique chunks, but may be larger than  that.
    slab_indices:
        Numpy array of integers with shape equal to the number of chunks along each
        axis, set to 0 for chunks that are full of fill_value and to i+1 for chunks that
        lie on base_slabs[i].
        It will be modified in place.
    slab_offsets:
        Numpy array of integers with shape matching slab_indices, mapping each chunk to
        the offset along axis 0 of its base slab. The offset of full chunks is always 0.
        It will be modified in place.
    fill_value: optional
        Value to fill chunks with where slab_indices=1. Defaults to 0.
    """

    #: current shape of the StagedChangesArray, e.g. downstream of resize()
    shape: tuple[int, ...]

    #: True if the user called resize() to alter the shape of the array; False otherwise
    _resized: bool

    #: Map from each chunk to the index of the corresponding slab in the slabs list
    slab_indices: NDArray[np_hsize_t]

    #: Offset of each chunk within the corresponding slab along axis 0
    slab_offsets: NDArray[np_hsize_t]

    #: Slabs of data, each containing one or more chunk stacked on top of each other
    #: along axis 0.
    #:
    #: slabs[0] contains exactly one chunk full of fill_value.
    #: It's broadcasted to the chunk_size and read-only.
    #: slabs[1:n_base_slabs + 1] are the slabs containing the original data.
    #: They can be any read-only numpy-like object.
    #: slabs[n_base_slabs + 1:] contain the staged chunks.
    #:
    #: Edge slabs that don't fully cover the chunk_size are padded with uninitialized
    #: cells; the shape of each staged slab is always
    #: (n*chunk_size[0], *chunk_size[1:]).
    #:
    #: When a slab no longer holds any current chunk, it is dereferenced and replaced
    #: with None in this list. This happens:
    #: 1. after modifying every chunk of a base slab, so that they now live entirely
    #:    in the staged slabs, and/or
    #: 2. when shrinking the array with resize().
    #: slabs[0] is never dereferenced.
    slabs: list[NDArray[T] | None]

    #: Number of base slabs in the slabs list.
    #: Staged slabs start at index n_base_slabs + 1.
    n_base_slabs: int

    #: Set to False to disable mutating methods
    writeable: bool

    __slots__ = tuple(__annotations__)

    def __init__(
        self,
        shape: tuple[int, ...],
        chunk_size: tuple[int, ...],
        base_slabs: Sequence[NDArray[T]],  # actually any numpy-compatible object
        slab_indices: ArrayLike,
        slab_offsets: ArrayLike,
        fill_value: Any | None = None,
    ):
        # Sanitize input (e.g. convert np.int64 to int)
        shape = tuple(int(i) for i in shape)
        chunk_size = tuple(int(i) for i in chunk_size)

        ndim = len(shape)
        if len(chunk_size) != ndim:
            raise ValueError("shape and chunk_size must have the same length")
        if any(s < 0 for s in shape):
            raise ValueError("shape must be non-negative")
        if any(c <= 0 for c in chunk_size):
            raise ValueError("chunk_size must be strictly positive")

        self.shape = shape
        self._resized = False
        self.writeable = True

        dtype = base_slabs[0].dtype if base_slabs else None
        if fill_value is None:
            # Unlike 0.0, this works for weird dtypes such as np.void
            fill_value = np.zeros((), dtype=dtype)
        else:
            fill_value = np.array(fill_value, dtype=dtype, copy=True)
            if fill_value.ndim != 0:
                raise ValueError("fill_value must be a scalar")
        assert fill_value.base is None

        for base_slab in base_slabs:
            if base_slab.dtype != fill_value.dtype:
                raise ValueError(
                    f"Mismatched dtypes {base_slab.dtype} != {fill_value.dtype}"
                )
            # Testing the minimum viable base_slab.shape would require a lot of nuance
            # and a full scan of the slab_indices and slab_offsets arrays, so it would
            # be overkill for a health check.
            if base_slab.ndim != ndim:
                raise ValueError(
                    "base_slabs must have the same dimensionality as shape"
                )

        self.slabs = [np.broadcast_to(fill_value, chunk_size)]
        self.slabs.extend(base_slabs)
        self.n_base_slabs = len(base_slabs)

        self.slab_indices = asarray(slab_indices, dtype=np_hsize_t)
        self.slab_offsets = asarray(slab_offsets, dtype=np_hsize_t)

        n_chunks = self.n_chunks
        if self.slab_indices.shape != n_chunks:
            raise ValueError(f"{self.slab_indices.shape=}; expected {n_chunks}")
        if self.slab_offsets.shape != n_chunks:
            raise ValueError(f"{self.slab_offsets.shape=}; expected {n_chunks}")

    @property
    def full_slab(self) -> NDArray[T]:
        return cast(NDArray[T], self.slabs[0])

    @property
    def fill_value(self) -> NDArray[T]:
        """Return array with ndim=0"""
        return cast(NDArray[T], self.full_slab.base)

    @property
    def dtype(self) -> np.dtype[T]:
        return self.full_slab.dtype

    @property
    def chunk_size(self) -> tuple[int, ...]:
        """Size of the tiles that will be modified at once. A write to
        less than a whole chunk will cause the remainder of the chunk
        to be read from the underlying array.
        """
        return self.full_slab.shape

    @property
    def itemsize(self) -> int:
        return self.full_slab.itemsize

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        """Size in bytes of this array if it were completely loaded. Actual used memory
        can be less or more depending on duplication between base and staged slabs,
        on full chunks, on chunks referenced multiple times, and on slab fragmentation.
        """
        return self.size * self.itemsize

    @property
    def n_chunks(self) -> tuple[int, ...]:
        """Number of chunks on each axis"""
        return tuple(
            ceil_a_over_b(s, c)
            for s, c in zip(self.shape, self.chunk_size, strict=True)
        )

    @property
    def n_slabs(self) -> int:
        """Total number of slabs"""
        return len(self.slabs)

    @property
    def n_staged_slabs(self) -> int:
        """Number of staged slabs containing modified chunks"""
        return self.n_slabs - self.n_base_slabs - 1

    @property
    def staged_slabs_start(self) -> int:
        """First index of staged slabs inside self.slabs"""
        return self.n_base_slabs + 1

    @property
    def base_slabs(self) -> Sequence[NDArray[T] | None]:
        return self.slabs[1 : self.staged_slabs_start]

    @property
    def staged_slabs(self) -> Sequence[NDArray[T] | None]:
        return self.slabs[self.staged_slabs_start :]

    @property
    def has_changes(self) -> bool:
        """Return True if any chunks have been modified or if a lazy transformation
        took place; False otherwise.
        """
        return self.n_staged_slabs > 0 or self._resized

    def __array__(self, dtype=None, copy=None) -> NDArray[T]:
        if copy is False:
            raise ValueError("Cannot return a ndarray view of a StagedChangesArray")
        out = self if dtype in (None, self.dtype) else self.astype(dtype)
        return np.asarray(out[()])

    def __len__(self) -> int:
        return self.shape[0]

    def __iter__(self) -> Iterator[NDArray[T]]:
        c = self.chunk_size[0]
        for start in range(0, self.shape[0], c):
            yield from self[start : start + c]

    def __delitem__(self, idx: Any) -> None:
        raise ValueError("Cannot delete array elements")

    def __repr__(self) -> str:
        return (
            f"StagedChangesArray<shape={self.shape}, chunk_size={self.chunk_size}, "
            f"dtype={self.dtype}, fill_value={self.fill_value.item()}, "
            f"{self.n_base_slabs} base slabs, {self.n_staged_slabs} staged slabs>"
            f"\nslab_indices:\n{self.slab_indices}"
            f"\nslab_offsets:\n{self.slab_offsets}"
        )

    # *Plan class factories. These are kept separate from the methods that consume them
    # for debugging purposes. Note that all plan feature a __repr__ method that is
    # intended to be used during demo or debugging sessions, e.g. in a Jupyter notebook.

    def _changes_plan(self) -> ChangesPlan:
        """Formulate a plan to export all the staged and unchanged chunks.

        This is a read-only operation.
        """
        return ChangesPlan(
            shape=self.shape,
            chunk_size=self.chunk_size,
            slab_indices=self.slab_indices,
            slab_offsets=self.slab_offsets,
        )

    def _getitem_plan(self, idx: Any) -> GetItemPlan:
        """Formulate a plan to get a slice of the array.

        This is a read-only operation.
        """
        return GetItemPlan(
            idx,
            shape=self.shape,
            chunk_size=self.chunk_size,
            slab_indices=self.slab_indices,
            slab_offsets=self.slab_offsets,
        )

    def _setitem_plan(self, idx: Any, copy: bool = True) -> SetItemPlan:
        """Formulate a plan to set a slice of the array.

        Parameters
        ----------
        idx:
            First argument of __setitem__. Note that the plan doesn't need to know about
            the value parameter.
        copy:
            False
                Generating the plan may update alter the state of the
                StagedChangesArray, so can only be done once without resulting in a
                broken state.
                This is faster, particularly when the impacted number of chunks is much
                smaller than the total, but you must follow through with the plan.
            True (default)
                Generating the plan won't alter the state of the StagedChangesArray
                until it's consumed by __setitem__.
                This is useful for debugging and testing.
        """
        # When writeable=False, we're going to abort the operation later.
        # Having copy=False would result in a corrupted state.
        copy = copy or not self.slab_indices.flags.writeable

        return SetItemPlan(
            idx,
            shape=self.shape,
            chunk_size=self.chunk_size,
            slab_indices=self.slab_indices.copy() if copy else self.slab_indices,
            slab_offsets=self.slab_offsets.copy() if copy else self.slab_offsets,
            n_slabs=self.n_slabs,
            n_base_slabs=self.n_base_slabs,
        )

    def _resize_plan(self, shape: tuple[int, ...], copy: bool = True) -> ResizePlan:
        """Formulate a plan to resize the array in place.

        See Also
        --------
        _setitem_plan
        """
        copy = copy or not self.slab_indices.flags.writeable

        return ResizePlan(
            old_shape=self.shape,
            new_shape=shape,
            chunk_size=self.chunk_size,
            slab_indices=self.slab_indices.copy() if copy else self.slab_indices,
            slab_offsets=self.slab_offsets.copy() if copy else self.slab_offsets,
            n_slabs=self.n_slabs,
            n_base_slabs=self.n_base_slabs,
        )

    def _load_plan(self, copy: bool = True) -> LoadPlan:
        """Formulate a plan to load all chunks from the base slabs into staged slabs.

        See Also
        --------
        _setitem_plan
        """
        copy = copy or not self.slab_indices.flags.writeable

        return LoadPlan(
            shape=self.shape,
            chunk_size=self.chunk_size,
            slab_indices=self.slab_indices.copy() if copy else self.slab_indices,
            slab_offsets=self.slab_offsets.copy() if copy else self.slab_offsets,
            n_slabs=self.n_slabs,
            n_base_slabs=self.n_base_slabs,
        )

    def _get_slab(
        self,
        idx: int | None,
        default: NDArray[T] | None = None,
        writeable: bool = False,
    ) -> NDArray[T]:
        """Fetch a slab by index, or the __getitem__/__setitem__ array if idx is None.

        If there was a previous call to copy() and the slab is going to be mutated,
        eagerly perform a deep-copy of the slab before returning it.
        If there was a previous call to astype(), eagerly convert the slab to the new
        dtype. In both cases, the returned slab replaces the previous one in self.slabs.
        """
        slab = self.slabs[idx] if idx is not None else default
        assert slab is not None
        assert slab.ndim == self.ndim

        if slab.dtype != self.dtype:
            # There was a previous call to astype().
            # `slab` Must be a staged slab.
            # The full slab and __setitem__ value are converted eagerly.
            # Base slabs are eagerly loaded or swapped out.
            assert idx is not None
            assert idx > self.n_base_slabs  # only staged slabs are converted lazily
            slab = asarray(slab, dtype=self.dtype)
            self.slabs[idx] = slab

        if writeable:
            # dst_slab is either a staged slab or the return value of __getitem__
            assert isinstance(slab, np.ndarray)
            if not slab.flags.writeable:
                assert idx is not None  # __getitem__ return value is always writeable
                assert idx > self.n_base_slabs  # Only staged slabs are writeable
                # There was a previous call to copy()
                slab = slab.copy()
                self.slabs[idx] = slab

        return slab

    def changes(
        self,
    ) -> Iterator[tuple[tuple[slice, ...], int, NDArray[T] | tuple[slice, ...]]]:
        """Yield all the changed chunks so far, as tuples of

        - slice index in the base virtual array
        - index of the slab in the slabs list
        - chunk value array, if a staged slab, or slice of the base slab otherwise

        Chunks that are completely full of the fill_value are not yielded.

        This lets you update the base virtual array:

        >> for idx, _, value in staged_array.changes():
        ..     if isinstance(value, np.ndarray):
        ..         virtual_base[idx] = value

        This is functionally a read-only operation; however chunks that were lazily
        converted with :meth:`astype` are going to be actually replaced in
        ``self.slabs``.
        """
        plan = self._changes_plan()

        for base_slice, slab_idx, slab_slice in plan.chunks:
            assert slab_idx > 0  # No full chunks
            if slab_idx <= self.n_base_slabs:
                yield base_slice, slab_idx, slab_slice
            else:
                slab = self._get_slab(slab_idx)
                chunk = slab[slab_slice]
                yield base_slice, slab_idx, chunk

    def __getitem__(self, idx: Any):
        """Get a slice of data from the array. This reads from the staged slabs
        in memory when available and from either the base slab or the fill_value
        otherwise.

        This is functionally a read-only operation; however chunks that were lazily
        converted with :meth:`astype` are going to be actually replaced in
        ``self.slabs``.

        Returns
        -------
        T if idx selects a scalar, otherwise NDArray[T]
        """
        plan = self._getitem_plan(idx)

        out = np.empty(plan.output_shape, dtype=self.dtype)
        out_view = out[plan.output_view]

        for tplan in plan.transfers:
            src_slab = self._get_slab(tplan.src_slab_idx)
            assert tplan.dst_slab_idx is None
            tplan.transfer(src_slab, out_view)

        # Return scalar value instead of a 0D array
        return out[()] if out.ndim == 0 else out

    def _apply_mutating_plan(
        self, plan: MutatingPlan, default_slab: NDArray[T] | None = None
    ) -> None:
        """Implement common workflow of __setitem__, resize, and load."""
        for shape in plan.append_slabs:
            self.slabs.append(np.empty(shape, dtype=self.dtype))

        for tplan in plan.transfers:
            src_slab = self._get_slab(tplan.src_slab_idx, default_slab)

            assert tplan.dst_slab_idx is not None
            assert tplan.dst_slab_idx > self.n_base_slabs
            dst_slab = self._get_slab(tplan.dst_slab_idx, writeable=True)

            tplan.transfer(src_slab, dst_slab)

        for slab_idx in plan.drop_slabs:
            assert slab_idx != 0  # Never drop the full slab
            self.slabs[slab_idx] = None

        # Even if we pass the indices arrays by reference to the *Plan
        # constructors, they may nonetheless be copied internally.
        self.slab_indices = plan.slab_indices
        self.slab_offsets = plan.slab_offsets

    def __setitem__(self, idx: Any, value: ArrayLike) -> None:
        """Update the slabs containing the chunks selected by the index.

        The full slab (slabs[0]) and the base slabs are read-only. If the selected
        chunks lie on any of them, append a new empty slab with the necessary space to
        hold all such chunks, then copy the chunks from the full slab or a base slab to
        the new slab, and finally update the new slab from the value parameter.
        """
        if not self.writeable:
            raise ValueError("assignment destination is read-only")

        plan = self._setitem_plan(idx, copy=False)
        if not plan.mutates:
            return

        # Preprocess value parameter
        # Avoid double deep-copy of array-like objects that support the __array_*
        # interface (e.g. sparse arrays).
        value = cast(NDArray[T], asarray(value, dtype=self.dtype))

        if plan.value_shape != value.shape:
            value = np.broadcast_to(value, plan.value_shape)
        value_view = value[plan.value_view]

        self._apply_mutating_plan(plan, value_view)

    def resize(self, shape: tuple[int, ...]) -> None:
        """Change the array shape in place and fill new elements with self.fill_value.

        When enlarging, edge chunks which are not exactly divisible by chunk size are
        partially filled with fill_value. This is a transfer from slabs[0] to a staged
        slab.

        If such slabs are not already in memory, they are first loaded.
        Just like in __setitem__, this appends a new empty slab to the slabs list, then
        transfers from a base slab to the new slab, and finally transfers from slabs[0]
        to the new slab.

        Slabs that are no longer needed are dereferenced; their location in the slabs
        list is replaced with None.
        This may cause base slabs to be dereferenced, but never the full slab.
        """
        if not self.writeable:
            raise ValueError("assignment destination is read-only")

        # Sanitize input (e.g. convert np.int64 to int)
        shape = tuple(int(s) for s in shape)

        plan = self._resize_plan(shape, copy=False)

        # A resize may change the slab_indices and slab_offsets, but won't necessarily
        # impact any chunks. In such cases, this is a no-op.
        self._apply_mutating_plan(plan)

        self.shape = shape
        self._resized = True

    def load(self) -> None:
        """Load all chunks that are not yet in memory from the base array."""
        if all(slab is None for slab in self.base_slabs):
            return
        if not self.writeable:
            raise ValueError("assignment destination is read-only")

        plan = self._load_plan(copy=False)
        self._apply_mutating_plan(plan)  # This may be a no-op

    def copy(self) -> StagedChangesArray[T]:
        """Return a writeable Copy-on-Write (CoW) copy of self.

        Staged slabs will be individually deep-copied upon first access on either side.
        Read-only full slab and base slabs are never copied.
        """
        # Force _setitem_plan, _resize_plan, and _load_plan to deep copy
        self.slab_indices.flags.writeable = False
        self.slab_offsets.flags.writeable = False
        for slab in self.staged_slabs:
            if slab is not None:
                # Force _get_slab() to deep-copy upon the first write access
                slab.flags.writeable = False

        out = copy.copy(self)  # Shallow object copy
        out.slabs = self.slabs.copy()  # Shallow list copy
        out.writeable = True  # Coherently with np.ndarray.copy()
        return out

    def astype(
        self,
        dtype: DTypeLike,
        base_slabs: Sequence[NDArray[T]] | None = None,
    ) -> StagedChangesArray:
        """Return a new StagedChangesArray with a different dtype.

        Staged slabs will be converted lazily, upon first access.

        Parameters
        ----------
        dtype:
            The new dtype.
        base_slabs: optional
            If provided, the new base slabs. Must have the same shapes as the original
            base_slabs be of the new dtype.
            If omitted, load into memory all chunks that are not yet staged and
            dereference the previous base slabs.
        """
        out = self.copy()
        dtype = np.dtype(dtype)
        if self.dtype == dtype:
            return out

        if base_slabs is None:
            # Load all base slabs, if any, into new staged slabs
            # and set all the base slabs to None
            out.load()
        else:
            # Hot-swap the base slabs
            for i, old in enumerate(out.base_slabs):
                if old is None:
                    continue
                new = base_slabs[i]
                if new is None:
                    raise TypeError(f"new base_slabs[{i}] is None but old one isn't")
                if new.dtype != dtype:
                    raise TypeError(
                        f"base_slabs[{i}] mismatched dtype: {new.dtype} != {dtype}"
                    )
                if new.shape != old.shape:
                    raise ValueError(
                        f"base_slabs[{i}] mismatched shape: {new.shape} != {old.shape}"
                    )
                out.slabs[i + 1] = new  # offset by the full slab at index 0

        # Convert the full slab; this has to happen after calling out.load().
        out.slabs[0] = np.broadcast_to(
            asarray(out.fill_value, dtype=dtype), out.chunk_size
        )

        # Leave the staged slabs as read-only views of the original slabs, with
        # original dtype. They will be converted lazily, upon first access, by
        # _get_slab(). This includes any base slabs we just loaded with out.load().
        return out

    def refill(self, fill_value: Any) -> StagedChangesArray[T]:
        """Create a copy of self with changed fill_value.

        TODO this method is implemented very naively, as it loads all chunks from the
        base slabs into the staged slabs, even when they don't contain any points set
        to the fill_value.
        """
        fill_value = np.asarray(fill_value, self.dtype)
        if fill_value.ndim != 0:
            raise ValueError("fill_value must be a scalar")

        out = self.copy()
        if fill_value == self.fill_value:
            return out

        out.load()
        out.slabs[0] = np.broadcast_to(fill_value, out.chunk_size)
        for idx, slab in enumerate(out.staged_slabs, start=out.staged_slabs_start):
            if slab is None:
                continue
            mask = slab == self.fill_value
            if np.any(mask):
                # Potentially remove read-only flag and convert dtype
                slab = out._get_slab(idx, writeable=True)
                slab[mask] = fill_value
        return out

    @staticmethod
    def full(
        shape: tuple[int, ...],
        *,
        chunk_size: tuple[int, ...],
        fill_value: Any | None = None,
        dtype: DTypeLike | None = None,
    ) -> StagedChangesArray:
        """Create a new StagedChangesArray with all chunks already in memory and
        full of fill_value.
        It won't consume any significant amounts of memory until it's modified.
        """
        # ceil_a_over_b coerces these to unsigned integers. The interpreter will
        # fail with MemoryError if either shape is negative or chunk_size is zero.
        if any(s < 0 for s in shape):
            raise ValueError("shape must be non-negative")
        if any(c <= 0 for c in chunk_size):
            raise ValueError("chunk_size must be strictly positive")
        n_chunks = tuple(
            ceil_a_over_b(s, c) for s, c in zip(shape, chunk_size, strict=True)
        )

        # StagedChangesArray.__init__ reads the dtype from fill_value
        if fill_value is not None:
            fill_value = np.array(fill_value, dtype=dtype)
        else:
            fill_value = np.zeros((), dtype=dtype)

        return StagedChangesArray(
            shape=shape,
            chunk_size=chunk_size,
            base_slabs=[],
            slab_indices=np.zeros(n_chunks, dtype=np_hsize_t),
            slab_offsets=np.zeros(n_chunks, dtype=np_hsize_t),
            fill_value=fill_value,
        )

    @staticmethod
    def from_array(
        arr: ArrayLike,
        *,
        chunk_size: tuple[int, ...],
        fill_value: Any | None = None,
        as_base_slabs: bool = True,
    ) -> StagedChangesArray:
        """Create a new StagedChangesArray from an array.

        Parameters
        ----------
        as_base_slabs:
            True (default)
                Set the base slabs as read-only views of ``arr``.
                This is mostly useful for debugging and testing.
            False
                Do not create any base slabs.
                Set the staged slabs as writeable views of ``arr`` if possible;
                otherwise as read-only views; otherwise as deep copies.
        """
        # Don't deep-copy array-like objects, as long as they allow for views
        arr = cast(np.ndarray, asarray(arr))

        if not as_base_slabs:
            # If a staged slab is not exactly divisible by chunk_size, it is going to be
            # problematic down the line if we call resize() to enlarge the array.
            # Use views of the array for all complete chunks and deep-copy the partial
            # edge chunks.
            shape_round_down = tuple(
                s // c * c for s, c in zip(arr.shape, chunk_size, strict=True)
            )
            if shape_round_down != arr.shape:
                out = StagedChangesArray.from_array(
                    arr[tuple(slice(s) for s in shape_round_down)],
                    chunk_size=chunk_size,
                    fill_value=fill_value,
                    as_base_slabs=False,
                )
                out.resize(arr.shape)
                _set_edges(out, arr, shape_round_down)
                return out

        out = StagedChangesArray.full(
            arr.shape, chunk_size=chunk_size, fill_value=fill_value, dtype=arr.dtype
        )
        if out.size == 0:
            return out

        # Iterate on all dimensions beyond the first. For each chunk, create a base slab
        # that is a view of arr of full length along axis 0 and 1 chunk in size along
        # all other axes.
        n_chunks = out.n_chunks
        slab_indices = np.arange(1, np.prod(n_chunks[1:]) + 1, dtype=np_hsize_t)
        out.slab_indices[()] = slab_indices.reshape(n_chunks[1:])[None]
        slab_offsets = np.arange(0, arr.shape[0], step=chunk_size[0])
        out.slab_offsets[()] = slab_offsets[(slice(None),) + (None,) * (out.ndim - 1)]

        for chunk_idx in itertools.product(*[list(range(c)) for c in n_chunks[1:]]):
            view_idx = (slice(None),) + tuple(
                slice(start := c * s, start + s)
                for c, s in zip(chunk_idx, chunk_size[1:], strict=True)
            )
            # Note: if the backend of arr doesn't support views
            # (e.g. h5py.Dataset), this is a deep-copy
            slab = arr[view_idx]
            if as_base_slabs:
                # Base slabs don't need to be numpy arrays but, if they are, mark them
                # as read-only. This is just a matter of hygene; we should never try
                # writing to them anyway.
                if isinstance(slab, np.ndarray):
                    slab.flags.writeable = False
            else:
                assert slab.shape[0] % chunk_size[0] == 0
                assert slab.shape[1:] == chunk_size[1:]
                if not isinstance(slab, np.ndarray):
                    # Staged slabs must be numpy arrays.
                    slab = np.asarray(slab)

            out.slabs.append(slab)

        if as_base_slabs:
            out.n_base_slabs = out.n_slabs - 1
        return out


@cython.cclass
@dataclass(init=False, eq=False, repr=False)
class TransferPlan:
    """Instructions to transfer data:

    - from a slab to the return value of __getitem__, or
    - from the value parameter of __setitem__ to a slab, or
    - between two slabs.

    Parameters
    ----------
    mappers:
        List of IndexChunkMapper objects, one for each axis, defining the selection.
    src_slab_idx:
        Index of the source slab in StagedChangesArray.slabs.
        During __setitem__, it must be None to indicate the value parameter.
    dst_slab_idx:
        Index of the destination slab in StagedChangesArray.slabs.
        During __getitem__, it must be None to indicate the output array.
    chunk_idxidx:
        2D array with shape (nchunks, ndim), one row per chunk being transferred and one
        columns per dimension. chunk_idxidx[i, j] represents the address on
        mappers[j].chunk_indices for the i-th chunk; in other words,
        mappers[j].chunk_indices[chunk_idxidx[i, j]] * chunk_size[j] is the address
        along axis j of the top-left corner of the chunk in the virtual dataset.
    src_slab_offsets:
        Offset of each chunk within the source slab along axis 0.
        Ignored during __setitem__.
    dst_slab_offsets:
        Offset of each chunk within the destination slab along axis 0.
        Ignored during __getitem__.
    slab_indices: optional
        StagedChangesArray.slab_indices. It will be updated in place.
        Ignored during __getitem__.
    slab_offsets: optional
        StagedChangesArray.slab_offsets. It will be updated in place.
        Ignored during __getitem__.
    """

    #: Index of the source slab in StagedChangesArray.slabs.
    #: During __setitem__, it can be None to indicate the value parameter.
    src_slab_idx: int | None

    #: Index of the destination slab in StagedChangesArray.slabs.
    #: During __getitem__, it's always None to indicate the output array.
    dst_slab_idx: int | None

    #: Parameters for read_many_slices().
    src_start: hsize_t[:, :]
    dst_start: hsize_t[:, :]
    count: hsize_t[:, :]
    src_stride: hsize_t[:, :]
    dst_stride: hsize_t[:, :]

    def __init__(
        self,
        mappers: list[IndexChunkMapper],
        *,
        src_slab_idx: int | None,
        dst_slab_idx: int | None,
        chunk_idxidx: hsize_t[:, :],
        src_slab_offsets: hsize_t[:],
        dst_slab_offsets: hsize_t[:],
        slab_indices: NDArray[np_hsize_t] | None = None,
        slab_offsets: NDArray[np_hsize_t] | None = None,
    ):
        self.src_slab_idx = src_slab_idx
        self.dst_slab_idx = dst_slab_idx
        nchunks = chunk_idxidx.shape[0]
        ndim = chunk_idxidx.shape[1]

        assert src_slab_idx != dst_slab_idx
        assert dst_slab_idx is None or dst_slab_idx > 0
        if src_slab_idx is None:
            transfer_type = TransferType.setitem
        elif dst_slab_idx is None:
            transfer_type = TransferType.getitem
        else:  # slab-to-slab
            transfer_type = TransferType.slab_to_slab

        # Generate slices for each axis, then put them in
        # pseudo cartesian product following chunk_idxidx.
        slices_nd = read_many_slices_params_nd(
            transfer_type,
            mappers,
            chunk_idxidx,
            src_slab_offsets,
            dst_slab_offsets,
        )
        # Refer to subchunk_map.pxd::ReadManySlicesNDColumn for column indices
        self.src_start = slices_nd[:, 0, :]
        self.dst_start = slices_nd[:, 1, :]
        self.count = slices_nd[:, 2, :]
        self.src_stride = slices_nd[:, 3, :]
        self.dst_stride = slices_nd[:, 4, :]

        # Finally, update slab_indices and slab_offsets in place
        if slab_indices is not None:
            assert slab_offsets is not None

            chunk_indices = np.empty((ndim, nchunks), dtype=np_hsize_t)
            chunk_indices_v: hsize_t[:, :] = chunk_indices
            for j in range(ndim):
                mapper: IndexChunkMapper = mappers[j]
                for i in range(nchunks):
                    chunk_indices_v[j, i] = mapper.chunk_indices[chunk_idxidx[i, j]]
            chunk_indices_tup = tuple(chunk_indices)
            slab_indices[chunk_indices_tup] = dst_slab_idx
            slab_offsets[chunk_indices_tup] = np.asarray(dst_slab_offsets)

    @cython.ccall
    def transfer(self, src: NDArray[T], dst: NDArray[T]):
        """Call read_many_slices() to transfer slices of data from src to dst"""
        read_many_slices(
            src,
            dst,
            self.src_start,
            self.dst_start,
            self.count,
            self.src_stride,
            self.dst_stride,
        )

    def __len__(self) -> int:
        """Return number of slices that will be transferred"""
        return self.src_start.shape[0]

    @cython.cfunc
    def _repr_idx(self, i: ssize_t, start: hsize_t[:, :], stride: hsize_t[:, :]) -> str:
        """Return a string representation of the i-th row"""
        ndim = start.shape[1]
        idx = []
        for j in range(ndim):
            start_ij = start[i, j]
            count_ij = self.count[i, j]
            stride_ij = stride[i, j]
            stop_ij = count2stop(start_ij, count_ij, stride_ij)
            idx.append(slice(start_ij, stop_ij, stride_ij))
        return format_ndindex(tuple(idx))

    def __repr__(self) -> str:
        """This is meant to be incorporated by the __repr__ method of the
        other *Plan classes
        """
        src = f"slabs[{i}]" if (i := self.src_slab_idx) is not None else "value"
        dst = f"slabs[{i}]" if (i := self.dst_slab_idx) is not None else "out"
        nslices = self.src_start.shape[0]
        s = f"\n  # {nslices} transfers from {src} to {dst}"
        for i in range(nslices):
            src_idx = self._repr_idx(i, self.src_start, self.src_stride)
            dst_idx = self._repr_idx(i, self.dst_start, self.dst_stride)
            s += f"\n  {dst}[{dst_idx}] = {src}[{src_idx}]"

        return s


@cython.cclass
@dataclass(init=False, repr=False)
class GetItemPlan:
    """Instructions to execute StagedChangesArray.__getitem__"""

    #: Shape of the array returned by __getitem__
    output_shape: tuple[int, ...]

    #: Index to slice the output array to add extra dimensions to ensure it's got
    #: the same dimensionality as the base array
    output_view: tuple[slice | None, ...]

    transfers: list[TransferPlan]

    def __init__(
        self,
        idx: Any,
        shape: tuple[int, ...],
        chunk_size: tuple[int, ...],
        slab_indices: NDArray[np_hsize_t],
        slab_offsets: NDArray[np_hsize_t],
    ):
        """Generate instructions to execute StagedChangesArray.__getitem__

        Parameters
        ----------
        idx:
            Arbitrary numpy fancy index passed as a parameter to __getitem__

        All other parameters are the matching attributes of StagedChangesArray
        """
        idx, mappers = index_chunk_mappers(idx, shape, chunk_size)
        self.output_shape = idx.newshape(shape)
        self.output_view = tuple(mapper.value_view_idx for mapper in mappers)
        self.transfers = []

        if not mappers:
            # Empty selection
            return

        chunks = _chunks_in_selection(
            slab_indices,
            slab_offsets,
            mappers,
            idxidx=True,
        )
        self.transfers.extend(
            _make_transfer_plans(
                mappers,
                chunks,
                src_slab_idx="chunks",
                dst_slab_idx=None,
            )
        )

    @property
    def head(self) -> str:
        ntransfers = sum(len(tplan) for tplan in self.transfers)
        return (
            f"GetItemPlan<output_shape={self.output_shape}, "
            f"output_view=[{format_ndindex(self.output_view)}], "
            f"{ntransfers} slice transfers among {len(self.transfers)} slab pairs>"
        )

    def __repr__(self) -> str:
        return self.head + "".join(str(tplan) for tplan in self.transfers)


@cython.cclass
@dataclass(repr=False)
class MutatingPlan:
    """Common ancestor of all plans that mutate StagedChangesArray"""

    #: Metadata arrays of StagedChangesArray. The parameters passed to __init__ may
    #: either be updated in place or replaced while formulating the plan.
    slab_indices: NDArray[np_hsize_t]
    slab_offsets: NDArray[np_hsize_t]

    #: Create new uninitialized slabs with the given shapes
    #: and append them StagedChangesArray.slabs
    append_slabs: list[tuple[int, ...]] = field(init=False, default_factory=list)

    #: data transfers between slabs or from the __setitem__ value to a slab.
    #: dst_slab_idx can include the slabs just created by append_slabs.
    transfers: list[TransferPlan] = field(init=False, default_factory=list)

    #: indices of StagedChangesArray.slabs to replace with None,
    #: thus dereferencing the slab. This must happen *after* the transfers.
    drop_slabs: list[int] = field(init=False, default_factory=list)

    @property
    def mutates(self) -> bool:
        """True if this plan alters the state of the StagedChangesArray"""
        return bool(self.transfers or self.drop_slabs)

    @property
    def head(self) -> str:
        """This is meant to be incorporated by the head() property of the subclasses"""
        ntransfers = sum(len(tplan) for tplan in self.transfers)
        return (
            f"append {len(self.append_slabs)} empty slabs, "
            f"{ntransfers} slice transfers among {len(self.transfers)} slab pairs, "
            f"drop {len(self.drop_slabs)} slabs>"
        )

    def __repr__(self) -> str:
        """This is meant to be incorporated by the __repr__ method of the subclasses"""
        s = self.head

        if self.append_slabs:
            max_slab_idx = self.slab_indices.max()
            slab_start_idx = int(max_slab_idx) - len(self.append_slabs) + 1
            assert slab_start_idx > 0
            for slab_idx, shape in enumerate(self.append_slabs, slab_start_idx):
                s += f"\n  slabs.append(np.empty({shape}))  # slabs[{slab_idx}]"

        s += "".join(str(tplan) for tplan in self.transfers)
        for slab_idx in self.drop_slabs:
            s += f"\n  slabs[{slab_idx}] = None"
        s += f"\nslab_indices:\n{self.slab_indices}"
        s += f"\nslab_offsets:\n{self.slab_offsets}"
        return s


@cython.cclass
@dataclass(init=False, repr=False)
class SetItemPlan(MutatingPlan):
    """Instructions to execute StagedChangesArray.__setitem__"""

    #: Shape the value parameter must be broadcasted to
    value_shape: tuple[int, ...]

    #: Index to slice the value parameter array to add extra dimensions to ensure it's
    #: got the same dimensionality as the base array
    value_view: tuple[slice | None, ...]

    def __init__(
        self,
        idx: Any,
        shape: tuple[int, ...],
        chunk_size: tuple[int, ...],
        slab_indices: NDArray[np_hsize_t],
        slab_offsets: NDArray[np_hsize_t],
        n_slabs: int,
        n_base_slabs: int,
    ):
        """Generate instructions to execute StagedChangesArray.__setitem__.

        Parameters
        ----------
        idx:
            Arbitrary numpy fancy index passed as a parameter to __setitem__

        All other parameters are the matching attributes and properties of
        StagedChangesArray.

        **Implementation notes**
        There are three use cases to cover:

        1. A chunk is full of fill_value or lies on a base slab,
           and is only partially covered by the selection. We need a 4-pass process:
           a. create a new empty slab;
           b. transfer the chunk from the base or full slab to the new slab;
           c. update slab_indices and slab_offsets to reflect the transfer;
           d. potentially dereference the base slab if it contains no other chunks;
           e. update the new slab with the __setitem__ value.

        2. A chunk is full of fill_value or lies on a base slab,
           and is wholly covered by the selection:
           a. create a new empty slab;
           b. update the new slab with the __setitem__ value;
           c. update slab_indices and slab_offsets to reflect that the chunk is no
              longer on the base or full slab;
           d. potentially dereference the base slab if it contains no other chunks.

        3. A chunk is on a staged slab:
           a. update the staged slab with the __setitem__ value.
        """
        super().__init__(slab_indices, slab_offsets)

        ndim = len(shape)
        idx, mappers = index_chunk_mappers(idx, shape, chunk_size)
        self.value_shape = idx.newshape(shape)
        self.value_view = tuple(mapper.value_view_idx for mapper in mappers)

        # We'll deep-copy later, only if needed
        if not mappers:
            # Empty selection
            return

        # Use case 1
        # Find all chunks that lie on a base or full slab and are only partially
        # covered by the selection.
        partial_chunks = _chunks_in_selection(
            slab_indices,
            slab_offsets,
            mappers,
            filter=lambda slab_idx: slab_idx <= n_base_slabs,
            idxidx=True,
            only_partial=True,
        )
        n_partial_chunks = partial_chunks.shape[0]
        if n_partial_chunks > 0:
            # Create a new empty slab and copy the chunks entirely from the base or full
            # slab. Then, update slab_indices and slab_offsets.

            self.append_slabs.append(
                (n_partial_chunks * chunk_size[0],) + chunk_size[1:]
            )
            entire_chunks_mappers: list[IndexChunkMapper] = [
                EntireChunksMapper(mapper) for mapper in mappers
            ]
            self.transfers.extend(
                _make_transfer_plans(
                    entire_chunks_mappers,
                    partial_chunks,
                    src_slab_idx="chunks",
                    dst_slab_idx=n_slabs,
                    slab_indices=slab_indices,  # Modified in place
                    slab_offsets=slab_offsets,  # Modified in place
                )
            )

        # Now all chunks are either on staged slabs or are wholly covered by the
        # selection. Effectively use case 1 has been simplified to use case 3: as long
        # as a chunk already lies on a staged slab, we don't care if the selection is
        # whole or partial.
        setitem_chunks = _chunks_in_selection(
            slab_indices,
            slab_offsets,
            mappers,
            idxidx=True,
        )
        n_setitem_chunks = setitem_chunks.shape[0]

        # Break the above into use cases 2 and 3
        cut: ssize_t = int(np.searchsorted(setitem_chunks[:, ndim], n_base_slabs + 1))
        if cut > 0:
            # Use case 2. These chunks lie on a read-only full or base slab, so we can't
            # overwrite them with the __setitem__ value parameter.

            self.append_slabs.append((int(cut) * chunk_size[0],) + chunk_size[1:])
            self.transfers.extend(
                _make_transfer_plans(
                    mappers,
                    setitem_chunks[:cut],
                    src_slab_idx=None,
                    dst_slab_idx=n_slabs + (n_partial_chunks > 0),
                    slab_indices=slab_indices,  # Modified in place
                    slab_offsets=slab_offsets,  # Modified in place
                )
            )

        if cut < n_setitem_chunks:
            # Use case 3. These chunks are on staged slabs and can be updated in place
            # with the __setitem__ value parameter.
            # Note that this includes chunks that were previously in use case 1.
            # slab_indices and slab_offsets don't need to be modified.
            self.transfers.extend(
                _make_transfer_plans(
                    mappers,
                    setitem_chunks[cut:],
                    src_slab_idx=None,
                    dst_slab_idx="chunks",
                )
            )

        # DO NOT perform a full scan of self.slab_indices in order to populate
        # self.drop_slabs. Everything so far has been O(selected chunks), do not
        # introduce an operation that is O(all chunks)!

    @property
    def head(self) -> str:
        return (
            f"SetItemPlan<value_shape={self.value_shape}, "
            f"value_view=[{format_ndindex(self.value_view)}], " + super().head
        )


@cython.cclass
@dataclass(init=False, repr=False)
class LoadPlan(MutatingPlan):
    """Load all chunks that have not been loaded yet from the base slabs."""

    def __init__(
        self,
        shape: tuple[int, ...],
        chunk_size: tuple[int, ...],
        slab_indices: NDArray[np_hsize_t],
        slab_offsets: NDArray[np_hsize_t],
        n_slabs: int,
        n_base_slabs: int,
    ):
        super().__init__(slab_indices, slab_offsets)
        if n_base_slabs == 0:
            return
        _, mappers = index_chunk_mappers((), shape, chunk_size)
        if not mappers:
            return  # size 0

        chunks = _chunks_in_selection(
            slab_indices,
            slab_offsets,
            mappers,
            filter=lambda slab_idx: (0 < slab_idx) & (slab_idx <= n_base_slabs),
            idxidx=True,
        )
        nchunks = chunks.shape[0]
        if nchunks > 0:
            self.append_slabs.append((nchunks * chunk_size[0],) + chunk_size[1:])

            self.transfers.extend(
                _make_transfer_plans(
                    mappers,
                    chunks,
                    src_slab_idx="chunks",
                    dst_slab_idx=n_slabs,
                    slab_indices=slab_indices,  # Modified in place
                    slab_offsets=slab_offsets,  # Modified in place
                )
            )

        # Also drop slabs that were previously loaded by SetItemPlan or ResizePlan, but
        # which were not in SetItemPlan.drop_slabs because of performance reasons.
        self.drop_slabs.extend(range(1, n_base_slabs + 1))

    @property
    def head(self) -> str:
        return "LoadPlan<" + super().head


@cython.cclass
@dataclass(init=False, repr=False)
class ChangesPlan:
    """Instructions to execute StagedChangesArray.changes()."""

    #: List of all chunks that aren't full of the fill_value.
    #:
    #: List of tuples of
    #: - index to slice the base array with
    #: - index of StagedChangesArray.slabs
    #: - index to slice the slab to retrieve the chunk value
    chunks: list[tuple[tuple[slice, ...], int, tuple[slice, ...]]]

    def __init__(
        self,
        shape: tuple[int, ...],
        chunk_size: tuple[int, ...],
        slab_indices: NDArray[np_hsize_t],
        slab_offsets: NDArray[np_hsize_t],
    ):
        """Generate instructions to execute StagedChangesArray.changes().

        All parameters are the matching attributes of StagedChangesArray.
        """
        self.chunks = []

        _, mappers = index_chunk_mappers((), shape, chunk_size)
        if not mappers:
            return  # size 0

        # Build rulers of slices for each axis
        dst_slices: list[list[slice]] = []  # Slices in the represented array
        slab_slices: list[list[slice]] = [[]]  # Slices in the slab (except axis 0)

        mapper: IndexChunkMapper
        for mapper in mappers:
            dst_slices_ix = []
            a: hsize_t = 0
            assert mapper.n_chunks > 0  # not size 0
            for _ in range(mapper.n_chunks - 1):
                b = a + mapper.chunk_size
                dst_slices_ix.append(slice(a, b, 1))
                a = b
            b = a + mapper.last_chunk_size
            dst_slices_ix.append(slice(a, b, 1))
            dst_slices.append(dst_slices_ix)

        # slab slices on axis 0 must be built on the fly for each chunk,
        # as each chunk has a different slab offset
        mapper = mappers[0]
        axis0_chunk_sizes: hsize_t[:] = np.full(
            mapper.n_chunks, mapper.chunk_size, dtype=np_hsize_t
        )
        axis0_chunk_sizes[mapper.n_chunks - 1] = mapper.last_chunk_size

        # slab slices on the other axes can be built with a ruler
        # (and they'll be all the same except for the last chunk)
        for mapper in mappers[1:]:
            slab_slices.append(
                [slice(0, mapper.chunk_size, 1)] * (mapper.n_chunks - 1)
                + [slice(0, mapper.last_chunk_size, 1)]
            )

        # Find all non-full chunks
        chunks = _chunks_in_selection(
            slab_indices,
            slab_offsets,
            mappers,
            filter=lambda slab_idx: slab_idx > 0,
            idxidx=False,
            sort_by_slab=False,
        )
        nchunks = chunks.shape[0]
        ndim = chunks.shape[1] - 2

        for i in range(nchunks):
            dst_ndslice = []
            for j in range(ndim):
                chunk_idx = chunks[i, j]
                dst_ndslice.append(dst_slices[j][chunk_idx])

            chunk_idx = chunks[i, 0]
            slab_idx = chunks[i, ndim]  # slab_indices[chunk_idx]
            start = chunks[i, ndim + 1]  # slab_offsets[chunk_idx]
            stop = start + axis0_chunk_sizes[chunk_idx]
            slab_ndslice = [slice(start, stop, 1)]
            for j in range(1, ndim):
                chunk_idx = chunks[i, j]
                slab_ndslice.append(slab_slices[j][chunk_idx])

            self.chunks.append((tuple(dst_ndslice), slab_idx, tuple(slab_ndslice)))

    @property
    def head(self) -> str:
        return f"ChangesPlan<{len(self.chunks)} chunks>"

    def __repr__(self) -> str:
        s = self.head
        fmt = format_ndindex
        for base_slice, slab_idx, slab_slice in self.chunks:
            s += f"\n  base[{fmt(base_slice)}] = slabs[{slab_idx}][{fmt(slab_slice)}]"
        return s


@cython.cclass
@dataclass(init=False, repr=False)
class ResizePlan(MutatingPlan):
    """Instructions to execute StagedChangesArray.resize()"""

    def __init__(
        self,
        old_shape: tuple[int, ...],
        new_shape: tuple[int, ...],
        chunk_size: tuple[int, ...],
        slab_indices: NDArray[np_hsize_t],
        slab_offsets: NDArray[np_hsize_t],
        n_slabs: int,
        n_base_slabs: int,
    ):
        """Generate instructions to execute StagedChangesArray.resize().

        Parameters
        ----------
        old_shape:
            StagedChangesArray.shape before the resize operation
        new_shape:
            StagedChangesArray.shape after the resize operation

        All other parameters are the matching attributes of StagedChangesArray.
        """
        if len(new_shape) != len(old_shape):
            raise ValueError(
                "Can't change dimensionality from {old_shape} to {new_shape}"
            )
        if any(s < 0 for s in new_shape):
            raise ValueError("shape must be non-negative")

        super().__init__(slab_indices, slab_offsets)
        if old_shape == new_shape:
            return

        # Shrinking along an axis can't alter chunks on the slabs; however, partial edge
        # chunks should be loaded into memory to avoid ending up with partially
        # overlapping chunks on disk, e.g. [10:19] vs. [10:17].
        # It can also reduce the amount of chunks impacted if we are also enlarging
        # along other axes, so it should be done first.
        # Finally, it can cause slabs to drop.
        shrunk_shape = tuple(
            min(o, n) for o, n in zip(old_shape, new_shape, strict=True)
        )
        if shrunk_shape != old_shape:
            chunks_slice = tuple(
                slice(ceil_a_over_b(s, c))
                for s, c in zip(shrunk_shape, chunk_size, strict=True)
            )
            # Just a view. This won't change the shape of the arrays when shrinking the
            # edge chunks without reducing the number of chunks.
            self.slab_indices = self.slab_indices[chunks_slice]
            self.slab_offsets = self.slab_offsets[chunks_slice]

            # Load partial edge chunks into memory to avoid ending up with partially
            # overlapping chunks on disk, e.g. [10:19] vs. [10:17].
            for axis, (old_size, shrunk_size, c) in enumerate(
                zip(old_shape, shrunk_shape, chunk_size, strict=True)
            ):
                if old_size > shrunk_size and shrunk_size % c != 0:
                    self._shrink_along_axis(
                        new_shape=new_shape,
                        chunk_size=chunk_size,
                        axis=axis,
                        n_slabs=n_slabs + len(self.append_slabs),
                        n_base_slabs=n_base_slabs,
                    )

        chunks_dropped = self.slab_indices.size < slab_indices.size

        if shrunk_shape != new_shape:
            # Enlarging along one or more axes. This is more involved than shrinking, as
            # we may need to potentially load and then update edge chunks.

            # If we're actually adding chunks, and not just resizing the edge chunks,
            # we need to enlarge slab_indices and slab_offsets too.
            pad_width = [
                (0, ceil_a_over_b(s, c) - n)
                for s, c, n in zip(
                    new_shape, chunk_size, self.slab_indices.shape, strict=True
                )
            ]
            # np.pad is a deep-copy; skip if unnecessary.
            if any(p != (0, 0) for p in pad_width):
                self.slab_indices = np.pad(self.slab_indices, pad_width)
                self.slab_offsets = np.pad(self.slab_offsets, pad_width)

            # No need to transfer anything if there are only full chunks
            if n_slabs + len(self.append_slabs) > 1:
                prev_shape = shrunk_shape
                for axis in range(len(new_shape)):
                    next_shape = new_shape[: axis + 1] + prev_shape[axis + 1 :]
                    if next_shape != prev_shape:
                        self._enlarge_along_axis(
                            old_shape=prev_shape,
                            new_shape=next_shape,
                            chunk_size=chunk_size,
                            axis=axis,
                            n_slabs=n_slabs + len(self.append_slabs),
                            n_base_slabs=n_base_slabs,
                        )
                        prev_shape = next_shape

        # Shrinking may drop any slab. Crucially, they may be staged slabs, and
        # dereferencing them means releasing memory.
        # Enlarging may only drop base slabs. Let's not do that, for the same
        # reason we don't do it in __setitem__: an all-too-common pattern is to
        # enlarge over and over again by one or a few points.
        if chunks_dropped:
            # This may set to None again slabs that were already None. That's fine.
            # On the upside, it also cleans up slabs dereferenced by __setitem__
            # or by enlarge operations.
            self.drop_slabs = np.setdiff1d(
                np.arange(1, n_slabs, dtype=np_hsize_t),  # Never drop the full slab
                np.unique(self.slab_indices),  # fairly expensive
                assume_unique=True,
            ).tolist()

    @cython.cfunc
    def _shrink_along_axis(
        self,
        new_shape: tuple[int, ...],
        chunk_size: tuple[int, ...],
        axis: ssize_t,
        n_slabs: int,
        n_base_slabs: int,
    ):
        """Shrink along a single axis.

        Load partial edge chunks into memory to avoid ending up with partially
        overlapping chunks on disk, e.g. [10:19] vs. [10:17].
        """
        new_size = new_shape[axis]
        new_floor_size = new_size - new_size % chunk_size[axis]
        assert new_floor_size < new_size

        self._load_edge_chunks_along_axis(
            shape=new_shape,
            chunk_size=chunk_size,
            axis=axis,
            floor_size=new_floor_size,
            size=new_size,
            n_slabs=n_slabs,
            n_base_slabs=n_base_slabs,
        )

    @cython.cfunc
    def _enlarge_along_axis(
        self,
        old_shape: tuple[int, ...],
        new_shape: tuple[int, ...],
        chunk_size: tuple[int, ...],
        axis: ssize_t,
        n_slabs: int,
        n_base_slabs: int,
    ):
        """Enlarge along a single axis"""
        old_size = old_shape[axis]

        # Old size, rounded down to the nearest chunk
        old_floor_size = old_size - old_size % chunk_size[axis]
        if old_floor_size == old_size:
            return  # Everything we're doing is adding extra empty chunks

        new_size = min(new_shape[axis], old_floor_size + chunk_size[axis])

        # Two steps:
        # 1. Find edge chunks on base slabs that were partial and became full, or
        #    vice versa, or remain partial but larger.
        #    Load them into a new slab at slab_idx=n_slabs.
        # 2. Find edge chunks that need filling with fill_value
        #    and transfer from slabs[0] (the full chunk) into them.
        #    This includes chunks we just transferred in the previous step)

        # Step 1
        self._load_edge_chunks_along_axis(
            shape=new_shape,
            chunk_size=chunk_size,
            axis=axis,
            floor_size=old_floor_size,
            size=old_size,
            n_slabs=n_slabs,
            n_base_slabs=n_base_slabs,
        )

        # Step 2
        idx = (slice(None),) * axis + (slice(old_size, new_size),)
        _, mappers = index_chunk_mappers(idx, new_shape, chunk_size)
        if not mappers:
            return  # Resizing from size 0

        chunks = _chunks_in_selection(
            self.slab_indices,
            self.slab_offsets,
            mappers,
            filter=lambda slab_idx: slab_idx > 0,
            idxidx=True,
        )
        nchunks = chunks.shape[0]

        if nchunks > 0:
            ndim = chunks.shape[1] - 2
            assert chunks[0, ndim] > n_base_slabs
            self.transfers.extend(
                _make_transfer_plans(
                    mappers,
                    chunks,
                    src_slab_idx=0,
                    dst_slab_idx="chunks",
                )
            )

    @cython.cfunc
    def _load_edge_chunks_along_axis(
        self,
        shape: tuple[int, ...],
        chunk_size: tuple[int, ...],
        axis: int,
        floor_size: int,
        size: int,  # Not necessarily shape[axis]
        n_slabs: int,
        n_base_slabs: int,
    ):
        """Load the edge chunks along an axis from the base slabs into a new slab"""
        if n_base_slabs == 0:
            return

        idx = (slice(None),) * axis + (slice(floor_size, size),)
        _, mappers = index_chunk_mappers(idx, shape, chunk_size)
        if not mappers:
            return  # Resizing to size 0

        chunks = _chunks_in_selection(
            self.slab_indices,
            self.slab_offsets,
            mappers,
            filter=lambda slab_idx: (0 < slab_idx) & (slab_idx <= n_base_slabs),
            idxidx=True,
        )
        nchunks = chunks.shape[0]
        if nchunks > 0:
            self.append_slabs.append((nchunks * chunk_size[0],) + chunk_size[1:])
            self.transfers.extend(
                _make_transfer_plans(
                    mappers,
                    chunks,
                    src_slab_idx="chunks",
                    dst_slab_idx=n_slabs,
                    slab_indices=self.slab_indices,  # Modified in place
                    slab_offsets=self.slab_offsets,  # Modified in place
                )
            )

    @property
    def head(self) -> str:
        return "ResizePlan<" + super().head


@cython.ccall
def _chunks_in_selection(
    slab_indices: NDArray[np_hsize_t],
    slab_offsets: NDArray[np_hsize_t],
    mappers: list[IndexChunkMapper],
    filter: Callable[[NDArray[np_hsize_t]], NDArray[np.bool_]] | None = None,
    idxidx: bint = False,
    sort_by_slab: bint = True,
    only_partial: bint = False,
) -> hsize_t[:, :]:
    """Find all chunks within a selection

    Parameters
    ----------
    slab_indices:
        StagedChangesArray.slab_indices
    slab_offsets:
        StagedChangesArray.slab_offsets
    mappers:
        Output of index_chunk_mappers()
    filter: optional
        Function to apply to a (selection of) slab_indices that returns a boolean mask.
        Chunks where the mask is False won't be returned.
    only_partial: optional
        If True, only return the chunks which are only partially covered by the
        selection along one or more axes.
    sort_by_slab: optional
        True (default)
            The chunks are returned sorted by slab index first and slaf offset next.
            Sorting by slab index will later allow cutting the result into separate
            calls to read_many_chunks() later. Sorting by slan offset within eacn slab
            is to improve access performance on disk-backed storage, where accessing two
            adjacent chunks will typically avoid a costly seek() syscall and may benefit
            from glibc-level and/or OS-level buffering.
        False
            The chunks are returned sorted by chunk index
    idxidx: optional
        If set to True, return the indices along chunk_indices instead of the values

    Returns
    -------
    Flattened sequence of chunks that match both the selection performed by the mappers
    along each axis and the mask performed by the filter on each individual chunk.

    The returned object is a 2D view of indices where each row corresponds to a chunk
    and ndim+2 columns:

    - columns 0:ndim are the chunk indices,
      or the indices to chunk_indices if idxidx=True.
      Multiply by chunk_size to get the top-left corner of the chunk
      within the virtual array.
    - column ndim is the slab index
      (point of slab_indices; the index within StagedChangesArray.slabs)
    - column ndim+1 is the slab offset
      (point of slab_offsets, the offset on axis 0 within the slab)

    **Examples**

    >>> index = (slice(5, 20), slice(15, 45))
    >>> chunk_size = (10, 10)
    >>> shape = (30, 60)
    >>> slab_indices = np.array(
    ...   [[0, 0, 1, 2, 0, 0],
    ...    [0, 0, 0, 0, 1, 0],
    ...    [0, 0, 2, 0, 1, 0]],
    ...   dtype=np_hsize_t
    ... )
    >>> slab_offsets = np.array(
    ...   [[0, 0, 50, 0,  0, 0],
    ...    [0, 0,  0, 0, 30, 0],
    ...    [0, 0, 60, 0,  0, 0]],
    ...   dtype=np_hsize_t
    ... )
    >>> _, mappers = index_chunk_mappers(index, shape, chunk_size)
    >>> tuple(list(m.chunk_indices) for m in mappers)
    ([0, 1], [1, 2, 3, 4])
    >>> tuple(m.chunks_indexer() for m in mappers)
    (slice(0, 2, 1), slice(1, 5, 1))
    >>> tuple(m.whole_chunks_idxidx() for m in mappers)
    (slice(1, 2, 1), slice(1, 3, 1))
    >>> np.asarray(_chunks_in_selection(
    ...     slab_indices, slab_offsets, mappers, sort_by_slab=False
    ... ))
    array([[ 0,  1,  0,  0],
           [ 0,  2,  1, 50],
           [ 0,  3,  2,  0],
           [ 0,  4,  0,  0],
           [ 1,  1,  0,  0],
           [ 1,  2,  0,  0],
           [ 1,  3,  0,  0],
           [ 1,  4,  1, 30]], dtype=uint64)
    >>> np.asarray(_chunks_in_selection(
    ...     slab_indices, slab_offsets, mappers, sort_by_slab=False, only_partial=True
    ... ))
    array([[ 0,  1,  0,  0],
           [ 0,  2,  1, 50],
           [ 0,  3,  2,  0],
           [ 0,  4,  0,  0],
           [ 1,  1,  0,  0],
           [ 1,  4,  1, 30]], dtype=uint64)

    chunk_indices        = ([0, 1], [1, 2, 3, 4])
    whole chunks_indices = ([0   ], [   2, 3   ])

    A chunk is partially selected if its coordinates are in the intersection of the
    indices across all axes, but they are not wholly selected along all axes.

    slab_indices  slab_offsets  selection  only_partial=False  only_partial=True
    001200        0 0 50 0 0 0  .pppp.     .xxxx.              .xxxx.
    000010        0 0 0 0 30 0  .pwwp.     .xxxx.              .x..x.
    002010        0 0 60 0 0 0  ......     ......              ......

                                p=partial  (0, 1) [0][ 0:]     (0, 1) [0][ 0:]
                                w=whole    (0, 2) [1][50:]     (0, 2) [1][50:]
                                           (0, 3) [2][ 0:]     (0, 3) [2][ 0:]
                                           (0, 4) [0][ 0:]     (0, 4) [0][ 0:]
                                           (1, 0) [0][ 0:]
                                           (1, 1) [0][ 0:]     (1, 1) [0][ 0:]
                                           (1, 2) [0][ 0:]
                                           (1, 3) [1][30:]     (1, 3) [1][30:]
    """
    mapper: IndexChunkMapper  # noqa: F841
    ndim = len(mappers)
    assert ndim > 0

    if only_partial:
        # A partial chunk is a chunk that is selected by chunks_indexer() along all
        # axes, but it is not selected by whole_chunks_idxidx() along at least one axis
        any_partial = False
        whole_chunks_idxidx = []
        for mapper in mappers:
            wcidx = mapper.whole_chunks_idxidx()
            whole_chunks_idxidx.append(wcidx)
            if not isinstance(wcidx, slice):
                any_partial = True
            elif wcidx != slice(0, len(mapper.chunk_indices), 1):
                any_partial = True

        if not any_partial:
            # All chunks are wholly selected
            return np.empty((0, ndim + 2), dtype=np_hsize_t)

    # Slice slab_indices and slab_offsets with mappers
    indexers = ix_with_slices(
        *[mapper.chunks_indexer() for mapper in mappers], shape=slab_indices.shape
    )
    slab_indices = slab_indices[indexers]
    slab_offsets = slab_offsets[indexers]

    if filter:
        mask = filter(slab_indices)
    else:
        mask = np.broadcast_to(True, slab_indices.shape)

    if only_partial:
        whole_chunks_idxidx_tup = ix_with_slices(*whole_chunks_idxidx, shape=mask.shape)
        if not filter:
            mask = mask.copy()  # broadcasted
        mask[whole_chunks_idxidx_tup] = False

    # Apply mask and flatten
    nz = np.nonzero(mask)
    slab_indices = slab_indices[nz]
    slab_offsets = slab_offsets[nz]

    if idxidx:
        # Don't copy when converting from np.intp to uint64 on 64-bit platforms
        columns = [asarray(nz_i, dtype=np_hsize_t) for nz_i in nz]
    else:
        columns = [
            np.asarray(mapper.chunk_indices)[nz_i]
            for mapper, nz_i in zip(mappers, nz, strict=True)
        ]
    columns += [slab_indices, slab_offsets]

    stacked = np.empty((slab_indices.size, ndim + 2), dtype=np_hsize_t)

    if sort_by_slab:
        sort_idx = np.lexsort((slab_offsets, slab_indices))
        for i, col in enumerate(columns):
            col.take(sort_idx, axis=0, out=stacked[:, i])
    else:
        for i, col in enumerate(columns):
            stacked[:, i] = col

    return stacked


def _make_transfer_plans(
    mappers: list[IndexChunkMapper],
    chunks: hsize_t[:, :],
    *,
    src_slab_idx: Literal[0, "chunks", None],
    dst_slab_idx: int | Literal["chunks", None],
    slab_indices: NDArray[np_hsize_t] | None = None,
    slab_offsets: NDArray[np_hsize_t] | None = None,
) -> Iterator[TransferPlan]:
    """Generate one or more TransferPlan, one for each pair of source and destination
    slabs.

    Parameters
    ----------
    slab_indices:
        StagedChangesArray.slab_indices. It will be updated in place.
        Ignored if dst_slab_idx is None.
    slab_offsets:
        StagedChangesArray.slab_offsets. It will be updated in place.
        Ignored if dst_slab_idx is None.
    mappers:
        List of IndexChunkMapper objects, one for each axis, defining the selection.
    chunks:
        Output of _chunks_in_selection(..., idxidx=True).
        One row per chunk to transfer.
        The first ndim columns are the indices to mappers[j].chunk_indices.
        for the i-th chunk; in other words,
        mappers[j].chunk_indices[chunk_idxidx[i, j]] * chunk_size[j] is the address
        along axis j of the top-left corner of the chunk in the virtual dataset.

        The last two columns are the slab_idx and slab offset, either source or
        destination depending on the other parameters.
    src_slab_idx:
        Index of the source slab in StagedChangesArray.slabs.
        int
            All transfers are from a single slab with this index.
        "chunks"
            Transfers are from the slabs pointed to by the slab_idx column in chunks.
        None
            Transfers are from the value parameter of __setitem__.
    dst_slab_idx:
        Index of the destination slab in StagedChangesArray.slabs.
        int
            All transfers are to a single slab with this index.
            This is used when filling a brand new slab.
        "chunks"
            Transfers are to the slabs pointed to by the slab_idx column in chunks.
            src_slab_idx can't be "chunks".
        None
            Transfers are to the return value of __getitem__.
            src_slab_idx can't be None.
    """
    nchunks = chunks.shape[0]
    ndim = chunks.shape[1] - 2
    chunk_size = mappers[0].chunk_size

    src_slab_offsets_v: hsize_t[:]
    dst_slab_offsets_v: hsize_t[:]

    if src_slab_idx is None:  # __setitem__
        src_slab_offsets_v = chunks[:0, 0]  # dummy; ignored
    elif src_slab_idx == 0:  # fill_value
        src_slab_offsets_v = np.zeros(nchunks, dtype=np_hsize_t)
    elif src_slab_idx == "chunks":
        src_slab_offsets_v = chunks[:, ndim + 1]
    else:  # pragma: nocover
        raise ValueError(f"Invalid {src_slab_idx=}")

    if dst_slab_idx is None:  # __getitem__
        assert src_slab_idx is not None
        dst_slab_offsets_v = chunks[:0, 0]  # dummy; ignored
    elif isinstance(dst_slab_idx, int):  # new slab
        assert dst_slab_idx > 0
        dst_slab_offsets_v = np.arange(
            0, nchunks * chunk_size, chunk_size, dtype=np_hsize_t
        )
    elif dst_slab_idx == "chunks":
        assert src_slab_idx != "chunks"
        dst_slab_offsets_v = chunks[:, ndim + 1]
    else:  # pragma: nocover
        raise ValueError(f"Invalid {dst_slab_idx=}")

    split_indices: ssize_t[:]
    if src_slab_idx == "chunks" or dst_slab_idx == "chunks":
        # Either multiple source slabs and single destination slab, or vice versa
        # Note that _chunks_in_selection returns chunks sorted by slab_idx.
        split_indices = np.flatnonzero(np.diff(chunks[:, ndim])) + 1
    else:
        # Single source and destination slabs
        split_indices = np.empty(0, dtype=np.intp)

    nsplits = len(split_indices)
    for i in range(nsplits + 1):
        start = 0 if i == 0 else split_indices[i - 1]
        stop = nchunks if i == nsplits else split_indices[i]

        slab_idx_i = int(chunks[start, ndim])
        src_slab_idx_i = slab_idx_i if src_slab_idx == "chunks" else src_slab_idx
        dst_slab_idx_i = slab_idx_i if dst_slab_idx == "chunks" else dst_slab_idx
        chunk_idx_group = chunks[start:stop, :ndim]
        src_slab_offsets_group = src_slab_offsets_v[start:stop]
        dst_slab_offsets_group = dst_slab_offsets_v[start:stop]

        yield TransferPlan(
            mappers,
            src_slab_idx=src_slab_idx_i,
            dst_slab_idx=dst_slab_idx_i,
            chunk_idxidx=chunk_idx_group,
            src_slab_offsets=src_slab_offsets_group,
            dst_slab_offsets=dst_slab_offsets_group,
            slab_indices=slab_indices,
            slab_offsets=slab_offsets,
        )


def _set_edges(
    dst: MutableArrayProtocol, src: ArrayProtocol, shape: tuple[int, ...]
) -> None:
    """Copy src into dst, but only for the edge area outside the given shape
    (aligned to the top left corner).

    This is equivalent to::

        mask = np.ones(dst.shape, dtype=bool)
        mask[*(slice(s) for s in shape)] = False
        dst[mask] = src[mask]

    except that the above would be ~O(dst.size) whereas this function is ~O(edge size).
    """
    assert src.shape == dst.shape
    assert len(shape) == dst.ndim
    for i, (start, stop) in enumerate(zip(shape, dst.shape, strict=True)):
        if stop > start:
            idx = tuple(slice(s) for s in shape[:i]) + (slice(start, stop),)
            dst[idx] = src[idx]
