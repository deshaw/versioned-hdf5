"""
Wrappers of h5py objects that work in memory

Much of this code is modified from code in h5py. See the LICENSE file for the
h5py license.
"""

from __future__ import annotations

import abc
import math
import posixpath
import textwrap
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable
from contextlib import suppress
from functools import cached_property
from typing import Any, ClassVar, Generic, TypeVar
from weakref import WeakValueDictionary

import numpy as np
from h5py import Dataset, Empty, Group, h5a, h5d, h5g, h5i, h5r, h5s, h5t, string_dtype
from h5py._hl.base import guess_dtype, phil, with_phil
from h5py._hl.selections import guess_shape
from ndindex import Slice, Tuple, ndindex
from numpy.typing import ArrayLike, DTypeLike

from versioned_hdf5.backend import (
    DEFAULT_CHUNK_SIZE,
    Filters,
    are_compatible_dtypes,
    is_vstring_dtype,
)
from versioned_hdf5.h5py_compat import HAS_NPYSTRINGS, h5py_astype
from versioned_hdf5.slicetools import build_slab_indices_and_offsets
from versioned_hdf5.staged_changes import StagedChangesArray
from versioned_hdf5.tools import NP_VERSION, asarray
from versioned_hdf5.typing_ import DEFAULT, Default, MutableArrayProtocol

try:
    from numpy.exceptions import AxisError  # numpy >=1.25
except ModuleNotFoundError:
    from numpy import AxisError  # numpy 1.24

T = TypeVar("T")


class InMemoryGroup(Group):
    _instances: ClassVar[WeakValueDictionary[h5g.GroupID, InMemoryGroup]] = (
        WeakValueDictionary({})
    )

    _subgroups: dict[str, InMemoryGroup]
    _data: dict[str, DatasetLike]
    _chunks: defaultdict[str, tuple[int, ...] | None]
    _filters: defaultdict[str, Filters]
    _parent: InMemoryGroup | None
    _initialized: bool
    _committed: bool

    def __new__(cls, bind: h5g.GroupID, _committed: bool = False):
        # Make sure each group only corresponds to one InMemoryGroup instance.
        # Otherwise a new instance would lose track of any datasets or
        # subgroups created in the old one.
        if bind in cls._instances:
            return cls._instances[bind]
        obj = super().__new__(cls)
        obj._initialized = False
        cls._instances[bind] = obj
        return obj

    def __init__(self, bind: h5g.GroupID, _committed: bool = False):
        """Create a new InMemoryGroup object by binding to a low-level GroupID.

        Parameters
        ----------
        bind : h5g.GroupID
            Low-level GroupID to bind to
        _committed : bool
            True if the group has already been committed, False otherwise.
        """
        if self._initialized:
            return
        self._subgroups = {}
        self._data = {}
        self._chunks = defaultdict(type(None))
        self._filters = defaultdict(Filters)
        self._parent = None
        self._initialized = True
        self._committed = _committed
        super().__init__(bind)

    def close(self):
        """Mark self and any subgroups as committed."""
        self._committed = True
        for name in self:
            obj = self[name]
            if isinstance(obj, InMemoryGroup):
                obj.close()

    # Based on Group.__repr__
    def __repr__(self):
        namestr = f'"{self.name}"' if self.name is not None else "(anonymous)"
        if not self:
            return "<Closed InMemoryGroup>"
        if self._committed:
            return f"<Committed InMemoryGroup {namestr}>"

        text = [f"<InMemoryGroup {namestr} ({len(list(self))} members)>"]
        for item in self.values():
            text.append(textwrap.indent(repr(item), prefix="  "))
        return "\n".join(text)

    def _check_committed(self):
        if self._committed:
            namestr = f'"{self.name}"' if self.name is not None else "(anonymous)"
            raise ValueError(f"InMemoryGroup {namestr} has already been committed")

    def __getitem__(self, name: str) -> InMemoryGroup | DatasetLike:
        dirname, basename = posixpath.split(name)
        if dirname:
            group = self[dirname]
            assert isinstance(group, InMemoryGroup)
            return group[basename]

        if name in self._data:
            return self._data[name]
        if name in self._subgroups:
            return self._subgroups[name]

        # If the name doesn't exist in self._data (the "in-memory"
        # part of InMemoryGroup), retrieve it from the actual underlying
        # h5py.Group, (i.e. the file itself).
        res = super().__getitem__(name)
        if isinstance(res, Group):
            self._subgroups[name] = self.__class__(res.id)
            return self._subgroups[name]
        if isinstance(res, Dataset):
            self._add_to_data(name, res)
            return self._data[name]
        raise NotImplementedError(f"Cannot handle {type(res)!r}")

    def __setitem__(self, name, obj):
        self._check_committed()
        self._add_to_data(name, obj)

    def _add_to_data(self, name, obj):
        dirname, basename = posixpath.split(name)
        if dirname:
            if dirname not in self:
                self.create_group(dirname)
            self[dirname][basename] = obj
            return

        if isinstance(obj, Dataset):
            wrapped_dataset = self._data[name] = DatasetWrapper(
                InMemoryDataset(obj.id, parent=self)
            )
            raw_data = wrapped_dataset.dataset.id.raw_data
            self._set_filters(name, Filters.from_dataset(raw_data))
        elif isinstance(obj, Group):
            self._subgroups[name] = InMemoryGroup(obj.id)
        elif isinstance(obj, InMemoryGroup):
            self._subgroups[name] = obj
        elif isinstance(obj, DatasetLike):
            self._data[name] = obj
            if isinstance(obj, DatasetWrapper) and isinstance(obj.dataset, Dataset):
                raw_data = obj.dataset.id.raw_data
                self._set_filters(name, Filters.from_dataset(raw_data))
        else:
            self._data[name] = DatasetWrapper(
                InMemoryArrayDataset(name, np.asarray(obj), parent=self)
            )

    def __delitem__(self, name):
        self._check_committed()
        dirname, basename = posixpath.split(name)
        if dirname:
            if not basename:
                del self[dirname]
            else:
                del self[dirname][basename]
            return

        if name in self._data:
            del self._data[name]
        elif name in self._subgroups:
            for i in self[name]:
                del self[name][i]
            del self._subgroups[name]
            super().__delitem__(name)
        else:
            raise KeyError(f"{name!r} is not in {self}")

    @property
    def parent(self):
        if self._parent is None:
            return super().parent
        return self._parent

    @parent.setter
    def parent(self, p):
        self._parent = p

    def create_group(self, name, track_order=None):
        self._check_committed()
        if name.startswith("/"):
            raise ValueError(
                "Root level groups cannot be created inside of versioned groups"
            )
        group = type(self)(super().create_group(name, track_order=track_order).id)
        g = group
        n = name
        while n:
            dirname, basename = posixpath.split(n)
            if not dirname:
                parent = self
            else:
                parent = type(self)(g.parent.id)
            parent._subgroups[basename] = g
            g.parent = parent
            g = parent
            n = dirname
        return group

    def create_dataset(
        self,
        name: str,
        shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        data: ArrayLike | None = None,
        fillvalue: Any | None = None,
        chunks: tuple[int, ...] | int | bool | None = None,
        # Filters
        compression: Any | None | Default = DEFAULT,
        compression_opts: Any | None | Default = DEFAULT,
        scaleoffset: int | None | Default = DEFAULT,
        shuffle: bool | Default = DEFAULT,
        fletcher32: bool | Default = DEFAULT,
        # Ignored (with a warning)
        **kwds: Any,
    ):
        self._check_committed()

        # Disregard ignored parameters with a warning
        for k, v in kwds.items():
            if v is not None:
                if k == "maxshape" and isinstance(v, tuple) and set(v) == {None}:
                    continue
                warnings.warn(
                    f"The {k} parameter is currently ignored for versioned datasets.",
                    stacklevel=2,
                )
        del kwds

        # In case of a nested path, call create_dataset on the leaf group
        dirname, basename = posixpath.split(name)
        if dirname:
            if dirname in self:
                group = self[dirname]
                assert isinstance(group, InMemoryGroup)
            else:
                group = self.create_group(dirname)
            return group.create_dataset(
                basename,
                shape=shape,
                dtype=dtype,
                data=data,
                fillvalue=fillvalue,
                chunks=chunks,
                compression=compression,
                compression_opts=compression_opts,
                scaleoffset=scaleoffset,
                shuffle=shuffle,
                fletcher32=fletcher32,
            )
        del name, dirname

        if dtype is not None and not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)

        if isinstance(data, Empty):
            raise NotImplementedError("Empty datasets are not supported.")

        if data is not None:
            if not isinstance(data, np.ndarray) or data.dtype.kind == "O":
                dtype = dtype or guess_dtype(data)
            data = np.asarray(data, order="C", dtype=dtype)
        if dtype is not None and dtype.kind == "T":
            dtype = string_dtype()  # Match plain h5py's behavior

        if shape is None:
            if data is None:
                raise TypeError("Either shape or data must be specified")
            shape = data.shape
        elif data is not None and data.shape != shape:
            raise ValueError(
                f"data.shape {data.shape} does not match specified shape {shape}"
            )

        if chunks in (True, None):
            if len(shape) == 1:
                chunks = (DEFAULT_CHUNK_SIZE,)
            else:
                raise NotImplementedError(
                    "chunks must be specified for multi-dimensional datasets"
                )
        elif chunks is False:
            raise ValueError("chunks=False is not supported for versioned datasets")
        elif isinstance(chunks, int) and not isinstance(chunks, bool):
            chunks = (chunks,)
        assert isinstance(chunks, tuple)

        if len(shape) != len(chunks):
            raise ValueError(
                f"Dimensions of chunks ({chunks}) must equal the dimensions of the "
                f"shape ({shape})"
            )
        if len(shape) == 0:
            raise NotImplementedError("Scalar datasets are not implemented.")

        ds: DatasetLike
        if data is not None:
            ds = InMemoryArrayDataset(
                basename,
                data,
                parent=self,
                fillvalue=fillvalue,
                chunks=chunks,
                # In case of dataset initialised from data with StringDType, have
                # ds._buffer.dtype.kind == "T" but outwards dtype must be object.
                dtype=dtype,
            )
            ds = DatasetWrapper(ds)
        else:
            ds = InMemorySparseDataset(
                basename,
                shape=shape,
                dtype=dtype,
                parent=self,
                fillvalue=fillvalue,
                chunks=chunks,
            )

        filters = Filters(
            compression=compression,
            compression_opts=compression_opts,
            scaleoffset=scaleoffset,
            shuffle=shuffle,
            fletcher32=fletcher32,
        )
        self._set_chunks(basename, chunks)
        self._set_filters(basename, filters)

        self[basename] = ds
        return ds

    def __iter__(self):
        names = list(self._data) + list(self._subgroups)
        for i in super().__iter__():
            if i in names:
                names.remove(i)
            yield i
        for i in names:
            yield i

    def __contains__(self, item):
        item = item + "/"
        root = self.versioned_root.name + "/"
        if item.startswith(root):
            item = item[len(root) :]
            if not item.rstrip("/"):
                return self == self.versioned_root
        item = item.rstrip("/")
        dirname, basename = posixpath.split(item)
        if dirname not in ["", "/"]:
            return dirname in self and basename in self[dirname]
        return any(i == item for i in self)

    def datasets(self):
        res = self._data.copy()

        def _get(name, item):
            if name in res:
                return
            if isinstance(item, (Dataset, DatasetLike, np.ndarray)):
                res[name] = item

        self.visititems(_get)

        return res

    @property
    def versioned_root(self) -> InMemoryGroup:
        p = self
        while p._parent is not None:
            p = p._parent
        return p

    def _recursion_to_root(
        self,
        dataset_name: str,
        cb: Callable[[InMemoryGroup, str], None],
    ) -> None:
        """Call `cb` on self and recursively on all parents up to the root.
        The callable must accept the node being currently visited and the name of the
        dataset relative to that node.
        """
        basename = posixpath.basename(dataset_name)
        full_name = basename
        p: InMemoryGroup | Group | None = self
        while isinstance(p, InMemoryGroup):
            cb(p, full_name)
            parent_basename = posixpath.basename(p.name)
            full_name = parent_basename + "/" + full_name
            p = p._parent

    def _set_chunks(self, dataset_name: str, value: tuple[int, ...] | None) -> None:
        def cb(node: InMemoryGroup, name: str) -> None:
            node._chunks[name] = value

        self._recursion_to_root(dataset_name, cb)

    def _set_filters(self, dataset_name: str, filters: Filters) -> None:
        def cb(node: InMemoryGroup, full_name: str) -> None:
            node._filters[full_name] = filters

        self._recursion_to_root(dataset_name, cb)

    def visititems(self, func):
        self._visit("", func)

    def _visit(self, prefix, func):
        for name in self:
            func(posixpath.join(prefix, name), self[name])
            if isinstance(self[name], InMemoryGroup):
                self[name]._visit(posixpath.join(prefix, name), func)

    # TODO: override other relevant methods here


class BufferMixin(abc.ABC):
    """Mixin for all staged datasets, handling dtype conversions.

    Special handling for NumPy StringDType, a.k.a. NpyStrings
    ---------------------------------------------------------
    In h5py, when you open or create a dataset with variable-width string dtype,
    regardless of whatever dtype= parameter you pass, it will result in a Dataset with
    object dtype. Then, you have to call `astype("T")` (a O(1) operation) to retrieve an
    AsTypeView and finally call `__getitem__` on the view, which directly reads from
    HDF5 into StringDType without passing by python object arrays.
    To write, you call `__setitem__` on the Dataset with object dtype, passing a
    NumPy array with StringDType. Again, there is ad-hoc h5py machinery that ensures
    that there is no intermediate conversion to python object arrays.

    This means that, in h5py, it is reasonably performant to create and destroy
    AsTypeView objects in rapid succession, and possibly interleave them with
    `__setitem__` calls::

        for i in range(ds.shape[0]):
            ds[i, 1] = ds.astype("T")[i, 0]

    We need to make sure that versioned-hdf5 is performant in the same use case.
    This means:
    - avoiding spurious intermediate conversions to python string arrays; and
    - avoiding loading/converting a whole array when only a slice is needed.

    To achieve this, methods `astype`, `__getitem__`, `__setitem__`, and `__array__`
    internally call `_maybe_swap_string_dtype`, which eagerly converts in-memory
    data and hot-swaps the base slab of the StagedChangesArray with a zero-cost
    AsTypeView.

    This means that if the user consistently reads and/or writes only StringDType
    arrays or only object string arrays, the whole machinery should be as performant
    as h5py, whereas if they interleave the two dtypes it will be very slow. This
    second use case should not be common.
    """

    _buffer: MutableArrayProtocol
    dtype: np.dtype  # Outwards dtype; not necessarily _buffer.dtype
    _swaps_counter: int = 0

    def _maybe_swap_string_dtype(self, dtype: np.dtype) -> None:
        if (
            not HAS_NPYSTRINGS
            or self._buffer.dtype == dtype
            or not (is_vstring_dtype(self._buffer.dtype) and is_vstring_dtype(dtype))
        ):
            return

        has_staged_changes = (
            self._buffer.n_staged_slabs > 0
            if isinstance(self._buffer, StagedChangesArray)
            else self._buffer.size > 0  # InMemoryArrayDataset
        )

        if has_staged_changes:
            self._swaps_counter += 1
            # A resize() to enlarge a freshly opened InMemoryDataset may load edge
            # chunks into memory as object dtype, which a later __getitem__, __setitem__
            # or astype() will then convert in memory to StringDType. No way to prevent
            # this (short of a config flag) as we don't know yet how the user will
            # access the dataset.
            if self._swaps_counter > 1:
                warnings.warn(
                    "Performing multiple internal conversions between object type and "
                    "StringDType in memory. This will result in poor performance. "
                    "You should use the same dtype for all reads and writes on the "
                    "dataset. Please make sure you're reading strings with "
                    "`ds.astype('T')[...]` and not with `ds[...].astype('T')`.",
                    UserWarning,
                    stacklevel=2,
                )

        self._buffer = self._astype_impl(dtype, writeable=True)

    def __array__(
        self,
        dtype: DTypeLike | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        out = self.astype(dtype or self.dtype)
        if copy and out is not self._buffer:
            copy = None
        if NP_VERSION < (2,):
            return np.array(out, copy=bool(copy))
        return np.asarray(out, copy=copy)

    def __getitem__(self, index) -> MutableArrayProtocol:
        self._maybe_swap_string_dtype(self.dtype)
        assert self._buffer.dtype == self.dtype  # Thanks to _maybe_swap_string_dtype
        return self._buffer[index]

    def __setitem__(self, index, value: ArrayLike) -> None:
        if hasattr(value, "dtype"):
            self._maybe_swap_string_dtype(value.dtype)
        self._buffer[index] = value

    def astype(self, dtype: DTypeLike) -> MutableArrayProtocol:
        dtype = np.dtype(dtype)
        self._maybe_swap_string_dtype(dtype)
        return self._astype_impl(dtype, writeable=False)

    @abc.abstractmethod
    def _astype_impl(self, dtype: np.dtype, writeable: bool) -> MutableArrayProtocol:
        """Return self._buffer as a new dtype. Return a view if possible."""


class FilterDescriptor(Generic[T]):
    """Compression or other filter property of a dataset.

    The data is stored on the parent MemoryGroup.
    """

    default: T
    name: str
    __slots__ = ("default", "name")

    def __init__(self, default: T):
        self.default = default

    def __set_name__(self, owner: type, name: str):
        self.name = name

    def __get__(self, instance: FiltersMixin | None, owner: type) -> T:
        if instance is None:
            return self  # type: ignore  # getattr on class. Called by Sphinx.
        basename = posixpath.basename(instance.name)
        filters = instance.parent._filters[basename]
        res = getattr(filters, self.name)
        return self.default if res is DEFAULT else res


class FiltersMixin:
    """Add properties for compression and other filters to datasets."""

    # See matching class backend.Filters
    name: str
    parent: InMemoryGroup

    compression: FilterDescriptor[Any | None] = FilterDescriptor(None)
    compression_opts: FilterDescriptor[Any | None] = FilterDescriptor(None)
    scaleoffset: FilterDescriptor[int | None] = FilterDescriptor(None)
    shuffle: FilterDescriptor[bool] = FilterDescriptor(False)
    fletcher32: FilterDescriptor[bool] = FilterDescriptor(False)


# Note: mixin methods override those from Dataset
class InMemoryDataset(BufferMixin, FiltersMixin, Dataset):
    """
    Class that looks like a h5py.Dataset but is backed by a versioned dataset

    The versioned dataset can be modified, which performs modifications
    in-memory only.
    """

    def __init__(self, bind, parent, *, readonly=False):
        # Hold a reference to the original bind so h5py doesn't invalidate the id
        # XXX: We need to handle deallocation here properly when our object
        # gets deleted or closed.
        self.orig_bind = bind
        Dataset.__init__(self, InMemoryDatasetID(bind.id), readonly=readonly)
        self._parent = parent
        self._attrs = dict(super().attrs)

    @cached_property
    def staged_changes(self) -> StagedChangesArray:
        dcpl = self.id.get_create_plist()
        slab_indices, slab_offsets = build_slab_indices_and_offsets(
            dcpl, self.id.shape, self.id.chunks
        )
        return StagedChangesArray(
            shape=self.id.shape,
            chunk_size=self.id.chunks,
            base_slabs=[self.id.raw_data],
            slab_indices=slab_indices,
            slab_offsets=slab_offsets,
            fill_value=self.fillvalue,
        )

    @property
    def _buffer(self) -> MutableArrayProtocol:
        """Hook for BufferMixin"""
        return self.staged_changes

    @_buffer.setter
    def _buffer(self, value: MutableArrayProtocol) -> None:
        """Hook for BufferMixin"""
        assert isinstance(value, StagedChangesArray)
        self.__dict__["staged_changes"] = value

    def _astype_impl(self, dtype: np.dtype, writeable: bool) -> MutableArrayProtocol:
        """Hook for BufferMixin"""
        # Backwards compatibility with h5py <3.13
        raw_data_view = h5py_astype(self.id.raw_data, dtype)  # AsTypeView
        out = self.staged_changes.astype(dtype, base_slabs=[raw_data_view])
        out.writeable = writeable
        return out

    def __repr__(self) -> str:
        name = posixpath.basename(posixpath.normpath(self.name))
        namestr = '"{}"'.format(name if name != "" else "/")
        return '<{} {}: shape {}, type "{}">'.format(
            self.__class__.__name__,
            namestr,
            self.shape,
            self.dtype.str,
        )

    @property
    def data_dict(self) -> dict[Tuple, Slice | np.ndarray]:
        return _staged_changes_to_data_dict(self.staged_changes)

    @property
    def dtype(self):
        """Override Dataset.dtype to allow hot-swapping
        equivalent dtypes, e.g. NpyStrings <-> object strings
        """
        return self.id.raw_data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.id.shape

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def chunks(self) -> tuple[int, ...]:
        return self.id.chunks

    @property
    def attrs(self):
        return self._attrs

    @property
    def parent(self) -> InMemoryGroup:  # type: ignore[override]
        return self._parent

    def resize(self, size, axis=None):
        """Resize the dataset, or the specified axis.

        The rank of the dataset cannot be changed.

        "size" should be a shape tuple, or if an axis is specified, an integer.

        BEWARE: This functions differently than the NumPy resize() method!
        The data is not "reshuffled" to fit in the new shape; each axis is
        grown or shrunk independently.  The coordinates of existing data are
        fixed.
        """
        self.parent._check_committed()
        new_shape = _normalize_resize_args(self.shape, size, axis)
        self.staged_changes.resize(new_shape)
        self.id.shape = new_shape

    @with_phil
    def __getitem__(
        self,
        args: (
            str
            | slice
            | tuple
            | list[int]
            | list[bool]
            | np.ndarray
            | h5r.RegionReference
        ),
    ) -> np.ndarray:
        """Read a slice from the HDF5 dataset given by the index.

        Takes slices and recarray-style field names (more than one is
        allowed!) in any order.  Obeys basic NumPy rules, including
        broadcasting.

        Parameters
        ----------
        args : any numpy one-dimensional or n-dimensional index | h5r.RegionReference
            Index to read from the Dataset.

            **Note:** more than one list/ndarray index will behave differently as numpy,
            as it will be interpreted to pick the given indices independently on each
            axis. Non-flat list/ndarray indices are not supported.

        Returns
        -------
        np.ndarray
            Array containing data from this dataset from the requested index
        """
        # This boilerplate code is based on h5py.Dataset.__getitem__
        args = args if isinstance(args, tuple) else (args,)

        # Sort field names from the rest of the args.
        names = tuple(x for x in args if isinstance(x, str))

        if names:
            # Read a subset of the fields in this structured dtype
            args = tuple(x for x in args if not isinstance(x, str))
            return self.fields(names[0] if len(names) == 1 else names)[args]

        # === Special-case region references ====

        if len(args) == 1 and isinstance(args[0], h5r.RegionReference):
            mtype = h5t.py_create(self.dtype)
            obj = h5r.dereference(args[0], self.id)
            if obj != self.id:
                raise ValueError("Region reference must point to this dataset")

            sid = h5r.get_region(args[0], self.id)
            mshape = guess_shape(sid)
            if mshape is None:
                # 0D with no data (NULL or deselected SCALAR)
                return Empty(self.dtype)
            out = np.empty(mshape, self.dtype)
            if out.size == 0:
                return out

            sid_out = h5s.create_simple(mshape)
            sid_out.select_all()
            self.id.read(sid_out, sid, out, mtype)
            return out

        # === END CODE FROM h5py.Dataset.__getitem__ ===
        return BufferMixin.__getitem__(self, args)

    @with_phil
    def __setitem__(self, args, val):
        """Write to the HDF5 dataset from a NumPy array.

        NumPy's broadcasting rules are honored, for "simple" indexing
        (slices and integers).  For advanced indexing, the shapes must
        match.
        """
        self.parent._check_committed()
        # This boilerplate code is based on h5py.Dataset.__setitem__
        args = args if isinstance(args, tuple) else (args,)

        # Sort field indices from the slicing
        names = tuple(x for x in args if isinstance(x, str))
        args = tuple(x for x in args if not isinstance(x, str))

        # Generally we try to avoid converting the arrays on the Python
        # side. However, for compound literals this is unavoidable.
        val_is_npystring = (
            HAS_NPYSTRINGS  # On NumPy >=2 but h5py <3.14, convert NpyStrings to object
            and hasattr(val, "dtype")
            and val.dtype.kind == "T"
        )
        vlen = h5t.check_vlen_dtype(self.dtype)
        if not val_is_npystring and vlen not in (None, bytes, str):
            try:
                val = np.asarray(val, dtype=vlen)
            except ValueError:
                with suppress(ValueError):
                    val = np.array(
                        [np.array(x, dtype=vlen) for x in val], dtype=self.dtype
                    )
            if vlen == val.dtype:
                if val.ndim > 1:
                    tmp = np.empty(shape=val.shape[:-1], dtype=object)
                    tmp.ravel()[:] = val.reshape(-1, val.shape[-1])
                else:
                    tmp = np.array([None], dtype=object)
                    tmp[0] = val
                val = tmp
        elif self.dtype.kind == "O" or (
            self.dtype.kind == "V"
            and (not isinstance(val, np.ndarray) or val.dtype.kind != "V")
            and (self.dtype.subdtype is None)
        ):
            if len(names) == 1 and self.dtype.fields is not None:
                # Single field selected for write, from a non-array source
                if names[0] not in self.dtype.fields:
                    raise ValueError(f"No such field for indexing: {names[0]}")
                dtype = self.dtype.fields[names[0]][0]
                cast_compound = True
            elif (
                hasattr(val, "dtype")
                and is_vstring_dtype(val.dtype)
                and is_vstring_dtype(self.staged_changes.dtype)
            ):
                dtype = val.dtype
                cast_compound = False
            else:
                dtype = self.staged_changes.dtype
                cast_compound = False

            val = np.asarray(val, dtype=dtype.base, order="C")
            if cast_compound:
                val = val.view(np.dtype([(names[0], dtype)]))
                val = val.reshape(val.shape[: len(val.shape) - len(dtype.shape)])
        else:
            val = np.asarray(val, dtype=self.staged_changes.dtype, order="C")

        # Check for array dtype compatibility and convert
        if self.dtype.subdtype is not None:
            shp = self.dtype.subdtype[1]
            valshp = val.shape[-len(shp) :]
            if valshp != shp:  # Last dimension has to match
                raise TypeError(
                    "When writing to array types, last N dimensions have to match "
                    f"(got {valshp}, but should be {shp})"
                )
            mtype = h5t.py_create(np.dtype((val.dtype, shp)))

        # Make a compound memory type if field-name slicing is required
        elif len(names) != 0:
            # Catch common errors
            if self.dtype.fields is None:
                raise TypeError("Illegal slicing argument (not a compound dataset)")
            mismatch = [x for x in names if x not in self.dtype.fields]
            if len(mismatch) != 0:
                mismatch = ", ".join(f'"{x}"' for x in mismatch)
                raise ValueError(
                    f"Illegal slicing argument (fields {mismatch} not in dataset type)"
                )

            # Write non-compound source into a single dataset field
            if len(names) == 1 and val.dtype.fields is None:
                subtype = h5t.py_create(val.dtype)
                mtype = h5t.create(h5t.COMPOUND, subtype.get_size())
                mtype.insert(self._e(names[0]), 0, subtype)

            # Make a new source type keeping only the requested fields
            else:
                fieldnames = [
                    x for x in val.dtype.names if x in names
                ]  # Keep source order
                mtype = h5t.create(h5t.COMPOUND, val.dtype.itemsize)
                for fieldname in fieldnames:
                    subtype = h5t.py_create(val.dtype.fields[fieldname][0])
                    offset = val.dtype.fields[fieldname][1]
                    mtype.insert(self._e(fieldname), offset, subtype)

        # Use mtype derived from array (let DatasetID.write figure it out)
        else:
            mtype = None

        # === END CODE FROM h5py.Dataset.__setitem__ ===
        BufferMixin.__setitem__(self, args, val)


class DatasetLike:
    """
    Superclass for classes that look like h5py.Dataset

    Subclasses should have the following properties defined (properties
    starting with an underscore will be computed if they are None)

    name
    shape
    dtype
    attrs
    _fillvalue
    parent (the parent group)
    """

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    attrs: dict[str, Any]
    _fillvalue: Any | None
    parent: InMemoryGroup

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def fillvalue(self) -> np.generic:
        fv = self._fillvalue
        if fv is None and self.dtype.metadata:
            # Custom h5py string dtype. Make sure to use a fillvalue of ''
            if (vlen := self.dtype.metadata.get("vlen")) is not None:
                fv = b"" if vlen is str else vlen()
            elif "h5py_encoding" in self.dtype.metadata:
                fv = b""

        if fv is not None:
            return np.asarray(fv, dtype=self.dtype)[()]
        return np.zeros((), dtype=self.dtype)[()]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __bool__(self) -> bool:
        return bool(self.size)

    def __len__(self) -> int:
        return self.len()

    def len(self) -> int:
        """Length of the first axis."""
        if len(self.shape) == 0:
            raise TypeError("Attempt to take len() of scalar dataset")
        return self.shape[0]

    def __repr__(self) -> str:
        name = posixpath.basename(posixpath.normpath(self.name))
        namestr = '"%s"' % (name if name != "" else "/")
        return '<{} {}: shape {}, type "{}">'.format(
            self.__class__.__name__,
            namestr,
            self.shape,
            self.dtype.str,
        )

    def __iter__(self) -> Iterable[np.ndarray | np.generic]:
        """Iterate over the first axis. TypeError if scalar.

        BEWARE: Modifications to the yielded data are *NOT* written to file.
        """
        shape = self.shape
        if len(shape) == 0:
            raise TypeError("Can't iterate over a scalar dataset")
        for i in range(shape[0]):
            yield self[i]  # type: ignore[index]


class InMemoryArrayDataset(BufferMixin, FiltersMixin, DatasetLike):
    """
    Class that looks like a h5py.Dataset but is backed by an array
    """

    def __init__(
        self,
        name: str,
        array: np.ndarray,
        *,
        parent: InMemoryGroup,
        fillvalue: Any | None = None,
        chunks: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        attrs: dict[str, Any] | None = None,
    ):
        if not array.flags.writeable:
            array = array.copy()
        self._buffer = array
        self.dtype = np.dtype(dtype) if dtype else array.dtype
        self.name = name
        self.attrs = attrs or {}
        self.parent = parent
        self._fillvalue = fillvalue
        if chunks is None:
            chunks = parent._chunks[name]
        self.chunks = chunks

    @property
    def shape(self) -> tuple[int, ...]:  # type: ignore[override]
        return self._buffer.shape

    def _astype_impl(self, dtype: np.dtype, writeable: bool) -> MutableArrayProtocol:
        """Hook for BufferMixin"""
        # Work around https://github.com/numpy/numpy/issues/28269
        # on NumPy >=2.0.0,<2.2.3 when converting from arrays of object strings to
        # NpyStrings
        out = asarray(self._buffer, dtype=dtype).view()
        out.flags.writeable = writeable
        return out

    def __setitem__(self, index, value):
        self.parent._check_committed()
        BufferMixin.__setitem__(self, index, value)

    def resize(self, size, axis=None):
        self.parent._check_committed()
        new_shape = _normalize_resize_args(self.shape, size, axis)
        if any(new > old for new, old in zip(new_shape, self.shape, strict=True)):
            raise AssertionError(  # pragma: nocover
                "Enlarging an InMemoryArrayDataset directly. "
                "This should be unreachable outside of artificial unit tests; "
                "see DatasetWrapper.resize()."
            )
        new_idx = tuple(slice(i) for i in new_shape)
        self._buffer = self._buffer[new_idx]


class InMemorySparseDataset(BufferMixin, FiltersMixin, DatasetLike):
    """
    Class that looks like a Dataset that has no data (only the fillvalue)
    """

    def __init__(self, name, *, shape, dtype, parent, chunks=None, fillvalue=None):
        if shape is None:
            raise TypeError("shape must be specified for sparse datasets")
        self.name = name
        self.attrs = {}
        self._fillvalue = fillvalue
        if chunks in [True, None]:
            if len(shape) == 1:
                chunks = (DEFAULT_CHUNK_SIZE,)
            else:
                raise NotImplementedError(
                    "chunks must be specified for multi-dimensional datasets"
                )
        self.parent = parent
        self.staged_changes = StagedChangesArray.full(
            shape=shape,
            chunk_size=tuple(chunks),
            dtype=dtype,
            fill_value=fillvalue,
        )
        # BufferMixin can later change self.staged_changes.dtype
        self.dtype = self.staged_changes.dtype

    @property
    def data_dict(self) -> dict[Tuple, Slice | np.ndarray]:
        return _staged_changes_to_data_dict(self.staged_changes)

    @property
    def shape(self):
        return self.staged_changes.shape

    @property
    def chunks(self) -> tuple[int, ...]:
        return self.staged_changes.chunk_size

    def _astype_impl(self, dtype: np.dtype, writeable: bool) -> MutableArrayProtocol:
        """Hook for BufferMixin"""
        out = self.staged_changes.astype(dtype)
        out.writeable = writeable
        return out

    @property
    def _buffer(self) -> MutableArrayProtocol:
        """Hook for BufferMixin"""
        return self.staged_changes

    @_buffer.setter
    def _buffer(self, value: MutableArrayProtocol) -> None:
        """Hook for BufferMixin"""
        assert isinstance(value, StagedChangesArray)
        self.staged_changes = value

    def __setitem__(self, index, value):
        self.parent._check_committed()
        BufferMixin.__setitem__(self, index, value)

    @classmethod
    def from_dataset(cls, dataset, parent=None):
        return cls(
            dataset.name,
            shape=dataset.shape,
            dtype=dataset.dtype,
            parent=parent or dataset.parent,
            chunks=dataset.chunks,
            fillvalue=dataset.fillvalue,
        )

    def resize(self, size, axis=None):
        new_shape = _normalize_resize_args(self.shape, size, axis)
        self.staged_changes.resize(new_shape)


def _staged_changes_to_data_dict(
    staged_changes: StagedChangesArray,
) -> dict[Tuple, Slice | np.ndarray]:
    """Transitional hack that converts a StagedChangsArray to a legacy data_dict.

    This was introduced when replacing the legacy system, which was wholly designed
    around the data_dict, with StagedChangesArray and it allowed not to modify from the
    get go all the code that is triggered upon commit.

    We intend to clean this up eventually.
    """
    # InMemoryDataset has exactly one raw_data buffer underlying
    # InMemorySparseDataset has none
    assert staged_changes.n_base_slabs < 2
    return {
        Tuple(*k): Slice(v[0]) if isinstance(v, tuple) else v
        for k, _, v in staged_changes.changes()
    }


def _normalize_resize_args(
    shape: tuple[int, ...],
    size: int | list[int] | tuple[int, ...] | np.ndarray,
    axis: int | None,
) -> tuple[int, ...]:
    """Normalize the parameters of Dataset.resize()"""
    ndim = len(shape)
    if axis is not None:
        if axis >= ndim or axis < -ndim:
            msg = f"axis {axis} is out of bounds for dataset of dimension {ndim}"
            raise AxisError(msg)
        try:
            size = int(size)  # type: ignore[arg-type]
        except TypeError:
            msg = f"size must be an integer when axis is specified, got {size!r}"
            raise TypeError(msg) from None

        new_shape = list(shape)
        new_shape[axis] = size
    else:
        if (
            isinstance(size, (list, tuple))
            or isinstance(size, np.ndarray)
            and size.ndim == 1
        ):
            new_shape = [int(i) for i in size]
        else:
            new_shape = [int(size)]
        if len(new_shape) != ndim:
            msg = f"Invalid shape {size} for dataset of dimension {ndim}"
            raise ValueError(msg)

    return tuple(new_shape)


class DatasetWrapper(DatasetLike):
    dataset: InMemoryDataset | InMemoryArrayDataset | InMemorySparseDataset

    def __init__(self, dataset):
        self.dataset = dataset

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __setitem__(self, index, value):
        if ndindex(index).expand(self.shape) != Tuple().expand(self.shape):
            # partial update. Just write to the buffer.
            self.dataset[index] = value
            return

        # index covers the whole dataset.
        # Hot-swap the current dataset with a InMemoryArrayDataset.
        # If the current dataset is already a InMemoryArrayDataset,
        # do not deep copy the data.
        buf_dtype = self.dataset._buffer.dtype
        if not hasattr(value, "dtype"):
            value = np.asarray(value, dtype=buf_dtype)
        value = np.broadcast_to(value, self.shape)
        # Don't convert value if value.dtype == StringDType and
        # buffer._dtype == object or the other way around.
        if not are_compatible_dtypes(buf_dtype, value.dtype):
            value = value.astype(buf_dtype)
        self.dataset = InMemoryArrayDataset(
            self.name,
            value,
            parent=self.parent,
            fillvalue=self.fillvalue,
            chunks=self.chunks,
            dtype=self.dtype,
            attrs=self.dataset.attrs,
        )

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def resize(self, size, axis=None):
        new_shape = _normalize_resize_args(self.dataset.shape, size, axis)
        if not isinstance(self.dataset, InMemoryArrayDataset) or all(
            new <= old for new, old in zip(new_shape, self.dataset.shape, strict=True)
        ):
            self.dataset.resize(new_shape)
            return

        # Enlarge an InMemoryArrayDataset.
        # Prevent a potentially very expensive use case where the user calls
        # resize() but then does not completely fill it up with data before they commit,
        # resulting in empty chunks that needlessly occupy RAM until the time they are
        # committed and all need to go through hashing.
        new_ds = InMemorySparseDataset(
            name=self.dataset.name,
            shape=self.dataset.shape,
            dtype=self.dataset.dtype,
            parent=self.dataset.parent,
            chunks=self.dataset.chunks,
            fillvalue=self.dataset.fillvalue,
        )
        # Use a writeable view of old buffer as the slabs, if geometry allows
        new_ds.staged_changes = StagedChangesArray.from_array(
            # Note: in case of variable-width strings, _buffer.dtype may be different
            # from dataset.dtype
            self.dataset._buffer,
            chunk_size=self.dataset.chunks,
            fill_value=self.dataset.fillvalue,
            as_base_slabs=False,
        )
        new_ds.resize(new_shape)
        self.dataset = new_ds


class InMemoryDatasetID(h5d.DatasetID):
    def __init__(self, _id):
        # super __init__ is handled by DatasetID.__cinit__ automatically
        with phil:
            sid = self.get_space()
            self._shape = sid.get_simple_extent_dims()

        attr = h5a.open(self, b"raw_data")
        htype = h5t.py_create(attr.dtype)
        _arr = np.ndarray(attr.shape, dtype=attr.dtype, order="C")
        attr.read(_arr, mtype=htype)
        raw_data_name = _arr[()]
        if isinstance(raw_data_name, bytes):
            raw_data_name = raw_data_name.decode("utf-8")

        fid = h5i.get_file_id(self)
        g = Group(fid)
        self.raw_data = g[raw_data_name]
        self.chunks = tuple(self.raw_data.attrs["chunks"])

        fillvalue_a = np.empty((1,), dtype=self.dtype)
        dcpl = self.get_create_plist()
        dcpl.get_fill_value(fillvalue_a)
        self.fillvalue = fillvalue_a[0]

    def set_extent(self, shape):
        raise NotImplementedError("Resizing an InMemoryDataset other than via resize()")

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, size):
        self._shape = size

    def write(self, mspace, fspace, arr_obj, mtype=None, dxpl=None):
        raise NotImplementedError(
            "Writing to an InMemoryDataset other than via __setitem__"
        )
