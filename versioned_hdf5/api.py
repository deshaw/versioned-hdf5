from h5py import Empty, Dataset, Group, h5d, h5s
from h5py._hl.selections import select
from h5py._hl.vds import VDSmap

import numpy as np

from contextlib import contextmanager
import datetime
import math

from .backend import initialize, CHUNK_SIZE
from .versions import (create_version, get_nth_previous_version,
                       set_current_version, all_versions)
from .slicetools import s2t, slice_size, split_slice

class VersionedHDF5File:
    """
    A Versioned HDF5 File

    This is the main entry-point of the library. To use a versioned HDF5 file,
    pass a h5py file to constructor. The methods on the resulting object can
    be used to view and create versions.

    Note that versioned HDF5 files have a special structure and should not be
    modified directly. Also note that once a version is created in the file,
    it should be treated as read-only. Some protections are in place to
    prevent accidental modification, but it is not possible in the HDF5 layer
    to make a dataset or group read-only, so modifications made outside of
    this library could result in breaking things.

    >>> import h5py
    >>> f = h5py.File('file.h5')
    >>> from versioned_hdf5 import VersionedHDF5File
    >>> file = VersionedHDF5File(f)

    Access versions using indexing

    >>> version1 = file['version1']

    This returns a group containing the datasets for that version.

    To create a new version, use :func:`stage_version`.

    >>> with file.stage_version('version2') as group:
    ...     group['dataset'] = ... # Modify the group
    ...

    When the context manager exits, the version will be written to the file.

    """
    def __init__(self, f):
        self.f = f
        if '_version_data' not in f:
            initialize(f)
        self._version_data = f['_version_data']
        self._versions = self._version_data['versions']

    @property
    def current_version(self):
        """
        The current version.

        The current version is used as the default previous version to
        :func:`stage_version`, and is also used for negative integer version
        indexing (the current version is `self[0]`).
        """
        return self._versions.attrs['current_version']

    @current_version.setter
    def current_version(self, version_name):
        set_current_version(self.f, version_name)

    def get_version_by_name(self, version):
        if version == '':
            version = '__first_version__'

        if version not in self._versions:
            raise KeyError(f"Version {version!r} not found")

        # TODO: Don't give an in-memory group if the file is read-only
        return InMemoryGroup(self._versions[version]._id)

    def __getitem__(self, item):
        if item is None:
            return self.get_version_by_name(self.current_version)
        elif isinstance(item, str):
            return self.get_version_by_name(item)
        elif isinstance(item, (int, np.integer)):
            if item > 0:
                raise IndexError("Integer version slice must be negative")
            return self.get_version_by_name(get_nth_previous_version(self.f,
                self.current_version, -item))
        elif isinstance(item, (datetime.datetime, np.datetime64)):
            raise NotImplementedError
        else:
            raise KeyError(f"Don't know how to get the version for {item!r}")

    def __iter__(self):
        return all_versions(self.f, include_first=False)

    @contextmanager
    def stage_version(self, version_name: str, prev_version=None, make_current=True):
        """
        Return a context manager to stage a new version

        The context manager returns a group, which should be modified in-place
        to build the new version. When the context manager exits, the new
        version will be written into the file.

        `version_name` should be the name for the version.

        `prev_version` should be the previous version which this version is
        based on. The group returned by the context manager will mirror this
        previous version. If it is `None` (the default), the previous
        version will be the current version. If it is `''`, there will be no
        previous version.

        If `make_current` is `True` (the default), the new version will be set
        as the current version. The current version is used as the default
        `prev_version` for any future `stage_version` call.

        """
        group = self[prev_version]
        yield group
        create_version(self.f, version_name, group.datasets(),
                       prev_version=prev_version, make_current=make_current)

class InMemoryGroup(Group):
    def __init__(self, bind):
        self._data = {}
        self._subgroups = {}
        super().__init__(bind)

    # Based on Group.__repr__
    def __repr__(self):
        if not self:
            r = u"<Closed InMemoryGroup>"
        else:
            namestr = (
                '"%s"' % self.name
            ) if self.name is not None else u"(anonymous)"
            r = '<InMemoryGroup %s (%d members)>' % (namestr, len(self))

        return r

    def __getitem__(self, name):
        if name in self._data:
            return self._data[name]
        if name in self._subgroups:
            return self._subgroups[name]

        res = super().__getitem__(name)
        if isinstance(res, Group):
            self._subgroups[name] = self.__class__(res.id)
            return self._subgroups[name]
        elif isinstance(res, Dataset):
            self._data[name] = InMemoryDataset(res.id)
            return self._data[name]
        else:
            raise NotImplementedError(f"Cannot handle {type(res)!r}")

    def __setitem__(self, name, obj):
        self._data[name] = obj

    def __delitem__(self, name):
        if name in self._data:
            del self._data
        super().__delitem__(name)

    def create_dataset(self, name, **kwds):
        data = super().create_dataset(name, **kwds)
        self[name] = data
        return data

    def datasets(self):
        res = self._data.copy()

        def _get(name, item):
            if name in res:
                return
            if isinstance(item, (Dataset, np.ndarray)):
                res[name] = item

        self.visititems(_get)

        return res

    #TODO: override other relevant methods here

class InMemoryDataset(Dataset):
    def __init__(self, bind, **kwargs):
        # Hold a reference to the original bind so h5py doesn't invalidate the id
        # XXX: We need to handle deallocation here properly when our object
        # gets deleted or closed.
        self.orig_bind = bind
        super().__init__(InMemoryDatasetID(bind.id), **kwargs)

class InMemoryDatasetID(h5d.DatasetID):
    def __init__(self, _id):
        # super __init__ is handled by DatasetID.__cinit__ automatically
        self.data_dict = {}

        dcpl = self.get_create_plist()
        # Same as dataset.get_virtual_sources
        virtual_sources = [
                VDSmap(dcpl.get_virtual_vspace(j),
                       dcpl.get_virtual_filename(j),
                       dcpl.get_virtual_dsetname(j),
                       dcpl.get_virtual_srcspace(j))
                for j in range(dcpl.get_virtual_count())]

        slice_map = {s2t(spaceid_to_slice(i.vspace)): spaceid_to_slice(i.src_space)
                     for i in virtual_sources}
        if any(len(i) != 1 for i in slice_map) or any(len(i) != 1 for i in slice_map.values()):

            raise NotImplementedError("More than one dimension is not yet supported")

        slice_map = {i[0]: j[0] for i, j in slice_map.items()}
        # TODO: Get the chunk size from the dataset
        for i in range(math.ceil(self.shape[0]/CHUNK_SIZE)):
            for t in slice_map:
                r = range(*t)
                if i*CHUNK_SIZE in r:
                    self.data_dict[i] = slice_map[t]

    def write(self, mspace, fspace, arr_obj, mtype=None, dxpl=None):
        if mtype is not None:
            raise NotImplementedError("mtype != None")
        mslice = spaceid_to_slice(mspace)
        fslice = spaceid_to_slice(fspace)
        if len(fslice) > 1 or len(self.shape) > 1:
            raise NotImplementedError("More than one dimension is not yet supported")
        data_dict = self.data_dict
        arr = arr_obj[mslice]
        if np.isscalar(arr):
            arr = arr.reshape((1,))

        # Chunks that are modified
        N0 = 0
        for i, s_ in split_slice(fslice[0]):
            # Based on Dataset.__getitem__
            if isinstance(self.data_dict[i], slice):
                selection = select(self.shape, (slice(i*CHUNK_SIZE, (i+1)*CHUNK_SIZE),), dsid=self)

                assert selection.nselect != 0

                a = np.ndarray(selection.mshape, self.dtype, order='C')

                # Read the data into the array a
                mspace = h5s.create_simple(selection.mshape)
                fspace = selection.id
                self.read(mspace, fspace, a, mtype, dxpl=dxpl)

                data_dict[i] = a

            N = N0 + slice_size(s_)
            data_dict[i][s_] = arr[N0:N]
            N0 = N

        return data_dict

def spaceid_to_slice(space):
    sel_type = space.get_select_type()

    if sel_type == h5s.SEL_ALL:
        return ()
    elif sel_type == h5s.SEL_HYPERSLABS:
        slices = []
        starts, strides, counts, blocks = space.get_regular_hyperslab()
        for _start, _stride, count, block in zip(starts, strides, counts, blocks):
            start = _start
            if not (block == 1 or count == 1):
                raise NotImplementedError("Nontrivial blocks are not yet supported")
            end = _start + (_stride*(count - 1) + 1)*block
            stride = _stride if block == 1 else 1
            slices.append(slice(start, end, stride))
        return tuple(slices)
    elif sel_type == h5s.SEL_NONE:
        return (slice(0, 0),)
    else:
        raise NotImplementedError("Point selections are not yet supported")

# This is adapted from h5py._hl.dataset.make_new_dset(). See the LICENSE file
# for the h5py license.
def _make_new_dset(shape, dtype, data):
    # Convert data to a C-contiguous ndarray
    if data is not None:
        if isinstance(data, Empty):
            raise NotImplementedError("Empty data not yet supported")
        from h5py._hl import base
        # normalize strings -> np.dtype objects
        if dtype is not None:
            _dtype = np.dtype(dtype)
        else:
            _dtype = None

        # if we are going to a f2 datatype, pre-convert in python
        # to workaround a possible h5py bug in the conversion.
        is_small_float = (_dtype is not None and
                          _dtype.kind == 'f' and
                          _dtype.itemsize == 2)
        data = np.asarray(data, order="C",
                             dtype=(_dtype if is_small_float
                                    else base.guess_dtype(data)))

    # Validate shape
    if shape is None:
        if data is None:
            if dtype is None:
                raise TypeError("One of data, shape or dtype must be specified")
            raise NotImplementedError("Empty data not yet supported")
        shape = data.shape
    else:
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if data is not None and (np.product(shape, dtype=np.ulonglong) != np.product(data.shape, dtype=np.ulonglong)):
            raise ValueError("Shape tuple is incompatible with data")

    return data
