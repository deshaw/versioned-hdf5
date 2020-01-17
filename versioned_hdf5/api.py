from h5py._hl.group import Group
from h5py import Empty

import numpy as np

from contextlib import contextmanager
import datetime

from .backend import initialize
from .versions import (create_version, get_nth_previous_version,
                       set_current_version, all_versions)

class VersionedHDF5File:
    def __init__(self, f):
        self.f = f
        if '_version_data' not in f:
            initialize(f)
        self._version_data = f['_version_data']
        self._versions = self._version_data['versions']

    @property
    def current_version(self):
        return self._versions.attrs['current_version']

    @current_version.setter
    def current_version(self, version_name):
        set_current_version(self.f, version_name)

    def get_version_by_name(self, version):
        if version == '':
            version = '__first_version__'

        if version not in self._versions:
            raise ValueError(f"Version {version!r} not found")

        # TODO: Don't give an in-memory group if the file is read-only
        return InMemoryGroup(self._versions[version]._id)

    def __getitem__(self, item):
        if item is None:
            return self.get_version_by_name(self.current_version)
        elif isinstance(item, str):
            return self.get_version_by_name(item)
        elif isinstance(item, (int, np.integer)):
            if item > 0:
                raise ValueError("Integer version slice must be negative")
            return self.get_version_by_name(get_nth_previous_version(self.f,
                self.current_version, -item))
        elif isinstance(item, (datetime.datetime, np.datetime64)):
            raise NotImplementedError
        else:
            raise TypeError(f"Don't know how to get the version for {item!r}")

    def __iter__(self):
        return all_versions(self.f, include_first=False)

    @contextmanager
    def stage_version(self, version_name, prev_version=None, make_current=True):
        group = self[prev_version]
        yield group
        create_version(self.f, version_name, group._data,
                       prev_version=prev_version, make_current=make_current)

class InMemoryGroup(Group):
    def __init__(self, bind):
        self._data = {}
        super().__init__(bind)

    def __getitem__(self, name):
        if name in self._data:
            return self._data[name]

        res = super().__getitem__(name)
        if isinstance(res, Group):
            raise NotImplementedError("Groups are not yet supported")

        res2 = np.array(res)
        self._data[name] = res2
        return res2

    def __setitem__(self, name, obj):
        self._data[name] = obj

    def create_dataset(self, name, *, shape=None, dtype=None, data=None, **kwds):
        if kwds:
            raise NotImplementedError

        data = _make_new_dset(shape, dtype, data)

        self[name] = data

    #TODO: override other relevant methods here

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
