from h5py import Empty, Dataset, Group

import numpy as np

from contextlib import contextmanager
import datetime

from .backend import initialize
from .versions import (create_version, get_nth_previous_version,
                       set_current_version, all_versions)

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
            raise ValueError(f"Version {version!r} not found")

        # TODO: Don't give an in-memory group if the file is read-only
        return InMemoryGroup(self._versions[version]._id)

    def __getitem__(self, item):
        try:
            if item is None:
                return self.get_version_by_name(self.current_version)
            elif isinstance(item, str):
                return self.get_version_by_name(item)
            elif isinstance(item, (int, np.integer)):
                if item > 0:
                    raise KeyError("Integer version slice must be negative")
                return self.get_version_by_name(get_nth_previous_version(self.f,
                    self.current_version, -item))
            elif isinstance(item, (datetime.datetime, np.datetime64)):
                raise NotImplementedError
            else:
                raise KeyError(f"Don't know how to get the version for {item!r}")
        except ValueError as e:
            raise KeyError(e.args[0]) from e

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
        super().__init__(bind)

    def __getitem__(self, name):
        if name in self._data:
            return self._data[name]

        res = super().__getitem__(name)
        if isinstance(res, Group):
            return self.__class__(res.id)

        res2 = np.array(res)
        self._data[name] = res2
        return res2

    def __setitem__(self, name, obj):
        self._data[name] = obj

    def __delitem__(self, name):
        if name in self._data:
            del self._data
        super().__delitem__(name)

    def create_dataset(self, name, *, shape=None, dtype=None, data=None, **kwds):
        if kwds:
            raise NotImplementedError

        data = _make_new_dset(shape, dtype, data)

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
