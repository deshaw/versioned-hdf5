from h5py._hl.group import Group

import numpy as np

from contextlib import contextmanager
import datetime

from .versions import create_version, get_nth_previous_version, set_current_version

class VersionedHDF5File:
    def __init__(self, f):
        self.f = f
        self._version_data = f['_version_data']
        self._versions = self._version_data['versions']

    @property
    def current_version(self):
        return self._versions.attrs['current_version']

    @current_version.setter
    def current_version(self, version_name):
        set_current_version(self.f, version_name)

    def get_version_by_name(self, version):
        if not version:
            version = '__first_version__'

        if version not in self._versions:
            raise ValueError(f"Version {version!r} not found")

        # TODO: Don't give an in-memory group if the file is read-only
        return InMemoryGroup(self._versions[version]._id)

    def __getitem__(self, item):
        if isinstance(item, str):
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

    #TODO: override other relevant methods here
