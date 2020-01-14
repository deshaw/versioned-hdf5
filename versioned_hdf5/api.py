from h5py._hl.group import Group

import numpy as np

from contextlib import contextmanager
import datetime

from .versions import create_version

class Data:
    def __init__(self, f):
        self.f = f
        self._version_data = f['_version_data']
        self._versions = self._version_data['versions']

    @property
    def version(self):
        return Version(self)

    def get_version_by_name(self, version):
        if version not in self._versions:
            raise ValueError(f"Version {version!r} not found")

        return InMemoryGroup(self._versions[version]._id)

class Version:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.data.get_version_by_name(item)
        elif isinstance(item, (int, datetime.datetime, np.datetime64)):
            raise NotImplementedError
        else:
            raise TypeError(f"Don't know how to get the version for {item!r}")

    @contextmanager
    def stage_version(self, version_name, prev_version):
        group = self[prev_version]
        yield group
        create_version(self.data.f, version_name, prev_version, group._data)

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

        res2 = np.asarray(res)
        self._data[name] = res2
        return res2

    def __setitem__(self, name, obj):
        raise NotImplementedError

    #TODO: override other relevant methods here
