"""
Public API functions

Everything outside of this file is considered internal API and is subject to
change.
"""
import logging
import numpy as np
from typing import Set, Optional

from contextlib import contextmanager
import datetime

from .backend import initialize
from .versions import (create_version_group, commit_version,
                       get_version_by_timestamp, get_nth_previous_version,
                       set_current_version, all_versions, delete_version, )
from .wrappers import InMemoryGroup
from .hashtable import Hashtable


logger = logging.getLogger(__name__)


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
    >>> f = h5py.File('file.h5') # doctest: +SKIP
    >>> from versioned_hdf5 import VersionedHDF5File
    >>> file = VersionedHDF5File(f) # doctest: +SKIP

    Access versions using indexing

    >>> version1 = file['version1'] # doctest: +SKIP

    This returns a group containing the datasets for that version.

    To create a new version, use :func:`stage_version`.

    >>> with file.stage_version('version2') as group: # doctest: +SKIP
    ...     group['dataset'] = ... # Modify the group
    ...

    When the context manager exits, the version will be written to the file.

    Finally, use

    >>> file.close() # doctest: +SKIP

    to close the `VersionedHDF5File` object (note that the `h5py` file object
    should be closed separately.)
    """
    def __init__(self, f):
        self.f = f
        if '_version_data' not in f:
            initialize(f)
        self._version_data = f['_version_data']
        self._versions = self._version_data['versions']
        self._closed = False
        self._version_cache = {}

    @property
    def closed(self):
        if self._closed:
            return self._closed
        if not self.f.id:
            self._closed = True
        return self._closed

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
        self._version_cache.clear()

    def get_version_by_name(self, version):
        if version.startswith('/'):
            raise ValueError("Versions cannot start with '/'. VersionedHDF5File should not be used to access the top-level of an h5py File.")

        if version == '':
            version = '__first_version__'

        if version not in self._versions:
            raise KeyError(f"Version {version!r} not found")

        g = self._versions[version]
        if not g.attrs['committed']:
            raise ValueError("Version groups cannot accessed from the VersionedHDF5File object before they are committed.")
        if self.f.file.mode == 'r':
            return g
        return InMemoryGroup(g._id, _committed=True)

    def get_version_by_timestamp(self, timestamp, exact=False):
        version = get_version_by_timestamp(self.f, timestamp, exact=exact)
        g = self._versions[version]
        if not g.attrs['committed']:
            raise ValueError("Version groups cannot accessed from the VersionedHDF5File object before they are committed.")
        if self.f.file.mode == 'r':
            return g
        return InMemoryGroup(g._id, _committed=True)

    def __getitem__(self, item):
        if self.closed:
            raise ValueError("File is closed")
        if item in self._version_cache:
            # We don't cache version names because those are already cheap to
            # lookup.
            return self._version_cache[item]

        if item is None:
            return self.get_version_by_name(self.current_version)
        elif isinstance(item, str):
            return self.get_version_by_name(item)
        elif isinstance(item, (int, np.integer)):
            if item > 0:
                raise IndexError("Integer version slice must be negative")
            self._version_cache[item] = self.get_version_by_name(get_nth_previous_version(self.f,
                self.current_version, -item))
            return self._version_cache[item]
        elif isinstance(item, (datetime.datetime, np.datetime64)):
            self._version_cache[item] = self.get_version_by_timestamp(item)
            return self._version_cache[item]
        else:
            raise KeyError(f"Don't know how to get the version for {item!r}")

    def __delitem__(self, item):
        """
        Delete a version

        If the version is the current version, the new current version will be
        set to the previous version.
        """
        if not isinstance(item, str):
            raise NotImplementedError("del is only supported for string keys")
        if item not in self._versions:
            raise KeyError(item)
        new_current = self.current_version if item != self.current_version else self[item].attrs['prev_version']
        delete_version(self.f, item, new_current)
        self._version_cache.clear()

    def __iter__(self):
        return all_versions(self.f, include_first=False)

    @contextmanager
    def stage_version(self, version_name: str, prev_version=None,
                      make_current=True, timestamp=None):
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

        `timestamp` may be a datetime.datetime or np.datetime64 timestamp for
        the version. Note that datetime.datetime timestamps must be in the UTC
        timezone (np.datetime64 timestamps are not timezone aware and are
        assumed to be UTC). If `timestamp` is `None` (the default) the current
        time when the context manager exits is used. When passing in a manual
        timestamp, be aware that no consistency checks are made to ensure that
        version timestamps are linear or not duplicated.

        """
        if self.closed:
            raise ValueError("File is closed")
        old_current = self.current_version
        group = create_version_group(self.f, version_name,
                                     prev_version=prev_version)


        try:
            yield group
            group.close()
            commit_version(group, group.datasets(), make_current=make_current,
                           chunks=group.chunks,
                           compression=group.compression,
                           compression_opts=group.compression_opts,
                           timestamp=timestamp)

            self._log_version_diff_stats(old_current, self.current_version)

        except:
            delete_version(self.f, version_name, old_current)
            raise
        finally:
            self._version_cache.clear()

    def close(self):
        """
        Make sure the VersionedHDF5File object is no longer reachable.
        """
        if not self._closed:
            del self.f
            del self._version_data
            del self._versions
            self._closed = True

    def __repr__(self):
        """
        Prints friendly status information.

        These messages are intended to be similar to h5py messages.
        """
        if self.closed:
            return "<Closed VersionedHDF5File>"
        else:
            return f"<VersionedHDF5File object \"{self.f.filename}\" (mode" \
                   f" {self.f.mode})>"

    def _get_hashes(self, name: str) -> Set[bytes]:
        """Get a set of hashes for the chunks in the dataset.

        Parameters
        ----------
        name : str
            Name of the dataset for which hashes are to be generated

        Returns
        -------
        Set[bytes]
            A set of hashes for the dataset
        """
        with Hashtable(self.f, name) as hashtable:
            return set(hashtable.keys())

    def _log_version_diff_stats(
        self,
        old_version: Optional[str] = None,
        new_version: Optional[str] = None,
    ):
        """Log some stats about differences between two versions.

        Parameters
        ----------
        old_version : Optional[str]
            Old version of the data to compare
        new_version : Optional[str]
            New version of the data to compare
        """
        old_datasets, new_datasets = {}, {}
        if old_version in self:
            old_datasets = self[old_version].datasets()
        if new_version in self:
            new_datasets = self[new_version].datasets()

        msg = [""]
        for name in sorted(set(old_datasets.keys()) | set(new_datasets.keys())):
            old_dataset = old_datasets.get(name, None)
            new_dataset = new_datasets.get(name, None)

            old_hashes, new_hashes = set(), set()
            old_shape, new_shape = None, None
            old_chunks, new_chunks = None, None

            if old_dataset:
                old_shape = old_dataset.shape
                old_chunks = old_dataset.chunks
                old_hashes = self._get_hashes(name)

            if new_dataset:
                new_shape = new_dataset.shape
                new_chunks = new_dataset.chunks
                new_hashes = self._get_hashes(name)

            chunks_reused = len(old_hashes & new_hashes)
            new_chunks_written = len(new_hashes - old_hashes)

            msg.append(
                f"  {name}: Shape: {old_shape} -> {new_shape}; "
                f"Chunks: {old_chunks} -> {new_chunks}; "
                f"New chunks written: {new_chunks_written}; "
                f"Number of chunks reused: {chunks_reused}"
            )
        logger.debug("\n".join(msg))
