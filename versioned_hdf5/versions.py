from uuid import uuid4
from collections import defaultdict
import posixpath as pp

from h5py import Dataset, Group
import datetime
import numpy as np

from .backend import write_dataset, write_dataset_chunks, create_virtual_dataset
from .wrappers import InMemoryGroup, DatasetWrapper, InMemoryDataset, InMemoryArrayDataset, InMemorySparseDataset

TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S.%f%z"

def create_version_group(f, version_name, prev_version=None):
    """
    Create the version group for a new version.

    prev_version should be a pre-existing version name, None, or ''
    If it is None, it defaults to the current version. If it is '', it creates
    a version with no parent version.
    """
    versions = f['_version_data/versions']

    if prev_version == '':
        prev_version = '__first_version__'
    elif prev_version is None:
        prev_version = versions.attrs['current_version']

    if version_name is None:
        version_name = str(uuid4())

    if version_name in versions:
        raise ValueError(f"There is already a version with the name {version_name}")
    if prev_version not in versions:
        raise ValueError(f"Previous version {prev_version!r} not found")

    group = InMemoryGroup(versions.create_group(version_name).id)
    group.attrs['prev_version'] = prev_version
    group.attrs['committed'] = False

    # Placeholder timestamp, just so the attr is there in case something calls
    # get_version_by_timestamp before the version is committed.
    ts = datetime.datetime.now(datetime.timezone.utc)
    group.attrs['timestamp'] = ts.strftime(TIMESTAMP_FMT)

    # Copy everything over from the previous version
    prev_group = versions[prev_version]

    def _get(name, item):
        if isinstance(item, (Group, InMemoryGroup)):
            group.create_group(name)
        elif isinstance(item, Dataset):
            group[name] = item
        else:
            raise NotImplementedError(f"{type(item)}")
        for k, v in item.attrs.items():
            group[name].attrs[k] = v

    prev_group.visititems(_get)
    return group

def commit_version(version_group, datasets, *,
                   make_current=True, chunks=None,
                   compression=None, compression_opts=None,
                   timestamp=None):
    """
    Create a new version

    datasets should be a dictionary mapping {path: dataset}, where `dataset`
    is either a numpy array, or a dictionary mapping {chunk_index:
    data_or_slice}, where `data_or_slice` is either an array or a slice
    pointing into the raw data for that chunk.

    If make_current is True, the new version will be set as the current version.

    Returns the group for the new version.
    """
    if 'committed' not in version_group.attrs:
        raise ValueError("version_group must be a group created by create_version_group()")
    if version_group.attrs['committed']:
        raise ValueError("This version group has already been committed")
    version_name = version_group.name.rsplit('/', 1)[1]
    versions = version_group.parent
    f = versions.parent.parent
    prev_version = versions[version_group.attrs['prev_version']]

    chunks = chunks or defaultdict(type(None))
    compression = compression or defaultdict(type(None))
    compression_opts = compression_opts or defaultdict(type(None))

    if make_current:
        versions.attrs['current_version'] = version_name

    for name, data in datasets.items():
        fillvalue = None
        if isinstance(data, DatasetWrapper):
            data = data.dataset
        if isinstance(data, (InMemoryDataset, InMemoryArrayDataset, InMemorySparseDataset)):
            attrs = data.attrs
            fillvalue = data.fillvalue
        else:
            attrs = {}

        shape = None
        if isinstance(data, InMemoryDataset):
            shape = data.shape
            if data.id._data_dict is None:
                # The virtual dataset was not changed from the previous
                # version. Just copy it to the new version directly.
                assert data.name.startswith(prev_version.name + '/')
                data_name = data.name[len(prev_version.name + '/'):]
                data_copy_name = pp.join(version_group.name, data_name)
                version_group.copy(data, data_copy_name)
                data_copy = f[data_copy_name]
                data_copy.attrs.clear()
                for k, v in data.attrs.items():
                    data_copy.attrs[k] = v
                continue
            data = data.id._data_dict
        if isinstance(data, dict):
            if chunks[name] is not None:
                raise NotImplementedError("Specifying chunk size with dict data")
            slices = write_dataset_chunks(f, name, data, shape=shape)
        elif isinstance(data, InMemorySparseDataset):
            write_dataset(f, name, np.empty((0,)*len(data.shape),
                                                     dtype=data.dtype), chunks=chunks[name],
                                   compression=compression[name],
                                   compression_opts=compression_opts[name],
                                   fillvalue=fillvalue)
            slices = write_dataset_chunks(f, name, data.data_dict, shape=data.shape)
        else:
            slices = write_dataset(f, name, data, chunks=chunks[name],
                                   compression=compression[name],
                                   compression_opts=compression_opts[name],
                                   fillvalue=fillvalue)
        if shape is None:
            if isinstance(data, dict):
                raw_data = f['_version_data'][name]['raw_data']
                shape = tuple([max(c.args[i].stop for c in slices) for i in range(len(tuple(raw_data.attrs['chunks'])))])
            else:
                shape = data.shape
        create_virtual_dataset(f, version_name, name, shape, slices, attrs=attrs,
                               fillvalue=fillvalue)
    version_group.attrs['committed'] = True

    if timestamp is not None:
        if isinstance(timestamp, datetime.datetime):
            if timestamp.utcoffset() != datetime.timedelta(0):
                raise ValueError("timestamp must be in UTC")
            version_group.attrs['timestamp'] = timestamp.strftime(TIMESTAMP_FMT)
        elif isinstance(timestamp, np.datetime64):
            version_group.attrs['timestamp'] = timestamp.astype(datetime.datetime).replace(tzinfo=datetime.timezone.utc).strftime(TIMESTAMP_FMT)
        else:
            raise TypeError("timestamp must be either a datetime.datetime or numpy.datetime64 object")
    else:
        ts = datetime.datetime.now(datetime.timezone.utc)
        version_group.attrs['timestamp'] = ts.strftime(TIMESTAMP_FMT)


def delete_version(f, version_name, new_current=None):
    """
    Delete version `version_name`.
    """
    versions = f['_version_data/versions']

    if version_name not in versions:
        raise ValueError(f"version {version_name!r} does not exist")
    if not new_current:
        new_current = '__first_version__'
    if new_current not in versions:
        raise ValueError(f"version {new_current!r} does not exist")

    del versions[version_name]
    versions.attrs['current_version'] = new_current

def get_nth_previous_version(f, version_name, n):
    versions = f['_version_data/versions']
    if version_name not in versions:
        raise IndexError(f"Version {version_name!r} not found")

    version = version_name
    for i in range(n):
        version = versions[version].attrs['prev_version']

        # __first_version__ is a meta-version and should not be returnable
        if version == '__first_version__':
            raise IndexError(f"{version_name!r} has fewer than {n} versions before it")

    return version

def get_version_by_timestamp(f, timestamp, exact=False):
    versions = f['_version_data/versions']
    if isinstance(timestamp, np.datetime64):
        ts = timestamp.astype(datetime.datetime).replace(tzinfo=datetime.timezone.utc).strftime(TIMESTAMP_FMT)
    elif isinstance(timestamp, datetime.datetime):
        if timestamp.utcoffset() != datetime.timedelta(0):
            raise ValueError("timestamp must be in UTC")
        ts = timestamp.strftime(TIMESTAMP_FMT)
    else:
        raise TypeError("timestamp must be either a datetime.datetime or numpy.datetime64 object")
    best_match = '__first_version__'
    best_ts = versions[best_match].attrs['timestamp']
    for version in versions:
        version_ts = versions[version].attrs['timestamp']
        if version != '__first_version__':
            if exact:
                if ts == version_ts:
                    return version
            else:
                if ts == version_ts:
                    return version
                # Find the version whose timestamp is closest to ts and before
                # it.
                if best_ts < version_ts <= ts:
                    best_match = version
                    best_ts = version_ts
    if best_match == '__first_version__':
        if exact:
            raise KeyError(f"Version with timestamp {timestamp} not found")
        raise KeyError(f"Version with timestamp before {timestamp} not found")
    return best_match

def set_current_version(f, version_name):
    versions = f['_version_data/versions']
    if version_name not in versions:
        raise ValueError(f"Version {version_name!r} not found")

    versions.attrs['current_version'] = version_name

def all_versions(f, *, include_first=False):
    """
    Return a generator that iterates all versions by name

    If include_first is True, it will include '__first_version__'.

    Note that the order of the versions is completely arbitrary.
    """
    versions = f['_version_data/versions']
    for version in versions:
        if version == '__first_version__':
            if include_first:
                yield version
        else:
            yield version
