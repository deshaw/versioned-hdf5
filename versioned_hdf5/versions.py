from __future__ import annotations

import datetime
import posixpath
from collections import defaultdict
from collections.abc import Mapping
from uuid import uuid4

import numpy as np
from h5py import Dataset, Group

from versioned_hdf5.backend import (
    Filters,
    commit_staged_changes,
    create_virtual_dataset,
    normalize_chunks,
    write_dataset,
)
from versioned_hdf5.staged_changes import StagedChangesArray
from versioned_hdf5.wrappers import (
    DatasetLike,
    DatasetWrapper,
    InMemoryArrayDataset,
    InMemoryDataset,
    InMemoryGroup,
    InMemorySparseDataset,
)

TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S.%f%z"
FORBIDDEN_NAMES = ["versions"]


def create_version_group(f, version_name, prev_version=None):
    """
    Create the version group for a new version.

    prev_version should be a pre-existing version name, None, or ''
    If it is None, it defaults to the current version. If it is '', it creates
    a version with no parent version.
    """
    versions = f["_version_data/versions"]

    if prev_version == "":
        prev_version = "__first_version__"
    elif prev_version is None:
        prev_version = versions.attrs["current_version"]

    if version_name is None:
        version_name = str(uuid4())

    if version_name in versions:
        raise ValueError(f"There is already a version with the name {version_name}")
    if prev_version not in versions:
        raise ValueError(f"Previous version {prev_version!r} not found")

    group = InMemoryGroup(versions.create_group(version_name).id)
    group.attrs["prev_version"] = prev_version
    group.attrs["committed"] = False

    # Placeholder timestamp, just so the attr is there in case something calls
    # get_version_by_timestamp before the version is committed.
    ts = datetime.datetime.now(datetime.timezone.utc)
    group.attrs["timestamp"] = ts.strftime(TIMESTAMP_FMT)

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


def commit_version(
    version_group: InMemoryGroup,
    datasets: dict[str, InMemoryDataset | DatasetLike],
    *,
    make_current: bool = True,
    chunks: Mapping[str, tuple[int, ...] | bool | None] | None = None,
    filters: Mapping[str, Filters] | None = None,
    timestamp: datetime.datetime | np.datetime64 | None = None,
):
    """
    Create a new version.

    datasets should be a dictionary mapping {path: dataset}, where `dataset`
    is a *Dataset object.

    If make_current is True, the new version will be set as the current version.

    If the user specifies a dataset name found in FORBIDDEN_NAMES, a ValueError
    will be raised.

    Returns the group for the new version.
    """
    if "committed" not in version_group.attrs:
        raise ValueError(
            "version_group must be a group created by create_version_group()"
        )
    if version_group.attrs["committed"]:
        raise ValueError("This version group has already been committed")
    version_name = version_group.name.rsplit("/", 1)[1]
    versions = version_group.parent
    f = versions.parent.parent
    prev_version = versions[version_group.attrs["prev_version"]]

    if not isinstance(chunks, defaultdict):
        chunks = defaultdict(type(None), **(chunks or {}))
    if not isinstance(filters, defaultdict):
        filters = defaultdict(Filters, **(filters or {}))

    if make_current:
        versions.attrs["current_version"] = version_name

    # Check all dataset names for forbidden names before attempting any data writes
    for name in datasets:
        if name in FORBIDDEN_NAMES:
            raise ValueError(f"{name} is a forbidden dataset name; aborting.")

    for name, data in datasets.items():
        if isinstance(data, DatasetWrapper):
            data = data.dataset

        if isinstance(data, InMemoryDataset):
            # New version of an existing dataset
            if not data.staged_changes.has_changes:
                # The virtual dataset was not changed from the previous
                # version. Just copy it to the new version directly.
                assert data.name.startswith(prev_version.name + "/")
                data_name = data.name[len(prev_version.name + "/") :]
                data_copy_name = posixpath.join(version_group.name, data_name)
                version_group.copy(data, data_copy_name)
                data_copy = f[data_copy_name]
                data_copy.attrs.clear()
                for k, v in data.attrs.items():
                    data_copy.attrs[k] = v
                continue
            # Commit the staged changes straight into raw_data + the on-disk hash table.
            slices = commit_staged_changes(f, name, data.staged_changes)
        elif isinstance(data, InMemorySparseDataset):
            # Either a new sparse dataset or DatasetWrapper performing a hotswap of its
            # inner dataset. Create the (empty) raw_data + hash table if they don't
            # exist yet; otherwise validate chunks, filters, fillvalue, and dtype
            # against them.
            write_dataset(
                f,
                name,
                np.empty((0,) * len(data.shape), dtype=data._buffer.dtype),
                chunks=chunks[name],
                filters=filters[name],
                fillvalue=data.fillvalue,
            )
            slices = commit_staged_changes(f, name, data.staged_changes)
        elif isinstance(data, InMemoryArrayDataset):
            # Either a new dense dataset or DatasetWrapper performing a hotswap of its
            # inner dataset.
            # Resolve the chunk size upfront: write_dataset() would otherwise guess
            # it from the empty array below instead of from the real data shape.
            ds_chunks = chunks[name]
            if f"_version_data/{name}/raw_data" not in f:
                ds_chunks = normalize_chunks(ds_chunks, data.shape, data._buffer.dtype)

            # Create the (empty) raw_data + hash table if they don't exist yet;
            # otherwise validate chunks, filters, fillvalue, and dtype against them.
            write_dataset(
                f,
                name,
                # Note: data._buffer.dtype could be StringDType while data.dtype
                # presents as object dtype. Avoid unnecessary conversion.
                np.empty((0,) * data.ndim, dtype=data._buffer.dtype),
                chunks=ds_chunks,
                filters=filters[name],
                fillvalue=data.fillvalue,
            )
            chunk_size = tuple(f[f"_version_data/{name}/raw_data"].attrs["chunks"])

            staged_changes = StagedChangesArray.from_array(
                data._buffer,
                chunk_size=chunk_size,
                fill_value=data.fillvalue,
                as_base_slabs=False,
            )
            slices = commit_staged_changes(f, name, staged_changes)
        else:
            raise AssertionError("Unreachable")

        create_virtual_dataset(
            f,
            version_name,
            name,
            data.shape,
            slices,
            attrs=data.attrs,
            fillvalue=data.fillvalue,
        )

    version_group.attrs["committed"] = True

    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc)
    elif isinstance(timestamp, datetime.datetime):
        if timestamp.utcoffset() != datetime.timedelta(0):
            raise ValueError("timestamp must be in UTC")
    elif isinstance(timestamp, np.datetime64):
        timestamp = timestamp.astype(datetime.datetime).replace(
            tzinfo=datetime.timezone.utc
        )
    else:
        raise TypeError(
            "timestamp must be either a datetime.datetime or numpy.datetime64 object"
        )
    version_group.attrs["timestamp"] = timestamp.strftime(TIMESTAMP_FMT)


def delete_version(f, version_name, new_current=None):
    """
    Delete version `version_name`.
    """
    versions = f["_version_data/versions"]

    if version_name not in versions:
        raise ValueError(f"version {version_name!r} does not exist")
    if not new_current:
        new_current = "__first_version__"
    if new_current not in versions:
        raise ValueError(f"version {new_current!r} does not exist")

    del versions[version_name]
    versions.attrs["current_version"] = new_current


def get_nth_previous_version(f, version_name, n):
    versions = f["_version_data/versions"]
    if version_name not in versions:
        raise IndexError(f"Version {version_name!r} not found")

    version = version_name
    for _ in range(n):
        version = versions[version].attrs["prev_version"]

        # __first_version__ is a meta-version and should not be returnable
        if version == "__first_version__":
            raise IndexError(f"{version_name!r} has fewer than {n} versions before it")

    return version


def get_version_by_timestamp(f, timestamp, exact=False):
    versions = f["_version_data/versions"]
    if isinstance(timestamp, np.datetime64):
        ts = (
            timestamp.astype(datetime.datetime)
            .replace(tzinfo=datetime.timezone.utc)
            .strftime(TIMESTAMP_FMT)
        )
    elif isinstance(timestamp, datetime.datetime):
        if timestamp.utcoffset() != datetime.timedelta(0):
            raise ValueError("timestamp must be in UTC")
        ts = timestamp.strftime(TIMESTAMP_FMT)
    else:
        raise TypeError(
            "timestamp must be either a datetime.datetime or numpy.datetime64 object"
        )
    best_match = None
    best_ts = None
    # Note: Due to low time resolution on Windows + Python 3.10/3.11, it is
    # possible that all non-first versions have the same timestamp of
    # __first_version__. Do not accidentally discard them.
    # In case of multiple "best match" versions with identical timestamps,
    # return the first one yielded by iterating the versions group (iteration
    # order is not guaranteed by HDF5/h5py).
    for version in versions:
        if version == "__first_version__":
            continue
        version_ts = versions[version].attrs["timestamp"]
        if version_ts == ts:
            return version
        # Find the version whose timestamp is closest to ts and before it.
        if not exact and version_ts < ts and (best_ts is None or best_ts < version_ts):
            best_match = version
            best_ts = version_ts
    if best_match is None:
        if exact:
            raise KeyError(f"Version with timestamp {timestamp} not found")
        raise KeyError(f"Version with timestamp before {timestamp} not found")
    return best_match


def set_current_version(f, version_name):
    versions = f["_version_data/versions"]
    if version_name not in versions:
        raise ValueError(f"Version {version_name!r} not found")

    versions.attrs["current_version"] = version_name


def all_versions(f, *, include_first=False):
    """
    Return a generator that iterates all versions by name

    If include_first is True, it will include '__first_version__'.

    Note that the order of the versions is completely arbitrary.
    """
    versions = f["_version_data/versions"]
    for version in versions:
        if version == "__first_version__":
            if include_first:
                yield version
        else:
            yield version
