from __future__ import annotations

import gc
import logging
import posixpath
from collections.abc import Iterable
from typing import Any

import numpy as np
from h5py import Dataset, File, Group, HLObject, VirtualLayout, h5s
from h5py import __version__ as h5py_version
from h5py._hl.selections import select
from h5py._selector import Selector
from h5py.h5i import get_name
from ndindex import ChunkSize, Slice, Tuple
from ndindex.ndindex import NDIndex

from versioned_hdf5.api import VersionedHDF5File
from versioned_hdf5.backend import (
    Filters,
    create_base_dataset,
    create_virtual_dataset,
    initialize,
    write_dataset,
    write_dataset_chunks,
)
from versioned_hdf5.hashtable import Hashtable
from versioned_hdf5.slicetools import spaceid_to_slice
from versioned_hdf5.typing_ import DEFAULT, Default
from versioned_hdf5.versions import all_versions
from versioned_hdf5.wrappers import (
    DatasetWrapper,
    InMemoryArrayDataset,
    InMemoryDataset,
    InMemoryGroup,
    InMemorySparseDataset,
)

logger = logging.getLogger(__name__)


def recreate_dataset(f, name, newf, callback=None):
    """
    Recreate dataset from all versions into `newf`

    `newf` should be a versioned hdf5 file/group that is already initialized
    (it may or may not be in the same physical file as f). Typically `newf`
    should be `tmp_group(f)` (see :func:`tmp_group`).

    `callback` should be a function with the signature

        callback(dataset, version_name)

    It will be called on every dataset in every version. It should return the
    dataset to be used for the new version. The dataset and its containing
    group should not be modified in-place. If a new copy of a dataset is to be
    used, it should be one of the dataset classes in versioned_hdf5.wrappers,
    and should placed in a temporary group, which you may delete after
    `recreate_dataset()` is done. The callback may also return None, in which
    case the dataset is deleted for the given version.

    Notes
    -----
    This function is only for advanced usage. Typical use-cases should
    use :func:`delete_version()` or :func:`modify_metadata()`.
    """
    if isinstance(f, VersionedHDF5File):
        f = f.f

    raw_data = f["_version_data"][name]["raw_data"]

    dtype = raw_data.dtype
    chunks = raw_data.chunks
    fillvalue = raw_data.fillvalue

    first = True
    for version_name in all_versions(f):
        if name in f["_version_data/versions"][version_name]:
            group = InMemoryGroup(
                f["_version_data/versions"][version_name].id, _committed=True
            )

            dataset = group[name]
            if callback:
                dataset = callback(dataset, version_name)
                if dataset is None:
                    continue

            dtype = dataset.dtype
            shape = dataset.shape
            chunks = dataset.chunks

            filters = Filters.from_dataset(dataset)
            fillvalue = dataset.fillvalue
            attrs = dataset.attrs
            if first:
                create_base_dataset(
                    newf,
                    name,
                    data=np.empty((0,) * len(dataset.shape), dtype=dtype),
                    dtype=dtype,
                    chunks=chunks,
                    fillvalue=fillvalue,
                    filters=filters,
                )
                first = False
            # Read in all the chunks of the dataset (we can't assume the new
            # hash table has the raw data in the same locations, even if the
            # data is unchanged).
            if isinstance(dataset, (InMemoryDataset, InMemorySparseDataset)):
                dataset.staged_changes.load()
                slices = write_dataset_chunks(newf, name, dataset.data_dict)
            else:
                slices = write_dataset(newf, name, dataset)
            create_virtual_dataset(
                newf,
                version_name,
                name,
                shape,
                slices,
                attrs=attrs,
                fillvalue=fillvalue,
            )


def tmp_group(f):
    """
    Create a temporary group in `f` for use with :func:`recreate_dataset`.
    """
    if isinstance(f, VersionedHDF5File):
        f = f.f

    if "__tmp__" not in f["_version_data"]:
        tmp = f["_version_data"].create_group("__tmp__")
        initialize(tmp)
        for version_name in all_versions(f):
            group = f["_version_data/versions"][version_name]
            new_group = tmp["_version_data/versions"].create_group(version_name)
            for k, v in group.attrs.items():
                new_group.attrs[k] = v
    else:
        tmp = f["_version_data/__tmp__"]
    return tmp


# See InMemoryDataset.fillvalue. In h5py3 variable length strings use None
# for the h5py fillvalue, but require a string fillvalue for NumPy.
def _get_np_fillvalue(data: Dataset) -> Any:
    """Get the fillvalue for an empty dataset.

    See InMemoryDataset.fillvalue. In h5py3 variable length strings use None
    for the h5py fillvalue, but require a string fillvalue for NumPy.

    Parameters
    ----------
    data : Dataset
        Data for which the fillvalue is to be retrieved

    Returns
    -------
    Any
        Value used to fill the empty dataset; can be any numpy scalar type supported by
        h5py
    """
    if data.fillvalue is not None:
        return data.fillvalue
    if data.dtype.metadata:
        if "vlen" in data.dtype.metadata:
            if h5py_version.startswith("3") and data.dtype.metadata["vlen"] is str:
                return b""
            return data.dtype.metadata["vlen"]()
        elif "h5py_encoding" in data.dtype.metadata:
            return data.dtype.type()
    return np.zeros((), dtype=data.dtype)[()]


def _recreate_raw_data(
    f: VersionedHDF5File | File,
    name: str,
    versions_to_delete: Iterable[str],
) -> dict[NDIndex, NDIndex] | None:
    """Create a new raw dataset without the chunks from versions_to_delete.

    Parameters
    ----------
    f : VersionedHDF5File | File
        File for which the raw data is to be reconstructed
    name : str
        Name of the dataset
    versions_to_delete : Iterable[str]
        Versions to omit from the reconstructed raw_data

    Returns
    -------
    dict[NDIndex, NDIndex] | None
        A mapping between old raw dataset chunks and the new raw dataset chunks

        If no chunks would be left, i.e., the dataset does not appear in any
        version not in versions_to_delete, None is returned.
    """
    chunks_to_keep = set()

    if isinstance(f, VersionedHDF5File):
        vf = f
    else:
        vf = VersionedHDF5File(f)

    for version in vf.versions:
        if version not in versions_to_delete and name in vf[version]:
            dataset = f["_version_data/versions"][version][name]

            if dataset.is_virtual:
                for i in dataset.virtual_sources():
                    chunks_to_keep.add(spaceid_to_slice(i.src_space))

    raw_data = f["_version_data"][name]["raw_data"]
    chunks = ChunkSize(raw_data.chunks)
    new_shape = (len(chunks_to_keep) * chunks[0], *chunks[1:])

    fillvalue = _get_np_fillvalue(raw_data)
    # Guard against existing _tmp_raw_data
    _delete_tmp_raw_data(f, name)

    sorted_chunks_to_keep = sorted(chunks_to_keep, key=lambda i: i.args[0].args[0])

    raw_data_chunks_map = {}
    for new_chunk, chunk in zip(
        chunks.indices(new_shape), sorted_chunks_to_keep, strict=False
    ):
        # Truncate the new slice if it isn't a full chunk
        to_set_fillvalue = []
        new_truncated = []
        for i in range(len(new_chunk.args)):
            end = new_chunk.args[i].start + len(chunk.args[i])
            new_truncated.append(Slice(new_chunk.args[i].start, end))

            # If one dimension is truncated, create slices into
            # all other dimensions to be set to the fillvalue
            if len(new_chunk.args[i]) != len(chunk.args[i]):
                to_fill = []
                for j in range(len(new_chunk.args)):
                    if j == i:
                        to_fill.append(Slice(end, new_chunk.args[i].stop))
                    else:
                        to_fill.append(
                            Slice(
                                new_chunk.args[j].start,
                                new_chunk.args[j].start + len(chunk.args[j]),
                            )
                        )
                to_set_fillvalue.append(Tuple(*to_fill))

        new_truncated = Tuple(*new_truncated)
        raw_data[new_truncated.raw] = raw_data[chunk.raw]

        # Set the fillvalue of any slices which were truncated.
        for tup in to_set_fillvalue:
            raw_data[tup.raw] = fillvalue
        raw_data_chunks_map[chunk] = new_truncated

    raw_data.resize(new_shape)

    return raw_data_chunks_map


def _delete_tmp_raw_data(f: File, name: str):
    """Delete _tmp_raw_data if it exists in the file.

    Parameters
    ----------
    f : File
        File in which _tmp_raw_data is to be removed
    name : str
        Name of the dataset where _tmp_raw_data is to be removed
    """
    if "_tmp_raw_data" in f["_version_data"][name]:
        del f["_version_data"][name]["_tmp_raw_data"]


def _recreate_hashtable(f, name, raw_data_chunks_map, tmp=False):
    """
    Recreate the hashtable for the dataset f, with only the new chunks in the
    raw_data_chunks_map.

    If tmp=True, a new hashtable called '_tmp_hash_table' is created.
    Otherwise the hashtable is replaced.
    """
    # We could just reconstruct the hashtable with from_raw_data, but that is
    # slow, so instead we recreate it manually from the old hashable and the
    # raw_data_chunks_map.
    new_hash_table = Hashtable(f, name, hash_table_name="_tmp_hash_table")
    old_inverse = Hashtable(f, name).inverse()

    for old_chunk, new_chunk in raw_data_chunks_map.items():
        if isinstance(old_chunk, Tuple):
            old_chunk = old_chunk.args[0]
        if isinstance(new_chunk, Tuple):
            new_chunk = new_chunk.args[0]

        new_hash_table[old_inverse[old_chunk.reduce()]] = new_chunk

    new_hash_table.write()

    if not tmp:
        del f["_version_data"][name]["hash_table"]
        f["_version_data"][name].move("_tmp_hash_table", "hash_table")


def _recreate_virtual_dataset(f, name, versions, raw_data_chunks_map, tmp=False):
    """
    Recreate every virtual dataset `name` in the versions `versions` according
    to the new raw_data chunks in `raw_data_chunks_map`.

    Returns a dict mapping the chunks from the old raw dataset to the chunks
    in the new raw dataset. Chunks not in the mapping were deleted. If the
    dict is empty, then no remaining version contains the given dataset.

    If tmp is True, the new virtual datasets are named `'_tmp_' + name` and
    are placed alongside the existing ones. Otherwise the existing virtual
    datasets are replaced.

    See Also
    --------
    create_virtual_dataset
    """
    raw_data = f["_version_data"][name]["raw_data"]

    for version_name in versions:
        if name not in f["_version_data/versions"][version_name]:
            continue

        group = f["_version_data/versions"][version_name]
        dataset = group[name]
        layout = VirtualLayout(dataset.shape, dtype=dataset.dtype)

        # If a dataset has no data except for the fillvalue, it will not be virtual
        if dataset.is_virtual:
            layout._src_filenames.add(b".")
            space = h5s.create_simple(dataset.shape)
            selector = Selector(space)
            raw_data_shape = raw_data.shape
            raw_data_name = raw_data.name.encode("utf-8")

            virtual_sources = dataset.virtual_sources()
            for vmap in virtual_sources:
                vspace, fname, _, src_space = vmap
                assert fname == "."

                vslice = spaceid_to_slice(vspace)
                src_slice = spaceid_to_slice(src_space)
                if src_slice not in raw_data_chunks_map:
                    raise ValueError(
                        f"Could not find the chunk for {vslice} ({src_slice} in the "
                        f"old raw dataset) for {name!r} in {version_name!r}"
                    )

                new_src_slice = raw_data_chunks_map[src_slice]
                vs_sel = select(raw_data_shape, new_src_slice.raw, dataset=None)
                sel = selector.make_selection(vslice.raw)
                layout.dcpl.set_virtual(sel.id, b".", raw_data_name, vs_sel.id)

        head, tail = posixpath.split(name)
        tmp_name = "_tmp_" + tail
        tmp_path = posixpath.join(head, tmp_name)
        dtype = raw_data.dtype
        fillvalue = dataset.fillvalue
        if dtype.metadata and (
            "vlen" in dtype.metadata or "h5py_encoding" in dtype.metadata
        ):
            # Variable length string dtype
            # (https://h5py.readthedocs.io/en/2.10.0/strings.html). Setting the
            # fillvalue in this case doesn't work
            # (https://github.com/h5py/h5py/issues/941).
            if fillvalue not in [0, "", b"", None]:
                raise ValueError(
                    "Non-default fillvalue not supported for variable length strings"
                )
            fillvalue = None
        tmp_dataset = group.create_virtual_dataset(
            tmp_path, layout, fillvalue=fillvalue
        )

        for key, val in dataset.attrs.items():
            tmp_dataset.attrs[key] = val

        if not tmp:
            del group[name]
            group.move(tmp_path, name)


def _is_empty(f: VersionedHDF5File, name: str, version: str) -> bool:
    """Return True if the dataset at the given version is empty, False otherwise.

    Assumes the dataset exists in the given verison.

    Parameters
    ----------
    f : VersionedHDF5File
        File where the dataset resides
    name : str
        Name of the dataset
    version : str
        Version of the dataset to check

    Returns
    -------
    bool
        True if the dataset is empty, False otherwise
    """
    return f["_version_data/versions"][version][name].len() == 0


def _exists_in_version(f: VersionedHDF5File, name: str, version: str) -> bool:
    """Check if a dataset exists in a given version.

    Parameters
    ----------
    f : VersionedHDF5File
        File where the dataset may reside
    name : str
        Name of the dataset
    version : str
        Version of the dataset to check

    Returns
    -------
    bool
        True if the dataset exists in the version, False otherwise
    """
    return name in f["_version_data/versions"][version]


def _all_extant_are_empty(
    f: VersionedHDF5File, name: str, versions: Iterable[str]
) -> bool:
    """Check if the given versions of a dataset are empty.

    Doesn't assume the dataset exists in any version.

    Parameters
    ----------
    f : VersionedHDF5File
        File where the dataset may reside
    name : str
        Name of the dataset
    version : str
        Version of the dataset to check

    Returns
    -------
    bool
        True if any version of the dataset that can be found is empty,
        False if a version exists which is not.
    """
    return all(
        not _exists_in_version(f, name, version) or _is_empty(f, name, version)
        for version in versions
    )


def _delete_dataset(f: VersionedHDF5File, name: str, versions_to_delete: Iterable[str]):
    """Delete the given dataset from the versions."""
    version_data = f["_version_data"]
    versions = version_data["versions"]

    if name == "versions":
        return

    versions_to_keep = set(versions) - set(versions_to_delete)

    # If the dataset is empty in the versions to delete, we don't
    # need to recreate the raw data, hash table, or virtual datasets.
    if _all_extant_are_empty(f, name, versions_to_delete):
        return

    raw_data_chunks_map = _recreate_raw_data(f, name, versions_to_delete)

    # If the dataset is not in any versions that are being kept, that
    # data must be deleted.
    if not any([name in versions[version] for version in versions_to_keep]):
        del version_data[name]
        return

    # Recreate the hash table.
    _recreate_hashtable(f, name, raw_data_chunks_map)

    # Recreate every virtual dataset in every kept version.
    _recreate_virtual_dataset(f, name, versions_to_keep, raw_data_chunks_map)


def _walk(g: HLObject, prefix: str = "") -> list[str]:
    """Traverse the object tree, returning all `raw_data` datasets.

    We use this instead of version_data.visit(delete_dataset) because
    visit() has trouble with the groups being deleted from under it.

    Parameters
    ----------
    g : HLObject
        Object containing datasets as descendants
    prefix : str
        Prefix to apply to object names; can be used to filter particular descendants

    Returns
    -------
    list[str]
        List of the names of `raw_data` datasets in g
    """
    datasets = []
    for name in g:
        obj = g[name]
        if isinstance(obj, Group):
            if "raw_data" in obj:
                datasets.append(prefix + name)
            else:
                datasets.extend(_walk(obj, prefix + name + "/"))

    return datasets


def delete_versions(
    f: VersionedHDF5File | File, versions_to_delete: str | Iterable[str]
):
    """Completely delete the given versions from a file

    This function should be used instead of deleting the version group
    directly, as this will not delete the underlying data that is unique to
    the version.
    """
    if isinstance(f, VersionedHDF5File):
        f = f.f

    version_data = f["_version_data"]
    if isinstance(versions_to_delete, str):
        versions_to_delete = [versions_to_delete]

    versions = version_data["versions"]

    if "__first_version__" in versions_to_delete:
        raise ValueError("Cannot delete first version")

    for version in versions_to_delete:
        if version not in versions:
            raise ValueError(f"Version {version!r} does not exist")

    current_version = versions.attrs["current_version"]
    while current_version in versions_to_delete:
        current_version = versions[current_version].attrs["prev_version"]

    for name in _walk(version_data):
        _delete_dataset(f, name, versions_to_delete)

    # find new prev_version which was not deleted
    versions_to_delete_set = set(versions_to_delete)
    for version_name in versions:
        if (
            version_name == "__first_version__"
            or version_name in versions_to_delete_set
        ):
            continue
        prev_version = versions[version_name].attrs["prev_version"]
        while prev_version in versions_to_delete_set:
            prev_version = _get_parent(versions, prev_version)
        versions[version_name].attrs["prev_version"] = prev_version

    # delete the version groups to delete
    for version_name in versions_to_delete:
        del versions[version_name]

    versions.attrs["current_version"] = current_version

    # Collect garbage here to handle intermittent slicing
    # issue; see https://github.com/deshaw/versioned-hdf5/pull/277
    # for a discussion about this.
    gc.collect()


def _get_parent(versions, version_name):
    return versions[version_name].attrs["prev_version"]


# Backwards compatibility
delete_version = delete_versions


def modify_metadata(
    f,
    dataset_name,
    *,
    chunks=None,
    dtype=None,
    fillvalue=None,
    # Filters
    compression: Any | None | Default = DEFAULT,
    compression_opts: Any | None | Default = DEFAULT,
    scaleoffset: int | None | Default = DEFAULT,
    shuffle: bool | Default = DEFAULT,
    fletcher32: bool | Default = DEFAULT,
):
    """
    Modify metadata for a versioned dataset in-place.

    The metadata is modified for all versions containing a dataset.

    `f` should be the h5py file or versioned_hdf5 VersionedHDF5File object.

    `dataset_name` is the name of the dataset in the version group(s).

    Metadata that may be modified are

    - `chunks`: must be compatible with the dataset shape
    - `dtype`: all data in the dataset is cast to the new dtype
    - `fillvalue`: see the note below
    - Filter settings (see :meth:`h5py.Group.create_dataset`):
      - `compression`
      - `compression_opts`
      - `scaleoffset`
      - `shuffle`
      - `fletcher32`

    If omitted, the given metadata is not modified.

    Notes
    -----
    For `fillvalue`, all values equal to the old fillvalue are updated to
    be the new fillvalue, regardless of whether they are explicitly stored or
    represented sparsely in the underlying HDF5 dataset. Also note that
    datasets without an explicitly set fillvalue have a default fillvalue
    equal to the default value of the dtype (e.g., 0. for float dtypes).

    For filters, passing a value of None is not the same as omitting the argument.
    For example, ``compression=None`` will decompress a dataset if it was compressed,
    and ``compression_opts=None`` will revert to the default options for the compression
    plugin, whereas omitting them will retain the previous preferences.
    """
    if isinstance(f, VersionedHDF5File):
        f = f.f

    def callback(dataset, version_name):  # noqa: ARG001
        _chunks = chunks or dataset.chunks
        _fillvalue = fillvalue if fillvalue is not None else dataset.fillvalue

        if isinstance(dataset, DatasetWrapper):
            dataset = dataset.dataset

        name = dataset.name[len(dataset.parent.name) + 1 :]
        if isinstance(dataset, (InMemoryDataset, InMemoryArrayDataset)):
            new_dataset = InMemoryArrayDataset(
                name,
                np.asarray(dataset._buffer, dtype=dtype),
                parent=tmp_parent,
                fillvalue=_fillvalue,
                chunks=_chunks,
            )
            if _fillvalue not in (None, dataset.fillvalue):
                new_dataset[new_dataset == dataset.fillvalue] = _fillvalue
        elif isinstance(dataset, InMemorySparseDataset):
            staged_changes = dataset.staged_changes
            if dtype not in (None, staged_changes.dtype):
                staged_changes = staged_changes.astype(dtype)
            if _fillvalue not in (None, dataset.fillvalue):
                staged_changes = staged_changes.refill(_fillvalue)
            if staged_changes is dataset.staged_changes:
                staged_changes = staged_changes.copy()

            new_dataset = InMemorySparseDataset(
                name,
                shape=dataset.shape,
                parent=tmp_parent,
                dtype=staged_changes.dtype,
                chunks=_chunks,
                fillvalue=_fillvalue,
            )
            new_dataset.staged_changes = staged_changes

        else:
            raise NotImplementedError(type(dataset))

        filters = Filters.from_dataset(dataset)
        if compression is not DEFAULT:
            filters.compression = compression
        if compression_opts is not DEFAULT:
            filters.compression_opts = compression_opts
        # compression_opts are implicitly set by create_dataset.
        # Undo them  if user calls modify_metadata(compression=None).
        elif filters.compression is None:
            filters.compression_opts = None

        if scaleoffset is not DEFAULT:
            filters.scaleoffset = scaleoffset
        if shuffle is not DEFAULT:
            filters.shuffle = shuffle
        if fletcher32 is not DEFAULT:
            filters.fletcher32 = fletcher32

        new_dataset.parent._set_filters(new_dataset.name, filters)

        return new_dataset

    newf = tmp_group(f)
    tmp_parent = InMemoryGroup(newf.create_group("__tmp_parent__").id)

    try:
        recreate_dataset(f, dataset_name, newf, callback=callback)

        swap(f, newf)
    finally:
        del newf[newf.name]


def swap(old, new):
    """
    Swap every dataset in old with the corresponding one in new

    Datasets in old that aren't in new are ignored.
    """
    move_names = []

    def _move(name, object):
        if isinstance(object, Dataset) and name in new:
            move_names.append(name)

    old.visititems(_move)
    for name in move_names:
        if new[name].is_virtual:
            # We cannot simply move virtual datasets, because they will still
            # point to the old raw_data location. So instead, we have to
            # recreate them, pointing to the new raw_data.
            oldd = old[name]
            newd = new[name]

            def _normalize(path):
                return path if path.endswith("/") else path + "/"

            def _replace_prefix(path, name1, name2):
                """Replace the prefix name1 with name2 in path"""
                name1 = _normalize(name1)
                name2 = _normalize(name2)
                return name2 + path[len(name1) :]

            def _new_vds_layout(d, name1, name2):
                """Recreate a VirtualLayout for d, replacing name1 with name2 in the
                source dset name
                """
                virtual_sources = d.virtual_sources()
                layout = VirtualLayout(d.shape, dtype=d.dtype)
                for vmap in virtual_sources:
                    vspace, fname, dset_name, src_space = vmap
                    assert dset_name.startswith(name1)
                    dset_name = _replace_prefix(dset_name, name1, name2)
                    layout.dcpl.set_virtual(
                        vspace,
                        fname.encode("utf-8"),
                        dset_name.encode("utf-8"),
                        src_space,
                    )
                return layout

            old_layout = _new_vds_layout(oldd, old.name, new.name)
            new_layout = _new_vds_layout(newd, new.name, old.name)
            old_fillvalue = old[name].fillvalue
            new_fillvalue = new[name].fillvalue
            old_attrs = dict(old[name].attrs)
            new_attrs = dict(new[name].attrs)
            del old[name]
            old.create_virtual_dataset(name, new_layout, fillvalue=new_fillvalue)
            for k, v in new_attrs.items():
                if isinstance(v, str) and v.startswith(new.name):
                    v = _replace_prefix(v, new.name, old.name)
                old[name].attrs[k] = v
            del new[name]
            new.create_virtual_dataset(name, old_layout, fillvalue=old_fillvalue)
            for k, v in old_attrs.items():
                if isinstance(v, str) and v.startswith(old.name):
                    v = _replace_prefix(v, old.name, new.name)
                new[name].attrs[k] = v
        else:
            # Invalidate any InMemoryGroups that point to these groups
            delete = []
            for bind in InMemoryGroup._instances:
                if get_name(bind) and (
                    get_name(bind).startswith(get_name(old.id))
                    or get_name(bind).startswith(get_name(new.id))
                ):
                    delete.append(bind)
            for d in delete:
                del InMemoryGroup._instances[d]
            old.move(name, posixpath.join(new.name, name + "__tmp"))
            new.move(name, posixpath.join(old.name, name))
            new.move(name + "__tmp", name)
