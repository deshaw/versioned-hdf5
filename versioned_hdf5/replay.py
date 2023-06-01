from __future__ import annotations
from typing import List, Iterable, Union, Dict, Any
from h5py import (
    VirtualLayout,
    h5s,
    HLObject,
    Dataset,
    Group,
    File,
    __version__ as h5py_version
)
from h5py._hl.vds import VDSmap
from h5py._hl.selections import select
from h5py.h5i import get_name

from ndindex import Slice, ChunkSize, Tuple
from ndindex.ndindex import NDIndex

import numpy as np

from copy import deepcopy
import posixpath as pp
from collections import defaultdict

from .versions import all_versions
from .wrappers import (InMemoryGroup, DatasetWrapper, InMemoryDataset,
                       InMemoryArrayDataset, InMemorySparseDataset, _groups)
from .api import VersionedHDF5File
from .backend import (create_base_dataset, write_dataset,
                      write_dataset_chunks, create_virtual_dataset,
                      initialize)
from .slicetools import spaceid_to_slice
from .hashtable import Hashtable

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

    Note: this function is only for advanced usage. Typical use-cases should
    use :func:`delete_version()` or :func:`modify_metadata()`.
    """
    if isinstance(f, VersionedHDF5File):
        f = f.f

    raw_data = f['_version_data'][name]['raw_data']

    dtype = raw_data.dtype
    chunks = raw_data.chunks
    compression = raw_data.compression
    compression_opts = raw_data.compression_opts
    fillvalue = raw_data.fillvalue

    first = True
    for version_name in all_versions(f):
        if name in f['_version_data/versions'][version_name]:
            group = InMemoryGroup(f['_version_data/versions'][version_name].id,
                                  _committed=True)

            dataset = group[name]
            if callback:
                dataset = callback(dataset, version_name)
                if dataset is None:
                    continue

            dtype = dataset.dtype
            shape = dataset.shape
            chunks = dataset.chunks
            compression = dataset.compression
            compression_opts = dataset.compression_opts
            fillvalue = dataset.fillvalue
            attrs = dataset.attrs
            if first:
                create_base_dataset(newf, name,
                                    data=np.empty((0,)*len(dataset.shape),
                                                  dtype=dtype),
                                    dtype=dtype,
                                    chunks=chunks,
                                    compression=compression,
                                    compression_opts=compression_opts,
                                    fillvalue=fillvalue)
                first = False
            # Read in all the chunks of the dataset (we can't assume the new
            # hash table has the raw data in the same locations, even if the
            # data is unchanged).
            if isinstance(dataset, (InMemoryDataset, InMemorySparseDataset)):
                for c, index in dataset.data_dict.copy().items():
                    if isinstance(index, Slice):
                        dataset[c.raw]
                        assert not isinstance(dataset.data_dict[c], Slice)
                slices = write_dataset_chunks(newf, name, dataset.data_dict)
            else:
                slices = write_dataset(newf, name, dataset)
            create_virtual_dataset(newf, version_name, name, shape, slices,
                                   attrs=attrs, fillvalue=fillvalue)

def tmp_group(f):
    """
    Create a temporary group in `f` for use with :func:`recreate_dataset`.
    """
    if isinstance(f, VersionedHDF5File):
        f = f.f

    if '__tmp__' not in f['_version_data']:
        tmp = f['_version_data'].create_group('__tmp__')
        initialize(tmp)
        for version_name in all_versions(f):
            group = f['_version_data/versions'][version_name]
            new_group = tmp['_version_data/versions'].create_group(version_name)
            for k, v in group.attrs.items():
                new_group.attrs[k] = v
    else:
        tmp = f['_version_data/__tmp__']
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
         if 'vlen' in data.dtype.metadata:
             if (h5py_version.startswith('3') and
                 data.dtype.metadata['vlen'] == str):
                 return bytes()
             return data.dtype.metadata['vlen']()
         elif 'h5py_encoding' in data.dtype.metadata:
             return data.dtype.type()
    return np.zeros((), dtype=data.dtype)[()]


def _recreate_raw_data(
    f: VersionedHDF5File,
    name: str,
    versions_to_delete: Iterable[str],
    tmp: bool = False
) -> Dict[NDIndex, NDIndex]:
    """
    Return a new raw data set for a dataset without the chunks from
    versions_to_delete.

    If no chunks would be left, i.e., the dataset does not appear in any
    version not in versions_to_delete, None is returned.

    If tmp is True, the new raw dataset is called '_tmp_raw_data' and is
    placed alongside the existing raw dataset. Otherwise the existing raw
    dataset is replaced.

    """
    chunks_map = defaultdict(dict)

    for version_name in all_versions(f):
        if (version_name in versions_to_delete
            or name not in f['_version_data/versions'][version_name]):
            continue

        dataset = f['_version_data/versions'][version_name][name]

        if dataset.is_virtual:
            virtual_sources = dataset.virtual_sources()
            slice_map = {spaceid_to_slice(i.vspace):
                         spaceid_to_slice(i.src_space) for i in
                         virtual_sources}
        else:
            slice_map = {}
        chunks_map[version_name].update(slice_map)

    chunks_to_keep = set().union(*[map.values() for map in
                                 chunks_map.values()])

    chunks_to_keep = sorted(chunks_to_keep, key=lambda i: i.args[0].args[0])

    raw_data = f['_version_data'][name]['raw_data']
    chunks = ChunkSize(raw_data.chunks)
    new_shape = (len(chunks_to_keep)*chunks[0], *chunks[1:])

    new_raw_data = f['_version_data'][name].create_dataset(
        '_tmp_raw_data', shape=new_shape, maxshape=(None,)+chunks[1:],
        chunks=raw_data.chunks, dtype=raw_data.dtype,
        compression=raw_data.compression,
        compression_opts=raw_data.compression_opts,
        fillvalue=raw_data.fillvalue)
    for key, val in raw_data.attrs.items():
        new_raw_data.attrs[key] = val

    r = raw_data[:]
    n = np.full(new_raw_data.shape, _get_np_fillvalue(raw_data), dtype=new_raw_data.dtype)
    raw_data_chunks_map = {}
    for new_chunk, chunk in zip(chunks.indices(new_shape), chunks_to_keep):
        # Shrink new_chunk to the size of chunk, in case chunk isn't a full
        # chunk in one of the dimensions.
        # TODO: Implement something in ndindex to do this.
        new_chunk = Tuple(
            *[Slice(new_chunk.args[i].start,
                      new_chunk.args[i].start+len(chunk.args[i]))
              for i in range(len(new_chunk.args))])
        raw_data_chunks_map[chunk] = new_chunk
        n[new_chunk.raw] = r[chunk.raw]

    new_raw_data[:] = n
    if not tmp:
        del f['_version_data'][name]['raw_data']
        f['_version_data'][name].move('_tmp_raw_data', 'raw_data')

    return raw_data_chunks_map

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
    old_hashtable = Hashtable(f, name)
    new_hash_table = Hashtable(f, name, hash_table_name='_tmp_hash_table')
    old_inverse = old_hashtable.inverse()

    for old_chunk, new_chunk in raw_data_chunks_map.items():
        if isinstance(old_chunk, Tuple):
            old_chunk = old_chunk.args[0]
        if isinstance(new_chunk, Tuple):
            new_chunk = new_chunk.args[0]

        new_hash_table[old_inverse[old_chunk.reduce()]] = new_chunk

    new_hash_table.write()

    if not tmp:
        del f['_version_data'][name]['hash_table']
        f['_version_data'][name].move('_tmp_hash_table', 'hash_table')

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

    """
    raw_data = f['_version_data'][name]['raw_data']

    for version_name in versions:
        if name not in f['_version_data/versions'][version_name]:
            continue

        group = f['_version_data/versions'][version_name]
        dataset = group[name]


        # See the comments in create_virtual_dataset
        layout = VirtualLayout(dataset.shape, dtype=dataset.dtype)
        layout_has_sources = hasattr(layout, 'sources')

        if not layout_has_sources:
            from h5py import _selector
            layout._src_filenames.add(b'.')
            space = h5s.create_simple(dataset.shape)
            selector = _selector.Selector(space)

        # If a dataset has no data except for the fillvalue, it will not be virtual
        if dataset.is_virtual:
            virtual_sources = dataset.virtual_sources()
            for vmap in virtual_sources:
                vspace, fname, dset_name, src_space = vmap
                fname = fname.encode('utf-8')
                assert fname == b'.', fname

                vslice = spaceid_to_slice(vspace)
                src_slice = spaceid_to_slice(src_space)
                if src_slice not in raw_data_chunks_map:
                    raise ValueError(f"Could not find the chunk for {vslice} ({src_slice} in the old raw dataset) for {name!r} in {version_name!r}")
                new_src_slice = raw_data_chunks_map[src_slice]

                if not layout_has_sources:
                    key = new_src_slice.raw
                    vs_sel = select(raw_data.shape, key, dataset=None)

                    sel = selector.make_selection(vslice.raw)
                    layout.dcpl.set_virtual(
                        sel.id, b'.', raw_data.name.encode('utf-8'), vs_sel.id
                    )
                else:
                    vs_sel = select(raw_data.shape, new_src_slice.raw, None)
                    layout_sel = select(dataset.shape, vslice.raw, None)
                    new_vmap = VDSmap(layout_sel.id, fname, dset_name, vs_sel.id)
                    layout.sources.append(new_vmap)

        head, tail = pp.split(name)
        tmp_name = '_tmp_' + tail
        tmp_path = pp.join(head, tmp_name)
        dtype = raw_data.dtype
        fillvalue = dataset.fillvalue
        if dtype.metadata and ('vlen' in dtype.metadata or 'h5py_encoding' in dtype.metadata):
            # Variable length string dtype
            # (https://h5py.readthedocs.io/en/2.10.0/strings.html). Setting the
            # fillvalue in this case doesn't work
            # (https://github.com/h5py/h5py/issues/941).
            if fillvalue not in [0, '', b'', None]:
                raise ValueError("Non-default fillvalue not supported for variable length strings")
            fillvalue = None
        tmp_dataset = group.create_virtual_dataset(tmp_path, layout, fillvalue=fillvalue)

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
    return not f['_version_data/versions'][version][name].is_virtual


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
    return name in f['_version_data/versions'][version]


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
    for version in versions:
        if _exists_in_version(f, name, version):
            if not _is_empty(f, name, version):
                return False
    return True


def _delete_dataset(f: VersionedHDF5File, name: str, versions_to_delete: Iterable[str]):
    """Delete the given dataset from the versions."""
    version_data = f['_version_data']
    versions = version_data['versions']

    if name == 'versions':
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


def _walk(g: HLObject, prefix: str = '') -> List[str]:
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
    List[str]
        List of the names of `raw_data` datasets in g
    """
    datasets = []
    for name in g:
        obj = g[name]
        if isinstance(obj, Group):
            if 'raw_data' in obj:
                datasets.append(prefix + name)
            else:
                datasets.extend(_walk(obj, prefix + name + '/'))

    return datasets


def delete_versions(
    f: Union[VersionedHDF5File, File],
    versions_to_delete: Iterable[str]
):
    """Completely delete the given versions from a file

    This function should be used instead of deleting the version group
    directly, as this will not delete the underlying data that is unique to
    the version.
    """
    if isinstance(f, VersionedHDF5File):
        f = f.f

    version_data = f['_version_data']
    if isinstance(versions_to_delete, str):
        versions_to_delete = [versions_to_delete]

    versions = version_data['versions']

    if '__first_version__' in versions_to_delete:
        raise ValueError("Cannot delete first version")

    for version in versions_to_delete:
        if version not in versions:
            raise ValueError(f"Version {version!r} does not exist")

    current_version = versions.attrs['current_version']
    while current_version in versions_to_delete:
        current_version = versions[current_version].attrs['prev_version']

    for name in _walk(version_data):
        _delete_dataset(f, name, versions_to_delete)

    for version_name in versions_to_delete:
        prev_version  = versions[version_name].attrs['prev_version']
        for _version in versions:
            if _version == '__first_version__':
                continue
            v = versions[_version]
            if v.attrs['prev_version'] == version_name:
                v.attrs['prev_version'] = prev_version
        del versions[version_name]

    versions.attrs['current_version'] = current_version

# Backwards compatibility
delete_version = delete_versions

def modify_metadata(f, dataset_name, *, chunks=None, compression=None,
                    compression_opts=None, dtype=None, fillvalue=None):
    """
    Modify metadata for a versioned dataset in-place.

    The metadata is modified for all versions containing a dataset.

    `f` should be the h5py file or versioned_hdf5 VersionedHDF5File object.

    `dataset_name` is the name of the dataset in the version group(s).

    Metadata that may be modified are

    - `chunks`: must be compatible with the dataset shape
    - `compression`: see `h5py.Group.create_dataset()`
    - `compression_opts`: see `h5py.Group.create_dataset()`
    - `dtype`: all data in the dataset is cast to the new dtype
    - `fillvalue`: see the note below

    If set to `None` (the default), the given metadata is not modified.

    Note for `fillvalue`, all values equal to the old fillvalue are updated to
    be the new fillvalue, regardless of whether they are explicitly stored or
    represented sparsely in the underlying HDF5 dataset. Also note that
    datasets without an explicitly set fillvalue have a default fillvalue
    equal to the default value of the dtype (e.g., 0. for float dtypes).

    """
    if isinstance(f, VersionedHDF5File):
        f = f.f

    def callback(dataset, version_name):
        _chunks = chunks or dataset.chunks
        _fillvalue = fillvalue or dataset.fillvalue

        if isinstance(dataset, DatasetWrapper):
            dataset = dataset.dataset

        name = dataset.name[len(dataset.parent.name)+1:]
        if isinstance(dataset, (InMemoryDataset, InMemoryArrayDataset)):
            new_dataset = InMemoryArrayDataset(name, dataset[()], tmp_parent,
                                               fillvalue=_fillvalue,
                                               chunks=_chunks)
            if _fillvalue:
                new_dataset[new_dataset == dataset.fillvalue] = _fillvalue
        elif isinstance(dataset, InMemorySparseDataset):
            new_dataset = InMemorySparseDataset(name, shape=dataset.shape,
                                                dtype=dataset.dtype,
                                                parent=tmp_parent,
                                                chunks=_chunks,
                                                fillvalue=_fillvalue)
            new_dataset.data_dict = deepcopy(dataset.data_dict)
            if _fillvalue:
                for a in new_dataset.data_dict.values():
                    a[a == dataset.fillvalue] = _fillvalue
        else:
            raise NotImplementedError(type(dataset))

        if compression:
            new_dataset.compression = compression
        if compression_opts:
            new_dataset.compression_opts = compression_opts

        if dtype:
            return new_dataset.as_dtype(name, dtype, tmp_parent)

        return new_dataset

    newf = tmp_group(f)
    tmp_parent = InMemoryGroup(newf.create_group('__tmp_parent__').id)

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
        if isinstance(object, Dataset):
            if name in new:
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
                return path if path.endswith('/') else path + '/'

            def _replace_prefix(path, name1, name2):
                """Replace the prefix name1 with name2 in path"""
                name1 = _normalize(name1)
                name2 = _normalize(name2)
                return name2 + path[len(name1):]

            def _new_vds_layout(d, name1, name2):
                """Recreate a VirtualLayout for d, replacing name1 with name2 in the source dset name"""
                virtual_sources = d.virtual_sources()
                layout = VirtualLayout(d.shape, dtype=d.dtype)
                for vmap in virtual_sources:
                    vspace, fname, dset_name, src_space = vmap
                    assert dset_name.startswith(name1)
                    dset_name = _replace_prefix(dset_name, name1, name2)
                    fname = fname.encode('utf-8')
                    new_vmap = VDSmap(vspace, fname, dset_name, src_space)
                    # h5py 3.3 changed the VirtualLayout code. See
                    # https://github.com/h5py/h5py/pull/1905.
                    if hasattr(layout, 'sources'):
                        layout.sources.append(new_vmap)
                    else:
                        layout.dcpl.set_virtual(vspace, fname,
                                                dset_name.encode('utf-8'), src_space)
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
            for bind in _groups:
                if get_name(bind) and (get_name(bind).startswith(get_name(old.id)) or get_name(bind).startswith(get_name(new.id))):
                    delete.append(bind)
            for d in delete:
                del _groups[d]
            old.move(name, pp.join(new.name, name + '__tmp'))
            new.move(name, pp.join(old.name, name))
            new.move(name + '__tmp', name)
