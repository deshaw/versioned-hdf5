from h5py import VirtualLayout, Dataset
from h5py._hl.vds import VDSmap
from h5py.h5i import get_name

from ndindex import Slice

import numpy as np

from copy import deepcopy
import posixpath as pp

from .versions import all_versions
from .wrappers import (InMemoryGroup, DatasetWrapper, InMemoryDataset,
                       InMemoryArrayDataset, InMemorySparseDataset, _groups)
from .api import VersionedHDF5File
from .backend import (create_base_dataset, write_dataset,
                      write_dataset_chunks, create_virtual_dataset,
                      initialize)

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

def delete_version(f, version):
    """
    Completely delete the version named `version` from the versioned file `f`.

    This function should be used instead of deleting the version group
    directly, as this will not delete the underlying data that is unique to
    the version.
    """
    if isinstance(f, VersionedHDF5File):
        f = f.f

    versions = f['_version_data/versions']

    def callback(dataset, version_name):
        if version_name == version:
            return
        return dataset

    newf = tmp_group(f)

    def _get(name):
        recreate_dataset(f, name, newf, callback=callback)

    versions[version].visit(_get)

    swap(f, newf)
    # swap() will swap out the datasets that are left intact. Any dataset in
    # the version that is not in newf should be deleted entirely, as that
    # means that it only existed in this version.
    to_delete = []
    def _visit(name, object):
        if isinstance(object, Dataset):
            if name not in newf['_version_data']:
                to_delete.append(name)
    versions[version].visititems(_visit)

    for name in to_delete:
        del f['_version_data'][name]
    del f['_version_data/versions'][version]

    del newf[newf.name]

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
                    layout.sources.append(new_vmap)
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
