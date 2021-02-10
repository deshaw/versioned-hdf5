import numpy as np
from h5py import VirtualLayout, VirtualSource, Dataset
from h5py._hl.vds import VDSmap
from h5py.h5i import get_name
from ndindex import Slice, ndindex, Tuple, ChunkSize

import posixpath as pp

from .hashtable import Hashtable

DEFAULT_CHUNK_SIZE = 2**12

def normalize_dtype(dtype):
    return np.array([], dtype=dtype).dtype

def get_chunks(shape, dtype, chunk_size):
    # TODO: Implement this
    if len(shape) > 1:
        raise NotImplementedError
    return (chunk_size,)

def initialize(f):
    import datetime
    from .versions import TIMESTAMP_FMT

    version_data = f.create_group('_version_data')
    versions = version_data.create_group('versions')
    versions.create_group('__first_version__')
    versions.attrs['current_version'] = '__first_version__'
    ts = datetime.datetime.now(datetime.timezone.utc)
    versions['__first_version__'].attrs['timestamp'] = ts.strftime(TIMESTAMP_FMT)


def create_base_dataset(f, name, *, shape=None, data=None, dtype=None,
    chunks=True, compression=None, compression_opts=None, fillvalue=None):

    # Validate shape (based on h5py._hl.dataset.make_new_dset)
    if shape is None:
        if data is None:
            if dtype is None:
                raise TypeError("One of data, shape or dtype must be specified")
            raise NotImplementedError("empty datasets are not yet implemented")
        shape = data.shape
    else:
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if data is not None and (np.product(shape, dtype=np.ulonglong) != np.product(data.shape, dtype=np.ulonglong)):
            raise ValueError("Shape tuple is incompatible with data")

    ndims = len(shape)
    if isinstance(chunks, int) and not isinstance(chunks, bool):
        chunks = (chunks,)
    if chunks in [True, None]:
        if ndims <= 1:
            chunks = (DEFAULT_CHUNK_SIZE,)
        else:
            raise NotImplementedError("chunks must be specified for multi-dimensional datasets")
    group = f['_version_data'].create_group(name)

    if dtype is None:
        # https://github.com/h5py/h5py/issues/1474
        dtype = data.dtype
    dtype = normalize_dtype(dtype)
    if dtype.metadata and ('vlen' in dtype.metadata or 'h5py_encoding' in dtype.metadata):
        # h5py string dtype
        # (https://h5py.readthedocs.io/en/2.10.0/strings.html). Setting the
        # fillvalue in this case doesn't work
        # (https://github.com/h5py/h5py/issues/941).
        if fillvalue not in [0, '', b'', None]:
            raise ValueError("Non-default fillvalue not supported for variable length strings")
        fillvalue = None
    dataset = group.create_dataset('raw_data', shape=(0,) + chunks[1:],
                                   chunks=tuple(chunks), maxshape=(None,) + chunks[1:],
                                   dtype=dtype, compression=compression,
                                   compression_opts=compression_opts,
                                   fillvalue=fillvalue)
    dataset.attrs['chunks'] = chunks
    return write_dataset(f, name, data, chunks=chunks)

def write_dataset(f, name, data, chunks=None, dtype=None, compression=None,
                  compression_opts=None, fillvalue=None):

    if name not in f['_version_data']:
        return create_base_dataset(f, name, data=data, dtype=dtype,
                                   chunks=chunks, compression=compression,
                                   compression_opts=compression_opts, fillvalue=fillvalue)

    ds = f['_version_data'][name]['raw_data']
    if isinstance(chunks, int) and not isinstance(chunks, bool):
        chunks = (chunks,)
    if chunks is None:
        chunks = tuple(ds.attrs['chunks'])
    else:
        if chunks != tuple(ds.attrs['chunks']):
            raise ValueError("Chunk size specified but doesn't match already existing chunk size")

    if dtype is not None:
        if dtype != ds.dtype:
            raise ValueError("dtype specified but doesn't match already existing dtype")

    if compression and compression != ds.compression or compression_opts and compression_opts != ds.compression_opts:
        raise ValueError("Compression options can only be specified for the first version of a dataset")
    if fillvalue is not None and fillvalue != ds.fillvalue:
        dtype = ds.dtype
        if dtype.metadata and ('vlen' in dtype.metadata or 'h5py_encoding' in dtype.metadata):
            # Variable length string dtype. The ds.fillvalue will be None in
            # this case (see create_virtual_dataset() below)
            pass
        else:
            raise ValueError(f"fillvalues do not match ({fillvalue} != {ds.fillvalue})")
    if data.dtype != ds.dtype:
        raise ValueError(f"dtypes do not match ({data.dtype} != {ds.dtype})")
    # TODO: Handle more than one dimension
    old_shape = ds.shape
    hashtable = Hashtable(f, name)
    slices = {}
    slices_to_write = {}
    chunk_size = chunks[0]

    if len(data.shape) != 0:
        for s in ChunkSize(chunks).indices(data.shape):
            idx = hashtable.largest_index
            data_s = data[s.raw]
            raw_slice = Slice(idx*chunk_size, idx*chunk_size + data_s.shape[0])
            data_hash = hashtable.hash(data_s)
            raw_slice2 = hashtable.setdefault(data_hash, raw_slice)
            if raw_slice2 == raw_slice:
                slices_to_write[raw_slice] = s
            slices[s] = raw_slice2

        ds.resize((old_shape[0] + len(slices_to_write)*chunk_size,) + chunks[1:])
        for raw_slice, s in slices_to_write.items():
            data_s = data[s.raw]
            idx = Tuple(raw_slice, *[slice(0, i) for i in data_s.shape[1:]])
            ds[idx.raw] = data[s.raw]
    return slices

def write_dataset_chunks(f, name, data_dict, shape=None):
    """
    data_dict should be a dictionary mapping chunk_size index to either an
    array for that chunk, or a slice into the raw data for that chunk

    """
    if name not in f['_version_data']:
        raise NotImplementedError("Use write_dataset() if the dataset does not yet exist")

    ds = f['_version_data'][name]['raw_data']
    chunks = tuple(ds.attrs['chunks'])
    chunk_size = chunks[0]

    if shape is None:
        shape = tuple(max(c.args[i].stop for c in data_dict) for i in
                      range(len(chunks)))
    # all_chunks = list(ChunkSize(chunks).indices(shape))
    # for c in data_dict:
    #     if c not in all_chunks:
    #         raise ValueError(f"data_dict contains extra chunks ({c})")

    hashtable = Hashtable(f, name)
    slices = {i: None for i in data_dict}
    data_to_write = {}
    for chunk, data_s in data_dict.items():
        if not isinstance(data_s, (slice, tuple, Tuple, Slice)) and data_s.dtype != ds.dtype:
            raise ValueError(f"dtypes do not match ({data_s.dtype} != {ds.dtype})")

        idx = hashtable.largest_index
        if isinstance(data_s, (slice, tuple, Tuple, Slice)):
            slices[chunk] = ndindex(data_s)
        else:
            raw_slice = Slice(idx*chunk_size, idx*chunk_size + data_s.shape[0])
            data_hash = hashtable.hash(data_s)
            raw_slice2 = hashtable.setdefault(data_hash, raw_slice)
            if raw_slice2 == raw_slice:
                data_to_write[raw_slice] = data_s
            slices[chunk] = raw_slice2

    assert None not in slices.values()
    old_shape = ds.shape
    ds.resize((old_shape[0] + len(data_to_write)*chunk_size,) + chunks[1:])
    for raw_slice, data_s in data_to_write.items():
        c = (raw_slice.raw,) + tuple(slice(0, i) for i in data_s.shape[1:])
        ds[c] = data_s
    return slices

def create_virtual_dataset(f, version_name, name, shape, slices, attrs=None, fillvalue=None):
    raw_data = f['_version_data'][name]['raw_data']
    slices = {c: s.reduce() for c, s in slices.items()}

    if len(raw_data) == 0:
        shape = ()
        layout = VirtualLayout((1,), dtype=raw_data.dtype)
        vs = VirtualSource('.', name=raw_data.name, shape=(1,), dtype=raw_data.dtype)
        layout[0] = vs[()]
    else:
        # Chunks in the raw dataset are expanded along the first dimension only.
        # Since the chunks are pointed to by virtual datasets, it doesn't make
        # sense to expand the chunks in the raw dataset along multiple dimensions
        # (the true layout of the chunks in the raw dataset is irrelevant).
        for c, s in slices.items():
            if len(c.args[0]) != len(s):
                raise ValueError(f"Inconsistent slices dictionary ({c.args[0]}, {s})")

        layout = VirtualLayout(shape, dtype=raw_data.dtype)
        vs = VirtualSource('.', name=raw_data.name, shape=raw_data.shape, dtype=raw_data.dtype)

        for c, s in slices.items():
            if c.isempty():
                continue
            # idx = Tuple(s, *Tuple(*[slice(0, i) for i in shape[1:]]).as_subindex(Tuple(*c.args[1:])).args)
            S = [Slice(0, shape[i], 1).as_subindex(c.args[i]) for i in range(1, len(shape))]
            idx = Tuple(s, *S)
            # assert c.newshape(shape) == vs[idx.raw].shape, (c, shape, s)
            layout[c.raw] = vs[idx.raw]

    dtype = raw_data.dtype
    if dtype.metadata and ('vlen' in dtype.metadata or 'h5py_encoding' in dtype.metadata):
        # Variable length string dtype
        # (https://h5py.readthedocs.io/en/2.10.0/strings.html). Setting the
        # fillvalue in this case doesn't work
        # (https://github.com/h5py/h5py/issues/941).
        if fillvalue not in [0, '', b'', None]:
            raise ValueError("Non-default fillvalue not supported for variable length strings")
        fillvalue = None

    virtual_data = f['_version_data/versions'][version_name].create_virtual_dataset(name, layout, fillvalue=fillvalue)

    if attrs:
        for k, v in attrs.items():
            virtual_data.attrs[k] = v
    virtual_data.attrs['raw_data'] = raw_data.name
    virtual_data.attrs['chunks'] = raw_data.chunks
    return virtual_data

def recreate_dataset(f, name, newf, callback=None):
    """
    Recreate dataset from all versions into newf

    newf should be a versioned hdf5 file/group that is already initialized (it
    may or may not be in the same physical file as f).

    callback should be a function with the signature

    callback(dataset, version_name)

    It will be called on every dataset in every version. It should return the
    dataset to be used for the new version. The dataset and its containing
    group should not be modified in-place. If a new copy of a dataset is to be
    used, it should be one of the dataset classes in versioned_hdf5.wrappers,
    and should placed in a temporary group, which may be deleted after
    recreate_dataset() is done. The callback may also return None, in which
    case the dataset is deleted for the given version.

    """
    from .versions import all_versions
    from .wrappers import InMemoryGroup, InMemoryDataset, InMemorySparseDataset

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
    from .versions import all_versions

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

def swap(old, new):
    """
    Swap every dataset in old with the corresponding one in new

    Datasets in old that aren't in new are ignored.
    """
    from .wrappers import _groups

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
                if get_name(bind).startswith(get_name(old.id)) or get_name(bind).startswith(get_name(new.id)):
                    delete.append(bind)
            for d in delete:
                del _groups[d]
            old.move(name, pp.join(new.name, name + '__tmp'))
            new.move(name, pp.join(old.name, name))
            new.move(name + '__tmp', name)
