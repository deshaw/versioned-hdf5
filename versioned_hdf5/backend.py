import numpy as np
from h5py import VirtualLayout, VirtualSource
from ndindex import Slice, ndindex, Tuple

from .hashtable import Hashtable
from .slicetools import split_chunks

DEFAULT_CHUNK_SIZE = 2**12

def get_chunks(shape, dtype, chunk_size):
    # TODO: Implement this
    if len(shape) > 1:
        raise NotImplementedError
    return (chunk_size,)

def initialize(f):
    version_data = f.create_group('_version_data')
    versions = version_data.create_group('versions')
    versions.create_group('__first_version__')
    versions.attrs['current_version'] = '__first_version__'

def create_base_dataset(f, name, *, shape=None, data=None, dtype=None,
    chunks=True, compression=None, compression_opts=None, fillvalue=None):

    # Validate shape (based on h5py._hl.dataset.make_new_dset
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
    if dtype is None:
        # https://github.com/h5py/h5py/issues/1474
        dtype = data.dtype

    ndims = len(shape)
    if isinstance(chunks, int) and not isinstance(chunks, bool):
        chunks = (chunks,)
    if chunks in [True, None]:
        if ndims <= 1:
            chunks = (DEFAULT_CHUNK_SIZE,)
        else:
            raise NotImplementedError("chunks must be specified for multi-dimensional datasets")
    group = f['_version_data'].create_group(name)
    dataset = group.create_dataset('raw_data', shape=(0,) + chunks[1:],
                                   chunks=chunks, maxshape=(None,) + chunks[1:],
                                   dtype=dtype, compression=compression,
                                   compression_opts=compression_opts,
                                   fillvalue=fillvalue)
    dataset.attrs['chunks'] = chunks
    return write_dataset(f, name, data, chunks=chunks)

def write_dataset(f, name, data, chunks=None, compression=None,
                  compression_opts=None, fillvalue=None):
    if name not in f['_version_data']:
        return create_base_dataset(f, name, data=data, chunks=chunks,
            compression=compression, compression_opts=compression_opts, fillvalue=fillvalue)

    ds = f['_version_data'][name]['raw_data']
    if isinstance(chunks, int) and not isinstance(chunks, bool):
        chunks = (chunks,)
    if chunks is None:
        chunks = tuple(ds.attrs['chunks'])
    else:
        if chunks != tuple(ds.attrs['chunks']):
            raise ValueError("Chunk size specified but doesn't match already existing chunk size")

    if compression or compression_opts:
        raise ValueError("Compression options can only be specified for the first version of a dataset")
    if fillvalue is not None and fillvalue != ds.fillvalue:
        raise ValueError(f"fillvalues do not match ({fillvalue} != {ds.fillvalue})")
    if data.dtype != ds.dtype:
        raise ValueError(f"dtypes do not match ({data.dtype} != {ds.dtype})")
    # TODO: Handle more than one dimension
    old_shape = ds.shape
    hashtable = Hashtable(f, name)
    slices = {}
    slices_to_write = {}
    chunk_size = chunks[0]
    for s in split_chunks(data.shape, chunks):
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

def write_dataset_chunks(f, name, data_dict):
    """
    data_dict should be a dictionary mapping chunk_size index to either an
    array for that chunk, or a slice into the raw data for that chunk

    """
    if name not in f['_version_data']:
        raise NotImplementedError("Use write_dataset() if the dataset does not yet exist")

    ds = f['_version_data'][name]['raw_data']
    chunks = tuple(ds.attrs['chunks'])
    chunk_size = chunks[0]

    shape = tuple(max(c.args[i].stop for c in data_dict) for i in
    range(len(chunks)))
    for c in split_chunks(shape, chunks):
        if c not in data_dict:
            raise ValueError(f"data_dict does not include all chunks ({c})")

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
        ds[raw_slice.raw] = data_s
    return slices

def create_virtual_dataset(f, version_name, name, slices, attrs=None, fillvalue=None):
    raw_data = f['_version_data'][name]['raw_data']
    chunks = tuple(raw_data.attrs['chunks'])
    slices = {c: s.reduce() for c, s in slices.items()}

    shape = tuple([max(c.args[i].stop for c in slices) for i in range(len(chunks))])
    # Chunks in the raw dataset are expanded along the first dimension only.
    # Since the chunks are pointed to by virtual datasets, it doesn't make
    # sense to expand the chunks in the raw dataset along multiple dimensions
    # (the true layout of the chunks in the raw dataset is irrelevant).
    for c, s in slices.items():
        if len(c.args[0]) != len(s):
            raise ValueError("Inconsistent slices dictionary")

    layout = VirtualLayout(shape, dtype=raw_data.dtype)
    vs = VirtualSource('.', name=raw_data.name, shape=raw_data.shape, dtype=raw_data.dtype)

    for c, s in slices.items():
        # TODO: This needs to handle more than one dimension
        idx = Tuple(s, *Tuple(*[slice(0, i) for i in shape]).as_subindex(c).args[1:])
        assert c.newshape(shape) == vs[idx.raw].shape, (c, shape, s)
        layout[c.raw] = vs[idx.raw]

    virtual_data = f['_version_data/versions'][version_name].create_virtual_dataset(name, layout, fillvalue=fillvalue)

    if attrs:
        for k, v in attrs.items():
            virtual_data.attrs[k] = v
    return virtual_data
