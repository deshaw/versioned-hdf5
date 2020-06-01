from h5py import VirtualLayout, VirtualSource
from ndindex import Slice, ndindex

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
    chunk_size=None, compression=None, compression_opts=None, fillvalue=None):
    chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
    group = f['_version_data'].create_group(name)
    if dtype is None:
        # https://github.com/h5py/h5py/issues/1474
        dtype = data.dtype
    dataset = group.create_dataset('raw_data', shape=(0,),
                                   chunks=(chunk_size,), maxshape=(None,),
                                   dtype=dtype, compression=compression,
                                   compression_opts=compression_opts,
                                   fillvalue=fillvalue)

    dataset.attrs['chunk_size'] = chunk_size
    return write_dataset(f, name, data, chunk_size=chunk_size)

def write_dataset(f, name, data, chunk_size=None, compression=None,
                  compression_opts=None, fillvalue=None):
    if name not in f['_version_data']:
        return create_base_dataset(f, name, data=data, chunk_size=chunk_size,
            compression=compression, compression_opts=compression_opts, fillvalue=fillvalue)

    ds = f['_version_data'][name]['raw_data']
    if chunk_size is None:
        chunk_size = ds.attrs['chunk_size']
    else:
        if chunk_size != ds.attrs['chunk_size']:
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
    slices = []
    slices_to_write = {}
    for s in split_chunks(data.shape, chunk_size):
        idx = hashtable.largest_index
        data_s = data[s.raw]
        raw_slice = Slice(idx*chunk_size, idx*chunk_size + data_s.shape[0])
        data_hash = hashtable.hash(data_s)
        raw_slice2 = hashtable.setdefault(data_hash, raw_slice)
        if raw_slice2 == raw_slice:
            slices_to_write[raw_slice] = s
        slices.append(raw_slice2)

    ds.resize((old_shape[0] + len(slices_to_write)*chunk_size,))
    for raw_slice, s in slices_to_write.items():
        ds[raw_slice.raw] = data[s.raw]
    return slices

def write_dataset_chunks(f, name, data_dict):
    """
    data_dict should be a dictionary mapping chunk_size index to either an
    array for that chunk, or a slice into the raw data for that chunk

    """
    if name not in f['_version_data']:
        raise NotImplementedError("Use write_dataset() if the dataset does not yet exist")

    ds = f['_version_data'][name]['raw_data']
    chunk_size = ds.attrs['chunk_size']
    # TODO: Handle more than one dimension
    nchunks = max(data_dict)
    if any(i not in data_dict for i in range(nchunks)):
        raise ValueError("data_dict does not include all chunks")

    hashtable = Hashtable(f, name)
    slices = [None for i in range(len(data_dict))]
    data_to_write = {}
    for chunk, data_s in data_dict.items():
        if not isinstance(data_s, (slice, tuple, Slice)) and data_s.dtype != ds.dtype:
            raise ValueError(f"dtypes do not match ({data_s.dtype} != {ds.dtype})")

        idx = hashtable.largest_index
        if isinstance(data_s, (slice, tuple, Slice)):
            slices[chunk] = ndindex(data_s)
        else:
            raw_slice = Slice(idx*chunk_size, idx*chunk_size + data_s.shape[0])
            data_hash = hashtable.hash(data_s)
            raw_slice2 = hashtable.setdefault(data_hash, raw_slice)
            if raw_slice2 == raw_slice:
                data_to_write[raw_slice] = data_s
            slices[chunk] = raw_slice2

    assert None not in slices
    old_shape = ds.shape
    ds.resize((old_shape[0] + len(data_to_write)*chunk_size,))
    for raw_slice, data_s in data_to_write.items():
        ds[raw_slice.raw] = data_s
    return slices

def create_virtual_dataset(f, version_name, name, slices, attrs=None, fillvalue=None):
    raw_data = f['_version_data'][name]['raw_data']
    chunk_size = raw_data.attrs['chunk_size']
    for s in slices[:-1]:
        if s.stop - s.start != chunk_size:
            raise NotImplementedError("Smaller than chunk size slice is only supported as the last slice.")
    shape = (chunk_size*(len(slices) - 1) + slices[-1].stop - slices[-1].start,)

    layout = VirtualLayout(shape, dtype=raw_data.dtype)
    vs = VirtualSource('.', name=raw_data.name, shape=raw_data.shape, dtype=raw_data.dtype)

    for i, s in enumerate(slices):
        # TODO: This needs to handle more than one dimension
        layout[i*chunk_size:i*chunk_size + s.stop - s.start] = vs[s.raw]

    virtual_data = f['_version_data/versions'][version_name].create_virtual_dataset(name, layout, fillvalue=fillvalue)

    if attrs:
        for k, v in attrs.items():
            virtual_data.attrs[k] = v
    return virtual_data
