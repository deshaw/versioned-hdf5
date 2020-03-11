from __future__ import print_function, division

from h5py import VirtualLayout, VirtualSource

import numpy as np

import math

from .hashtable import Hashtable

CHUNK_SIZE = 2**12

def get_chunks(shape, dtype):
    # TODO: Implement this
    if len(shape) > 1:
        raise NotImplementedError
    return (CHUNK_SIZE,)

def split_chunks(shape):
    if len(shape) > 1:
        raise NotImplementedError

    for i in range(int(math.ceil(shape[0]/CHUNK_SIZE))):
        yield slice(CHUNK_SIZE*i, CHUNK_SIZE*(i + 1))

def initialize(f):
    version_data = f.create_group('_version_data')
    versions = version_data.create_group('versions')
    versions.create_group('__first_version__')
    versions.attrs['current_version'] = '__first_version__'

def create_base_dataset(f, name, shape=None, data=None, dtype=np.float64):
    group = f['/_version_data'].create_group(name)
    group.create_dataset('raw_data', shape=(0,), chunks=(CHUNK_SIZE,),
                         maxshape=(None,), dtype=dtype)
    return write_dataset(f, name, data)

def write_dataset(f, name, data):
    from .slicetools import s2t, t2s

    if name not in f['/_version_data']:
        return create_base_dataset(f, name, data=data)

    ds = f['/_version_data'][name]['raw_data']
    if data.dtype != ds.dtype:
        raise ValueError("dtypes do not match ({data.dtype} != {ds.dtype})".format(data=data, ds=ds))
    # TODO: Handle more than one dimension
    old_shape = ds.shape
    hashtable = Hashtable(f, name)
    slices = []
    slices_to_write = {}
    for s in split_chunks(data.shape):
        idx = hashtable.largest_index
        data_s = data[s]
        raw_slice = slice(idx*CHUNK_SIZE, idx*CHUNK_SIZE + data_s.shape[0])
        data_hash = hashtable.hash(data_s)
        raw_slice2 = hashtable.setdefault(data_hash, raw_slice)
        if raw_slice2 == raw_slice:
            slices_to_write[s2t(raw_slice)] = s
        slices.append(raw_slice2)

    ds.resize((old_shape[0] + len(slices_to_write)*CHUNK_SIZE,))
    for raw_slice, s in slices_to_write.items():
        ds[t2s(raw_slice)] = data[s]
    return slices

def write_dataset_chunks(f, name, data_dict):
    """
    data_dict should be a dictionary mapping chunk_size index to either an
    array for that chunk, or a slice into the raw data for that chunk

    """
    from .slicetools import s2t, t2s

    if name not in f['/_version_data']:
        raise NotImplementedError("Use write_dataset() if the dataset does not yet exist")

    ds = f['/_version_data'][name]['raw_data']
    # TODO: Handle more than one dimension
    nchunks = max(data_dict)
    if any(i not in data_dict for i in range(nchunks)):
        raise ValueError("data_dict does not include all chunks")

    hashtable = Hashtable(f, name)
    slices = [None for i in range(len(data_dict))]
    data_to_write = {}
    for chunk, data_s in data_dict.items():
        if not isinstance(data_s, (slice, tuple)) and data_s.dtype != ds.dtype:
            raise ValueError("dtypes do not match ({data_s.dtype} != {ds.dtype})".format(data_s=data_s, ds=ds))

        idx = hashtable.largest_index
        if isinstance(data_s, (slice, tuple)):
            slices[chunk] = data_s
        else:
            raw_slice = slice(idx*CHUNK_SIZE, idx*CHUNK_SIZE + data_s.shape[0])
            data_hash = hashtable.hash(data_s)
            raw_slice2 = hashtable.setdefault(data_hash, raw_slice)
            if raw_slice2 == raw_slice:
                data_to_write[s2t(raw_slice)] = data_s
            slices[chunk] = raw_slice2

    assert None not in slices
    old_shape = ds.shape
    ds.resize((old_shape[0] + len(data_to_write)*CHUNK_SIZE,))
    for raw_slice, data_s in data_to_write.items():
        ds[t2s(raw_slice)] = data_s
    return slices

def create_virtual_dataset(f, version_name, name, slices):
    for s in slices[:-1]:
        if s.stop - s.start != CHUNK_SIZE:
            raise NotImplementedError("Smaller than chunk size slice is only supported as the last slice.")
    shape = (CHUNK_SIZE*(len(slices) - 1) + slices[-1].stop - slices[-1].start,)

    raw_data = f['_version_data'][name]['raw_data']
    layout = VirtualLayout(shape, dtype=raw_data.dtype)
    vs = VirtualSource(raw_data)

    for i, s in enumerate(slices):
        # TODO: This needs to handle more than one dimension
        layout[i*CHUNK_SIZE:i*CHUNK_SIZE + s.stop - s.start] = vs[s]

    virtual_data = f['_version_data/versions'][version_name].create_virtual_dataset(name, layout)
    return virtual_data
