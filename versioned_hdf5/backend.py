from h5py import VirtualLayout, VirtualSource

import math

from .hashtable import hashtable

CHUNK_SIZE = 2**20

def get_chunks(shape):
    # TODO: Implement this
    if len(shape) > 1:
        raise NotImplementedError
    return (CHUNK_SIZE,)

def split_chunks(shape):
    if len(shape) > 1:
        raise NotImplementedError

    for i in range(math.ceil(shape[0]/CHUNK_SIZE)):
        yield slice(CHUNK_SIZE*i, CHUNK_SIZE*(i + 1))

def initialize(f):
    f.create_group('_version_data')

def create_base_dataset(f, name, *, shape=None, data=None):
    group = f['/_version_data'].create_group(name)
    group.create_dataset('raw_data', shape=(0,),
                              chunks=(CHUNK_SIZE,), maxshape=(None,))

    return write_dataset(f, name, data)

# Helper functions to workaround slices not being hashable
def s2t(s):
    return (s.start, s.stop)

def t2s(t):
    return slice(*t)

def write_dataset(f, name, data):
    if name not in f['/_version_data']:
        return create_base_dataset(f, name, data=data)

    ds = f['/_version_data'][name]['raw_data']
    # TODO: Handle more than one dimension
    old_shape = ds.shape
    h = hashtable(f, name)
    slices = []
    slices_to_write = {}
    for s in split_chunks(data.shape):
        idx = h.largest_index
        data_s = data[s]
        raw_slice = slice(idx*CHUNK_SIZE, idx*CHUNK_SIZE + data_s.shape[0])
        data_hash = h.hash(data_s)
        raw_slice2 = h.setdefault(data_hash, raw_slice)
        if raw_slice2 == raw_slice:
            slices_to_write[s2t(raw_slice)] = s
        slices.append(raw_slice2)

    ds.resize((old_shape[0] + len(slices_to_write)*CHUNK_SIZE,))
    for raw_slice, s in slices_to_write.items():
        ds[t2s(raw_slice)] = data[s]
    return slices

def create_virtual_dataset(f, version_name, name, slices):
    for s in slices[:-1]:
        if s.stop - s.start != CHUNK_SIZE:
            raise NotImplementedError("Smaller than chunk size slice is only supported as the last slice.")
    shape = (CHUNK_SIZE*(len(slices) - 1) + slices[-1].stop - slices[-1].start,)

    layout = VirtualLayout(shape)
    vs = VirtualSource(f['_version_data'][name]['raw_data'])

    for i, s in enumerate(slices):
        # TODO: This needs to handle more than one dimension
        layout[i*CHUNK_SIZE:i*CHUNK_SIZE + s.stop - s.start] = vs[s]

    virtual_data = f['_version_data'][version_name].create_virtual_dataset(name, layout)
    return virtual_data
