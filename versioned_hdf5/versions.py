from h5py import VirtualLayout, VirtualSource

import math
import hashlib

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
    if data is not None and shape is not None:
        raise ValueError("Only one of data or shape should be passed")
    if shape is None:
        shape = data.shape
    group = f['/_version_data'].create_group(name)
    create_hashtable(f, name)
    ds = group.create_dataset('raw_data', shape=shape, data=data,
                                                chunks=get_chunks(shape),
                                                maxshape=(None,)*len(shape))

    ds.resize((math.ceil(shape[0]/CHUNK_SIZE)*CHUNK_SIZE,))

    slices = []
    for i, s in enumerate(split_chunks(data.shape)):
        raw_slice = slice(i*CHUNK_SIZE, i*CHUNK_SIZE + s.stop - s.start)
        slices.append(raw_slice)
    return slices

def hash(data):
    return hashlib.sha256(bytes(data)).digest()

# TODO: Wrap this in a dict-like class
def create_hashtable(f, name):
    hash_size = 32 # hash_size = hashlib.sha256().digest_size

    # TODO: Use get_chunks() here (the real chunk size should be based on
    # bytes, not number of elements)
    keys_chunks = (CHUNK_SIZE//hash_size, hash_size)
    keys = f['/_version_data'][name].create_dataset('hash_table_keys',
                                             shape=keys_chunks, dtype='B',
                                             chunks=keys_chunks,
                                             maxshape=(None, hash_size))
    keys.attrs['largest_index'] = 0
    values_chunks = (CHUNK_SIZE//2, 2)
    values = f['/_version_data'][name].create_dataset('hash_table_values',
                                             shape=values_chunks, dtype='u8',
                                             chunks=values_chunks,
                                             maxshape=(None, 2))
    return keys, values

def load_hashtable(f, name):
    keys = f['/_version_data'][name]['hash_table_keys']
    largest_index = keys.attrs['largest_index']
    values = f['/_version_data'][name]['hash_table_values']
    return {bytes(keys[i]): tuple(values(i)) for i in range(largest_index)}

def write_dataset(f, name, data):
    if name not in f['/_version_data']:
        return create_base_dataset(f, name, data=data)

    ds = f['/_version_data'][name]['raw_data']
    # TODO: Handle more than one dimension
    old_shape = ds.shape
    idx = ds.shape[0]//CHUNK_SIZE
    slices = []
    ds.resize((old_shape[0] + math.ceil(data.shape[0]/CHUNK_SIZE)*CHUNK_SIZE,))
    for i, s in enumerate(split_chunks(data.shape), idx):
        data_s = data[s]
        raw_slice = slice(i*CHUNK_SIZE, i*CHUNK_SIZE + data_s.shape[0])
        ds[raw_slice] = data_s
        slices.append(raw_slice)
    return slices

def create_virtual_dataset(f, name, slices):
    for s in slices[:-1]:
        if s.stop - s.start != CHUNK_SIZE:
            raise NotImplementedError("Smaller than chunk size slice is only supported as the last slice.")
    shape = (CHUNK_SIZE*(len(slices) - 1) + slices[-1].stop - slices[-1].start,)

    layout = VirtualLayout(shape)
    vs = VirtualSource(f['_version_data'][name]['raw_data'])

    for i, s in enumerate(slices):
        # TODO: This needs to handle more than one dimension
        layout[i*CHUNK_SIZE:i*CHUNK_SIZE + s.stop - s.start] = vs[s]

    virtual_data = f.create_virtual_dataset(name, layout)
    return virtual_data
