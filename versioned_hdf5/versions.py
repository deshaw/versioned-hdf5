from h5py import VirtualLayout, VirtualSource
import numpy as np

import math
import hashlib
from collections.abc import MutableMapping


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
    # h = hashtable(f, name)
    ds = group.create_dataset('raw_data', shape=shape, data=data,
                                                chunks=get_chunks(shape),
                                                maxshape=(None,)*len(shape))

    ds.resize((math.ceil(shape[0]/CHUNK_SIZE)*CHUNK_SIZE,))

    slices = []
    for i, s in enumerate(split_chunks(data.shape)):
        raw_slice = slice(i*CHUNK_SIZE, i*CHUNK_SIZE + s.stop - s.start)
        slices.append(raw_slice)
    return slices

class hashtable(MutableMapping):
    def __init__(self, f, name):
        self.f = f
        self.name = name
        if 'hash_table' in f['_version_data'][name]:
            self._load_hashtable()
        else:
            self._create_hashtable()

        self.hash_table = f['/_version_data'][name]['hash_table']

    hash_size = 32 # hash_size = hashlib.sha256().digest_size

    def hash(self, data):
        return hashlib.sha256(bytes(data)).digest()

    @property
    def largest_index(self):
        return self.hash_table.attrs['largest_index']

    @largest_index.setter
    def largest_index(self, value):
        self.hash_table.attrs['largest_index'] = value

    def _create_hashtable(self):
        f = self.f
        name = self.name

        # TODO: Use get_chunks() here (the real chunk size should be based on
        # bytes, not number of elements)
        dtype = np.dtype([('hash', 'B', (self.hash_size,)), ('shape', 'u8', (2,))])
        hash_table = f['/_version_data'][name].create_dataset('hash_table',
                                                 shape=(CHUNK_SIZE,), dtype=dtype,
                                                 chunks=(CHUNK_SIZE,),
                                                 maxshape=(None,))
        hash_table.attrs['largest_index'] = 0
        self.hash_table = hash_table
        self._d = {}
        self._indices = {}

    def _load_hashtable(self):
        hash_table = self.hash_table
        largest_index = self.largest_index

        self._d = {bytes(hash_table[i][0]): tuple(hash_table[i][1]) for i in range(largest_index)}
        self._indices = {bytes(hash_table[i][0]): i for i in range(largest_index)}

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value):
        if not isinstance(key, bytes):
            raise TypeError("key must be bytes")
        if len(key) != self.hash_size:
            raise ValueError("key must be %d bytes" % self.hash_size)

        kv = (list(key), value)
        if key in self._d:
            assert bytes(self.hash_table[self._indices[key]])[0] == key
            self.hash_table[self._indices[key]] = kv
        else:
            self.hash_table[self.largest_index] = kv
            self._indices[key] = self.largest_index
            self.largest_index += 1
        self._d[key] = value

    def __delitem__(self, key):
        raise NotImplementedError

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

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
