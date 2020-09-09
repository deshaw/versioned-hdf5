import numpy as np
from ndindex import Slice, Tuple

import hashlib
from collections.abc import MutableMapping
from functools import lru_cache

@lru_cache()
class Hashtable(MutableMapping):
    """
    A proxy class representing the hash table for an array

    The hash table for an array is a mapping from {sha256_hash: slice}, where
    slice is a slice for the data in the array.

    General usage should look like

        h = Hashtable(f, name)
        data_hash = h.hash(data[raw_slice])
        raw_slice = h.setdefault(data_hash, raw_slice)

    where setdefault will insert the hash into the table if it
    doesn't exist, and return the existing entry otherwise.

    hashtable.largest_index is the largest index in the array that has slices
    mapped to it.

    """
    def __init__(self, f, name, chunk_size=None):
        from .backend import DEFAULT_CHUNK_SIZE

        self.f = f
        self.name = name
        self.chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
        if 'hash_table' in f['_version_data'][name]:
            self._load_hashtable()
        else:
            self._create_hashtable()
        self._largest_index = None

        self.hash_table = f['_version_data'][name]['hash_table']

    hash_function = hashlib.sha256
    hash_size = hash_function().digest_size

    def hash(self, data):
        return self.hash_function(data.data.tobytes() + bytes(str(data.shape), 'ascii')).digest()

    @property
    def largest_index(self):
        if self._largest_index is None:
            self._largest_index = self.hash_table.attrs['largest_index']
        return self._largest_index

    @largest_index.setter
    def largest_index(self, value):
        self._largest_index = value
        self.hash_table.attrs['largest_index'] = value

    def _create_hashtable(self):
        f = self.f
        name = self.name

        # TODO: Use get_chunks() here (the real chunk size should be based on
        # bytes, not number of elements)
        dtype = np.dtype([('hash', 'B', (self.hash_size,)), ('shape', 'i8', (2,))])
        hash_table = f['_version_data'][name].create_dataset('hash_table',
                                                 shape=(self.chunk_size,), dtype=dtype,
                                                 chunks=(self.chunk_size,),
                                                 maxshape=(None,))
        hash_table.attrs['largest_index'] = 0
        self.hash_table = hash_table
        self._d = {}
        self._indices = {}

    def _load_hashtable(self):
        hash_table = self.f['_version_data'][self.name]['hash_table']
        largest_index = hash_table.attrs['largest_index']
        hash_table_arr = hash_table[:largest_index]
        hashes = bytes(hash_table_arr['hash'])
        shapes = hash_table_arr['shape']
        self._d = {hashes[i*self.hash_size:(i+1)*self.hash_size]: Slice(*shapes[i]) for i in range(largest_index)}
        self._indices = {k: i for i, k in enumerate(self._d)}

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value):
        if not isinstance(key, bytes):
            raise TypeError("key must be bytes")
        if len(key) != self.hash_size:
            raise ValueError("key must be %d bytes" % self.hash_size)
        if isinstance(value, Tuple):
            if len(value.args) > 1:
                raise NotImplementedError("Chunking in more other than the first dimension")
            value = value.args[0]
        if not isinstance(value, (slice, Slice)):
            raise TypeError("value must be a slice object")
        value = Slice(value)
        if value.isempty():
            return
        if value.step not in [1, None]:
            raise ValueError("only step-1 slices are supported")

        kv = (list(key), (value.start, value.stop))
        if key in self._d:
            if bytes(self.hash_table[self._indices[key]])[0] != key:
                raise ValueError("The key %s is already in the hashtable under another index.")
            self.hash_table[self._indices[key]] = kv
        else:
            self.hash_table[self.largest_index] = kv
            self._indices[key] = self.largest_index
            self.largest_index += 1
            if self.largest_index >= self.hash_table.shape[0]:
                self.hash_table.resize((self.hash_table.shape[0] + self.chunk_size,))
        self._d[key] = Slice(value)

    def __delitem__(self, key):
        raise NotImplementedError

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)
