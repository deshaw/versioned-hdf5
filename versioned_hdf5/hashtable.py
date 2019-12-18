import numpy as np

import hashlib
from collections.abc import MutableMapping

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
        from .versions import CHUNK_SIZE

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
        hash_table = self.f['/_version_data'][self.name]['hash_table']
        largest_index = hash_table.attrs['largest_index']

        self._d = {bytes(hash_table[i][0]): slice(*hash_table[i][1]) for i in range(largest_index)}
        self._indices = {bytes(hash_table[i][0]): i for i in range(largest_index)}

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value):
        if not isinstance(key, bytes):
            raise TypeError("key must be bytes")
        if len(key) != self.hash_size:
            raise ValueError("key must be %d bytes" % self.hash_size)
        if not isinstance(value, slice):
            raise TypeError("value must be a slice object")
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
        self._d[key] = value

    def __delitem__(self, key):
        raise NotImplementedError

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)
