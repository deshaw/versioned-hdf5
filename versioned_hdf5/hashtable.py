import numpy as np
from ndindex import Slice, Tuple, ChunkSize

import hashlib
from collections.abc import MutableMapping
from functools import lru_cache

class Hashtable(MutableMapping):
    """
    A proxy class representing the hash table for an array

    The hash table for an array is a mapping from {sha256_hash: slice}, where
    slice is a slice for the data in the array.

    General usage should look like

        with Hashtable(f, name) as h:
            data_hash = h.hash(data[raw_slice])
            raw_slice = h.setdefault(data_hash, raw_slice)

    where setdefault will insert the hash into the table if it
    doesn't exist, and return the existing entry otherwise.

    hashtable.largest_index is the next index in the array that slices
    should be mapped to.

    Note that for performance reasons, the hashtable does not write to the
    dataset until you call write() or it exit as a context manager.

    """
    # Cache instances of the class for performance purposes. This works off
    # the assumption that nothing else modifies the version data.

    # This is done here because putting @lru_cache() on the class breaks the
    # classmethods. Warning: This does not normalize kwargs, so it is possible
    # to have multiple hashtable instances for the same hashtable.
    @lru_cache()
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        return obj

    def __init__(self, f, name, *, chunk_size=None, hash_table_name='hash_table'):
        from .backend import DEFAULT_CHUNK_SIZE

        self.f = f
        self.name = name
        self.chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
        self.hash_table_name = hash_table_name

        if hash_table_name in f['_version_data'][name]:
            self._load_hashtable()
        else:
            self._create_hashtable()
        self._largest_index = None
        self.hash_table = f['_version_data'][name][hash_table_name][:]
        self.hash_table_dataset = f['_version_data'][name][hash_table_name]

    @classmethod
    def from_raw_data(cls, f, name, chunk_size=None, hash_table_name='hash_table'):
        if hash_table_name in f['_version_data'][name]:
            raise ValueError(f"a hash table {hash_table_name!r} for {name!r} already exists")

        hashtable = cls(f, name, chunk_size=chunk_size, hash_table_name=hash_table_name)

        raw_data = f['_version_data'][name]['raw_data']
        chunks = ChunkSize(raw_data.chunks)
        for c in chunks.indices(raw_data.shape):
            data_hash = hashtable.hash(raw_data[c.raw])
            hashtable.setdefault(data_hash, c.args[0])

        hashtable.write()
        return hashtable

    hash_function = hashlib.sha256
    hash_size = hash_function().digest_size

    def hash(self, data):
        return self.hash_function(data.data.tobytes() + bytes(str(data.shape), 'ascii')).digest()

    @property
    def largest_index(self):
        if self._largest_index is None:
            self._largest_index = self.hash_table_dataset.attrs['largest_index']
        return self._largest_index

    @largest_index.setter
    def largest_index(self, value):
        self._largest_index = value

    def write(self):
        largest_index = self.largest_index
        if largest_index >= self.hash_table_dataset.shape[0]:
            self.hash_table_dataset.resize((largest_index,))

        # largest_index is here for backwards compatibility for when the hash
        # table shape used to always be chunk_size aligned.
        self.hash_table_dataset.attrs['largest_index'] = self.largest_index
        self.hash_table_dataset[:largest_index] = self.hash_table[:largest_index]

    def inverse(self):
        r"""
        Return a dictionary mapping Slice: array_of_hash.

        The Slices are all `reduce()`\d.
        """
        return {Slice(*s).reduce(): h for h, s in self.hash_table}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            return
        self.write()

    def _create_hashtable(self):
        f = self.f
        name = self.name

        # TODO: Use get_chunks() here (the real chunk size should be based on
        # bytes, not number of elements)
        dtype = np.dtype([('hash', 'B', (self.hash_size,)), ('shape', 'i8', (2,))])
        hash_table = f['_version_data'][name].create_dataset(self.hash_table_name,
                                                             shape=(1,), dtype=dtype,
                                                             chunks=(self.chunk_size,),
                                                             maxshape=(None,),
                                                             compression='lzf')
        hash_table.attrs['largest_index'] = 0
        self._indices = {}

    def _load_hashtable(self):
        hash_table = self.f['_version_data'][self.name][self.hash_table_name]
        largest_index = hash_table.attrs['largest_index']
        hash_table_arr = hash_table[:largest_index]
        hashes = bytes(hash_table_arr['hash'])
        hashes = [hashes[i*self.hash_size:(i+1)*self.hash_size] for i in range(largest_index)]
        self._indices = {k: i for i, k in enumerate(hashes)}

    def __getitem__(self, key):
        if isinstance(key, np.ndarray):
            key = key.tobytes()
        i = self._indices[key]
        shapes = self.hash_table['shape']
        return Slice(*shapes[i])

    def __setitem__(self, key, value):
        if isinstance(key, np.ndarray):
            key = key.tobytes()
        if not isinstance(key, bytes):
            raise TypeError(f"key must be bytes, got {type(key)}")
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
        if key in self._indices:
            if bytes(self.hash_table[self._indices[key]])[0] != key:
                raise ValueError("The key %s is already in the hashtable under another index.")
            self.hash_table[self._indices[key]] = kv
        else:
            if self.largest_index >= self.hash_table.shape[0]:
                newshape = (self.hash_table.shape[0] + self.chunk_size,)
                new_hash_table = np.zeros(newshape, dtype=self.hash_table.dtype)
                new_hash_table[:self.hash_table.shape[0]] = self.hash_table
                self.hash_table = new_hash_table
            self.hash_table[self.largest_index] = kv
            self._indices[key] = self.largest_index
            self.largest_index += 1

    def __delitem__(self, key):
        raise NotImplementedError

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)
