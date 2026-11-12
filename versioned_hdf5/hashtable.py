import hashlib
import struct
import threading
from collections.abc import MutableMapping

import numpy as np
from h5py import File
from ndindex import ChunkSize, Slice, Tuple

from versioned_hdf5._lock import with_phil
from versioned_hdf5.slicetools import spaceid_to_slice


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

    hash_function = hashlib.sha256
    hash_size = hash_function().digest_size

    # Locking. Methods that read or write the on-disk dataset (__init__, write and the
    # alternate constructors) hold phil, like everything else that performs more than
    # one HDF5 call. The in-memory table (hash_table, _indices, _largest_index) is
    # instead guarded by the instance-local _lock, so that looking up or inserting a
    # hash does not stall unrelated h5py work in other threads.
    #
    # _lock is a leaf lock: nothing that can acquire phil (i.e. no h5py call) may run
    # while it is held. Acquiring phil first and then _lock is fine; the opposite
    # order would deadlock against a thread doing just that.

    @with_phil
    def __init__(self, f, name, *, chunk_size=None, hash_table_name="hash_table"):
        from .backend import DEFAULT_CHUNK_SIZE

        self.f = f
        self.name = name
        self.chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
        self.hash_table_name = hash_table_name

        if hash_table_name in f["_version_data"][name]:
            self.hash_table_dataset = f["_version_data"][name][hash_table_name]
            self.hash_table, self._indices = self._load_hashtable(
                self.hash_table_dataset
            )
        else:
            self.hash_table_dataset = self._create_hashtable()
            self.hash_table = self.hash_table_dataset[:]
            self._indices = {}

        # Read eagerly, so that largest_index never needs HDF5 (and thus phil) later
        self._largest_index = self.hash_table_dataset.attrs["largest_index"]
        self._lock = threading.Lock()

    @classmethod
    @with_phil
    def from_versions_traverse(
        cls,
        f: File,
        name: str,
    ):
        """Traverse all versions of a dataset, writing to a brand new hash table.

        Parameters
        ----------
        f : File
            File for which a hash table is to be generated
        name : str
            Name of the dataset for which a hash table is to be generated
        """
        name = name.removeprefix("/_version_data/")

        for version_group in f["_version_data"]["versions"].values():
            if name in version_group:
                dataset = version_group[name]
                with Hashtable(f, name) as hashtable:
                    if dataset.is_virtual:
                        for vs in dataset.virtual_sources():
                            sl = spaceid_to_slice(vs.src_space)
                            dl = spaceid_to_slice(vs.vspace)
                            assert len(dl.raw) == len(sl.raw)
                            data_slice = dataset[dl.raw]
                            slice_hash = hashtable.hash(data_slice)
                            if slice_hash not in hashtable:
                                hashtable[slice_hash] = sl.raw[0]

    @classmethod
    @with_phil
    def from_raw_data(cls, f, name, chunk_size=None, hash_table_name="hash_table"):
        if hash_table_name in f["_version_data"][name]:
            raise ValueError(
                f"a hash table {hash_table_name!r} for {name!r} already exists"
            )

        hashtable = cls(f, name, chunk_size=chunk_size, hash_table_name=hash_table_name)

        raw_data = f["_version_data"][name]["raw_data"]
        chunks = ChunkSize(raw_data.chunks)
        for c in chunks.indices(raw_data.shape):
            data_hash = hashtable.hash(raw_data[c.raw])
            hashtable.setdefault(data_hash, c.args[0])

        hashtable.write()
        return hashtable

    def hash(self, data: np.ndarray):
        """
        Compute hash for `data` array.

        This is a pure function of `data` and takes no lock.
        """
        # Object dtype arrays store the ids of the elements, which may or may not be
        # reused, making it unsuitable for hashing. Instead, we need to make a combined
        # hash with the value of each element.
        hash_value = self.hash_function()  # type: ignore

        if data.dtype.kind == "T":
            # Ensure that StringDType and object type strings produce the same hash.
            # TODO this can be accelerated in C/Cython
            #
            # DO NOT use hash_value.update(data)!
            # Besides producing a different hash, it also suffers from hash collisions
            # for long strings: https://github.com/numpy/numpy/issues/29226
            data = data.astype(object)

        if data.dtype == object:
            for value in data.flat:
                if isinstance(value, str):
                    # default to utf-8 encoding since it's a superset of ascii (the only
                    # other valid encoding supported in h5py)
                    value = value.encode("utf-8")
                # hash the length of value ('Q' is unsigned long long, which is 64 bit
                # on all of today's architectures)
                # Use little-endian byte order (e.g. x86, ARM) for consistency
                # everywhere, even on big-endian architectures (e.g. PowerPC).
                hash_value.update(struct.pack("<Q", len(value)))
                # Hash the buffer of bytes.
                hash_value.update(value)
        else:
            hash_value.update(np.ascontiguousarray(data))

        hash_value.update(str(data.shape).encode("ascii"))
        return hash_value.digest()

    @property
    def largest_index(self):
        return self._largest_index

    @largest_index.setter
    def largest_index(self, value):
        self._largest_index = value

    @with_phil
    def write(self):
        # Take a consistent snapshot of the in-memory table, then release _lock
        # before touching HDF5: see the note on locking above.
        with self._lock:
            largest_index = self._largest_index
            records = self.hash_table[:largest_index].copy()

        if largest_index >= self.hash_table_dataset.shape[0]:
            self.hash_table_dataset.resize((largest_index,))

        # largest_index is here for backwards compatibility for when the hash
        # table shape used to always be chunk_size aligned.
        self.hash_table_dataset.attrs["largest_index"] = largest_index
        self.hash_table_dataset[:largest_index] = records

    def inverse(self):
        r"""
        Return a dictionary mapping Slice: array_of_hash.

        The Slices are all `reduce()`\d.
        """
        with self._lock:
            records = self.hash_table.copy()
        return {Slice(*s).reduce(): h for h, s in records}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            return
        self.write()

    @with_phil
    def _create_hashtable(self):
        f = self.f
        name = self.name

        # TODO: The real chunk size should be based on bytes, not number of elements
        dtype = np.dtype([("hash", "B", (self.hash_size,)), ("shape", "i8", (2,))])
        hash_table = f["_version_data"][name].create_dataset(
            self.hash_table_name,
            shape=(1,),
            dtype=dtype,
            chunks=(self.chunk_size,),
            maxshape=(None,),
            compression="lzf",
        )
        hash_table.attrs["largest_index"] = 0
        return hash_table

    @with_phil
    def _load_hashtable(self, hash_table_dataset):
        largest_index = hash_table_dataset.attrs["largest_index"]
        hash_table_arr = hash_table_dataset[:largest_index]
        hashes = bytes(hash_table_arr["hash"])
        hashes = [
            hashes[i * self.hash_size : (i + 1) * self.hash_size]
            for i in range(largest_index)
        ]
        return hash_table_arr, {k: i for i, k in enumerate(hashes)}

    def __getitem__(self, key):
        if isinstance(key, np.ndarray):
            key = key.tobytes()
        with self._lock:
            i = self._indices[key]
            start, stop = self.hash_table["shape"][i]
        return Slice(start, stop)

    def __setitem__(self, key, value):
        if isinstance(key, np.ndarray):
            key = key.tobytes()
        if not isinstance(key, bytes):
            raise TypeError(f"key must be bytes, got {type(key)}")
        if len(key) != self.hash_size:
            raise ValueError(f"key must be {self.hash_size} bytes")
        if isinstance(value, Tuple):
            if len(value.args) > 1:
                raise NotImplementedError(
                    "Chunking in more other than the first dimension"
                )
            value = value.args[0]
        if isinstance(value, slice):
            value = Slice(value.start, value.stop, value.step)
        elif isinstance(value, Slice):
            pass
        else:
            raise TypeError("value must be a slice object")
        if value.isempty():
            return
        if value.step not in [1, None]:
            raise ValueError("only step-1 slices are supported")

        kv = (list(key), (value.start, value.stop))
        with self._lock:
            i = self._indices.get(key)
            if i is not None:
                if bytes(self.hash_table[i])[0] != key:
                    raise ValueError(
                        f"The key {key!r} is already in the hashtable under another "
                        "index."
                    )
                self.hash_table[i] = kv
            else:
                i = self._largest_index
                if i >= self.hash_table.shape[0]:
                    newshape = (self.hash_table.shape[0] + self.chunk_size,)
                    new_hash_table = np.zeros(newshape, dtype=self.hash_table.dtype)
                    new_hash_table[: self.hash_table.shape[0]] = self.hash_table
                    self.hash_table = new_hash_table
                self.hash_table[i] = kv
                self._indices[key] = i
                self._largest_index = i + 1

    def __delitem__(self, key):
        raise NotImplementedError

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)
