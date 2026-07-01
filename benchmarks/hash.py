"""Benchmarks for hash_slab (Cython SHA256) vs. naive hashlib."""

import hashlib
import struct

import numpy as np
from versioned_hdf5.hash import hash_slab

from .common import require_npystrings

# 1 KiB, 64 KiB, 1 MiB of float64 per chunk
CHUNK_SIZES = [(8, 16), (64, 128), (1024, 1024)]
# 1 KiB, 64 KiB
STRING_CHUNK_SIZES = [(8, 16), (64, 128)]
# Total bytes transferred per benchmark call ~16 MiB
TOTAL_BYTES = 16 * 1024 * 1024 // 8


class TimeHashSlab:
    """Benchmark hash_slab with a contiguous NumPy slab."""

    params = [CHUNK_SIZES, [False, True]]
    param_names = ["chunk_size", "edge"]

    def setup(self, chunk_size, edge):
        rng = np.random.default_rng(42)
        n_chunks = TOTAL_BYTES // np.prod(chunk_size)
        self.src = rng.random(
            (n_chunks * chunk_size[0], chunk_size[1]), dtype=np.float64
        )
        self.hash_table = np.zeros((n_chunks, 4), dtype=np.uint64)
        self.hash_rows = np.arange(n_chunks, dtype=np.uint64)
        self.src_start = np.arange(0, self.src.shape[0], chunk_size[0], dtype=np.uint64)
        self.count = np.empty((n_chunks, 2), dtype=np.uint64)
        self.count[:, 0] = chunk_size[0]
        self.count[:, 1] = chunk_size[1] - 1 if edge else chunk_size[1]

    def time_hash_slab(self, chunk_size, edge):
        hash_slab(
            self.src,
            self.hash_table,
            self.hash_rows,
            self.src_start,
            self.count,
        )

    def time_hash_slab_naive(self, chunk_size, edge):
        """Naive Python hashlib reimplementation of hash_slab."""
        for hash_row, src_start, count in zip(
            self.hash_rows, self.src_start, self.count, strict=True
        ):
            idx = tuple(
                slice(src_start, src_start + c) if i == 0 else slice(c)
                for i, c in enumerate(count)
            )
            chunk = self.src[idx]
            h = hashlib.sha256()
            h.update(np.ascontiguousarray(chunk))
            h.update(str(chunk.shape).encode("ascii"))
            self.hash_table[hash_row, :] = np.frombuffer(h.digest(), dtype=np.uint64)


class TimeHashSlabStrings:
    """Benchmark hash_slab with string arrays (object dtype and StringDType)"""

    params = [
        ["O", "T"],
        [8, 64, 256],
        STRING_CHUNK_SIZES,
    ]
    param_names = ["dtype", "max_nchars", "chunk_size"]

    TOTAL_ELEMENTS = 32768

    def setup(self, dtype, max_nchars, chunk_size):
        if dtype == "T":
            require_npystrings()

        rng = np.random.default_rng(42)
        n_chunks = self.TOTAL_ELEMENTS // (chunk_size[0] * chunk_size[1])

        # Generate random fixed-length strings matching common.Benchmark.rand_strings
        rand_chars = (
            rng.integers(
                ord("0"), ord("z"), (self.TOTAL_ELEMENTS, max_nchars), dtype=np.uint8
            )
            .view("S1")
            .astype("U1")
            .tolist()
        )
        strings = ["".join(row) for row in rand_chars]
        np_dtype = object if dtype == "O" else "T"
        self.src = np.asarray(strings, dtype=np_dtype).reshape(
            n_chunks * chunk_size[0], chunk_size[1]
        )

        self.hash_table = np.zeros((n_chunks, 4), dtype=np.uint64)
        self.hash_rows = np.arange(n_chunks, dtype=np.uint64)
        self.src_start = np.arange(0, self.src.shape[0], chunk_size[0], dtype=np.uint64)
        self.count = np.empty((n_chunks, 2), dtype=np.uint64)
        self.count[:, 0] = chunk_size[0]
        self.count[:, 1] = chunk_size[1]

    def time_hash_slab(self, dtype, max_nchars, chunk_size):
        hash_slab(
            self.src,
            self.hash_table,
            self.hash_rows,
            self.src_start,
            self.count,
        )

    def time_hash_slab_naive(self, dtype, max_nchars, chunk_size):
        """Naive Python hashlib reimplementation of hash_slab for strings."""
        for hash_row, src_start, count in zip(
            self.hash_rows, self.src_start, self.count, strict=True
        ):
            idx = tuple(
                slice(src_start, src_start + c) if i == 0 else slice(c)
                for i, c in enumerate(count)
            )
            chunk = self.src[idx]
            h = hashlib.sha256()
            if chunk.dtype.kind == "T":
                chunk = chunk.astype(object)
            for value in chunk.flat:
                if isinstance(value, str):
                    value = value.encode("utf-8")
                assert isinstance(value, bytes)
                h.update(struct.pack("<Q", len(value)))
                h.update(value)
            h.update(str(chunk.shape).encode("ascii"))
            self.hash_table[hash_row, :] = np.frombuffer(h.digest(), dtype=np.uint64)
