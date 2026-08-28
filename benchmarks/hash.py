"""Benchmarks for hash_slab (Cython SHA256) vs. naive hashlib."""

import hashlib
import struct
from contextlib import suppress

import numpy as np

with suppress(ImportError):  # Allow asv-compare vs. older releases
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
        self.chunk_size = chunk_size

    def time_hash_slab(self, chunk_size, edge):
        hash_slab(
            self.src,
            self.hash_table,
            self.hash_rows,
            self.src_start,
            self.count,
            self.chunk_size,
        )

    peakmem_hash_slab = time_hash_slab

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


class TimeHashSlabNonContig:
    """Benchmark hash_slab with non-C-contiguous slabs.

    **layout**

    step_outer
        strided along axis 0, contiguous along the innermost axis (hashed as one
        EVP_DigestUpdate per row, without any copy)
    transpose
        transposed slab (chunk-sized scratch buffer path)
    step_inner
        strided along the innermost axis (chunk-sized scratch buffer path)
    broadcast
        read-only broadcasted slab, e.g. the full slab of a StagedChangesArray
        (chunk-sized scratch buffer path)
    """

    params = [["step_outer", "transpose", "step_inner", "broadcast"], CHUNK_SIZES]
    param_names = ["layout", "chunk_size"]

    def setup(self, layout, chunk_size):
        rng = np.random.default_rng(42)
        n_chunks = TOTAL_BYTES // np.prod(chunk_size)
        n_rows = n_chunks * chunk_size[0]
        cs1 = chunk_size[1]
        if layout == "step_outer":
            self.src = rng.random((2 * n_rows, cs1), dtype=np.float64)[::2]
        elif layout == "transpose":
            self.src = rng.random((cs1, n_rows), dtype=np.float64).T
        elif layout == "step_inner":
            self.src = rng.random((n_rows, 2 * cs1), dtype=np.float64)[:, ::2]
        elif layout == "broadcast":
            self.src = np.broadcast_to(1.5, (n_rows, cs1))
        else:
            raise AssertionError("unreachable")  # pragma: nocover
        assert self.src.shape == (n_rows, cs1)

        self.hash_table = np.zeros((n_chunks, 4), dtype=np.uint64)
        self.hash_rows = np.arange(n_chunks, dtype=np.uint64)
        self.src_start = np.arange(0, n_rows, chunk_size[0], dtype=np.uint64)
        self.count = np.empty((n_chunks, 2), dtype=np.uint64)
        self.count[:, 0] = chunk_size[0]
        self.count[:, 1] = cs1
        self.chunk_size = chunk_size

    def time_hash_slab(self, layout, chunk_size):
        hash_slab(
            self.src,
            self.hash_table,
            self.hash_rows,
            self.src_start,
            self.count,
            self.chunk_size,
        )

    peakmem_hash_slab = time_hash_slab


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
        self.chunk_size = chunk_size

    def time_hash_slab(self, dtype, max_nchars, chunk_size):
        hash_slab(
            self.src,
            self.hash_table,
            self.hash_rows,
            self.src_start,
            self.count,
            self.chunk_size,
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
