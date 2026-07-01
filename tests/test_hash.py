"""Test versioned_hdf5.hash.hash_slab.

See test_hash_legacy_compat.py for cross-checks against the legacy hash algorithm.
"""

import hashlib
import struct

import numpy as np
import pytest
from versioned_hdf5.hash import hash_slab

from versioned_hdf5.cytools import np_hsize_t
from versioned_hdf5.h5py_compat import HAS_NPYSTRINGS


def reference(chunk: np.ndarray) -> bytes:
    """Reference (slow) implementation of a single-chunk hash"""
    h = hashlib.sha256()
    if chunk.dtype.kind == "T":
        chunk = chunk.astype(object)
    if chunk.dtype == object:
        for value in chunk.flat:
            if isinstance(value, str):
                value = value.encode("utf-8")
            assert isinstance(value, bytes)
            h.update(struct.pack("<Q", len(value)))
            h.update(value)
    else:
        h.update(np.ascontiguousarray(chunk))
    h.update(str(chunk.shape).encode("ascii"))
    return h.digest()


def rows_as_digests(hash_table: np.ndarray) -> list[bytes]:
    return [row.view(np.uint8).tobytes() for row in hash_table]


def test_single_chunk():
    slab = np.arange(10, dtype="i8")
    ht = np.zeros((1, 4), dtype=np.uint64)
    hash_slab(
        slab,
        ht,
        np.array([0], dtype=np_hsize_t),
        np.array([0], dtype=np_hsize_t),
        np.array([[10]], dtype=np_hsize_t),
    )
    assert rows_as_digests(ht)[0] == reference(slab)


def test_multiple_chunks_route_to_rows():
    """Each chunk's digest lands in the row named by hash_rows;
    unlisted rows stay untouched.
    """
    slab = np.arange(16, dtype="i4")  # 4 chunks
    ht = np.zeros((4, 4), dtype=np.uint64)
    # Hash chunks 0 and 3 only, and (to prove hash_rows is honoured)
    # deliberately route chunk at offset 8 to row 2.
    hash_slab(
        slab,
        ht,
        np.array([0, 3, 2], dtype=np_hsize_t),
        np.array([0, 12, 8], dtype=np_hsize_t),
        np.array([[4], [4], [4]], dtype=np_hsize_t),
    )
    digests = rows_as_digests(ht)
    assert digests[0] == reference(slab[0:4])
    assert digests[1] == b"\x00" * 32  # never written
    assert digests[2] == reference(slab[8:12])
    assert digests[3] == reference(slab[12:16])


def test_edge_chunk_ignores_uninitialised_memory():
    """A chunk shorter than the physical slab width must hash only its valid region,
    never the (possibly garbage) memory past the edge.
    """
    slab = np.zeros((6, 5), dtype="i8")
    slab[:5] = np.arange(25).reshape(5, 5)
    # chunk 1 occupies physical rows [3:6] but is only valid for 2 rows ([3:5]);
    # row 5 is "uninitialised".
    src_start = np.array([0, 3], dtype=np_hsize_t)
    count = np.array([[3, 5], [2, 5]], dtype=np_hsize_t)
    rows = np.array([0, 1], dtype=np_hsize_t)

    ht = np.zeros((2, 4), dtype=np.uint64)
    hash_slab(slab, ht, rows, src_start, count)
    h0, h1 = rows_as_digests(ht)
    assert h0 == reference(slab[0:3, :])
    assert h1 == reference(slab[3:5, :])

    # Poison the uninitialised row and re-hash: the digests must be unchanged.
    slab[5] = 999
    ht2 = np.zeros((2, 4), dtype=np.uint64)
    hash_slab(slab, ht2, rows, src_start, count)
    assert rows_as_digests(ht2) == [h0, h1]


def test_column_edge_is_made_contiguous():
    """A chunk that is non-contiguous in the slab (trimmed on a non-leading axis) is
    hashed as if C-contiguous.
    """
    slab = np.arange(20, dtype="i4").reshape(4, 5)
    # Trim columns: slab[0:4, 0:3] is a non-contiguous view.
    ht = np.zeros((1, 4), dtype=np.uint64)
    hash_slab(
        slab,
        ht,
        np.array([0], dtype=np_hsize_t),
        np.array([0], dtype=np_hsize_t),
        np.array([[4, 3]], dtype=np_hsize_t),
    )
    sub = slab[0:4, 0:3]
    assert not sub.flags.c_contiguous
    assert rows_as_digests(ht)[0] == reference(sub)


def test_full_slab_broadcast():
    """hash_slab works on a read-only broadcasted array (the StagedChangesArray full
    slab) and matches the contiguous materialization.
    """
    fill = np.broadcast_to(np.array(42, dtype="u2"), (3, 4))
    assert not fill.flags.writeable
    ht = np.zeros((1, 4), dtype=np.uint64)
    hash_slab(
        fill,
        ht,
        np.array([0], dtype=np_hsize_t),
        np.array([0], dtype=np_hsize_t),
        np.array([[3, 4]], dtype=np_hsize_t),
    )
    assert rows_as_digests(ht)[0] == reference(np.full((3, 4), 42, dtype="u2"))


def test_empty_chunk():
    """A chunk with a zero-length axis hashes just its (empty) shape string."""
    slab = np.zeros((0, 3), dtype="f8")
    ht = np.zeros((1, 4), dtype=np.uint64)
    hash_slab(
        slab,
        ht,
        np.array([0], dtype=np_hsize_t),
        np.array([0], dtype=np_hsize_t),
        np.array([[0, 3]], dtype=np_hsize_t),
    )
    (h,) = rows_as_digests(ht)
    assert h == hashlib.sha256(b"(0, 3)").digest()
    assert h == reference(slab[0:0, 0:3])


def test_no_chunks_is_noop():
    slab = np.arange(10, dtype="i8")
    ht = np.zeros((1, 4), dtype=np.uint64)
    hash_slab(
        slab,
        ht,
        np.zeros(0, dtype=np_hsize_t),
        np.zeros(0, dtype=np_hsize_t),
        np.zeros((0, 1), dtype=np_hsize_t),
    )
    assert rows_as_digests(ht) == [b"\x00" * 32]  # Never written


@pytest.mark.parametrize("dtype", ["i8", "f4", "u1", "c16", "U3", "S2", "V4", "b1"])
def test_pod_dtypes_multichunk(dtype):
    if dtype in ("U3", "S2", "V4"):
        slab = np.array([b"abcd"[: np.dtype(dtype).itemsize]] * 6).astype(dtype)
    elif dtype == "b1":
        slab = np.array([True, False, True, False, True, True])
    else:
        slab = np.arange(6, dtype=dtype)
    ht = np.zeros((3, 4), dtype=np.uint64)
    hash_slab(
        slab,
        ht,
        np.array([0, 1, 2], dtype=np_hsize_t),
        np.array([0, 2, 4], dtype=np_hsize_t),
        np.array([[2], [2], [2]], dtype=np_hsize_t),
    )
    digests = rows_as_digests(ht)
    for j, start in ((0, 0), (1, 2), (2, 4)):
        assert digests[j] == reference(slab[start : start + 2])


def test_object_slab():
    slab = np.array([b"a", "bb", b"ccc", "dddd", b"", "f"], dtype=object)
    ht = np.zeros((3, 4), dtype=np.uint64)
    hash_slab(
        slab,
        ht,
        np.array([0, 1, 2], dtype=np_hsize_t),
        np.array([0, 2, 4], dtype=np_hsize_t),
        np.array([[2], [2], [2]], dtype=np_hsize_t),
    )
    digests = rows_as_digests(ht)
    for j, start in enumerate([0, 2, 4]):
        assert digests[j] == reference(slab[start : start + 2])


def test_object_edge_chunk_ignores_uninitialised():
    slab = np.array(["a", "bb", "ccc", "junk"], dtype=object)
    src_start = np.array([2], dtype=np_hsize_t)
    count = np.array([[1]], dtype=np_hsize_t)  # only "ccc"
    rows = np.array([0], dtype=np_hsize_t)
    ht = np.zeros((1, 4), dtype=np.uint64)
    hash_slab(slab, ht, rows, src_start, count)
    assert rows_as_digests(ht)[0] == reference(slab[2:3])


@pytest.mark.skipif(not HAS_NPYSTRINGS, reason="StringDType requires NumPy >=2.0")
def test_npystrings_slab():
    slab = np.array(["a", "bb", "ccc", "dddd"], dtype="T")
    ht = np.zeros((2, 4), dtype=np.uint64)
    hash_slab(
        slab,
        ht,
        np.array([0, 1], dtype=np_hsize_t),
        np.array([0, 2], dtype=np_hsize_t),
        np.array([[2], [2]], dtype=np_hsize_t),
    )
    digests = rows_as_digests(ht)
    assert digests[0] == reference(slab[0:2])
    assert digests[1] == reference(slab[2:4])


def test_identical_chunks_same_hash():
    slab = np.array([1, 2, 1, 2], dtype="i8")  # chunk 0 == chunk 1
    ht = np.zeros((2, 4), dtype=np.uint64)
    hash_slab(
        slab,
        ht,
        np.array([0, 1], dtype=np_hsize_t),
        np.array([0, 2], dtype=np_hsize_t),
        np.array([[2], [2]], dtype=np_hsize_t),
    )
    assert rows_as_digests(ht)[0] == rows_as_digests(ht)[1]
