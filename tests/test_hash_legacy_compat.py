"""Test that versioned_hdf5.hash.hash_slab produces hashes that are bit-for-bit
identical to the legacy versioned_hdf5.hashtable.Hashtable.hash.
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pytest
from versioned_hdf5.hash import hash_slab

from versioned_hdf5.backend import create_base_dataset
from versioned_hdf5.cytools import np_hsize_t
from versioned_hdf5.h5py_compat import HAS_NPYSTRINGS
from versioned_hdf5.hashtable import Hashtable


@pytest.fixture
def legacy_hash(h5file):
    create_base_dataset(h5file, "x", data=np.empty((0,)))
    return Hashtable(h5file, "x").hash


def hash_one(chunk: np.ndarray) -> bytes:
    """Hash a single chunk via hash_slab (treating it as a one-chunk slab) and return
    the 32-byte digest.
    """
    chunk = np.asarray(chunk)
    hash_table = np.zeros((1, 4), dtype=np.uint64)
    hash_slab(
        chunk,
        hash_table,
        np.zeros(1, dtype=np_hsize_t),  # hash_rows
        np.zeros(1, dtype=np_hsize_t),  # src_start
        np.asarray([chunk.shape], dtype=np_hsize_t),  # count
    )
    return hash_table[0].view(np.uint8).tobytes()


def _all_chunks() -> Generator[tuple[str, np.ndarray]]:
    # Numeric dtypes
    for dt in ["i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8"]:
        yield (dt, np.arange(12, dtype=dt))
        yield (f"{dt}-2d", np.arange(12, dtype=dt).reshape(3, 4))

    for dt in ["f2", "f4", "f8"]:
        a = np.array([0.0, -0.0, 1.5, -2.5, np.nan, np.inf, -np.inf, 3.25], dtype=dt)
        yield (dt, a)
        yield (f"{dt}-2d", a.reshape(2, 4))

    for dt in ["c8", "c16"]:
        yield (dt, np.array([1 + 2j, -3 - 4j, 0j, complex(np.nan, 1)], dtype=dt))

    # Other fixed-width dtypes
    yield ("bool", np.array([True, False, True, True, False]))
    yield ("datetime64", np.array(["2021-01-01", "2022-02-02", "NaT"], "M8[D]"))
    yield ("timedelta64", np.array([1, 2, 3, -4], dtype="m8[s]"))
    yield ("bytes", np.array([b"abc", b"de", b"", b"fgh"], dtype="S3"))
    yield ("unicode", np.array(["abc", "de", "", "☃"], dtype="U3"))
    yield ("void", np.array([b"abc", b"xyz", b"\x00\x01\x02"], dtype="V3"))
    yield (
        (
            "structured",
            np.array(
                [(1, 2.5), (3, 4.5), (-7, np.nan)], dtype=[("a", "i4"), ("b", "f8")]
            ),
        )
    )

    # VLEN object strings (may or may not be encoded)
    yield ("object-str", np.array(["a", "bb", "ccc"], dtype=object))
    yield ("object-bytes", np.array([b"a", b"bb", b"ccc"], dtype=object))
    yield (
        "object-mixed-2d",
        np.array([b"a", "bb", b"ccc", "d"], dtype=object).reshape(2, 2),
    )
    # Empty elements: the per-element length prefix is what disambiguates these
    # from a single concatenated string (the data_version 3 bug).
    yield ("object-empty", np.array(["", "", ""], dtype=object))
    yield ("object-concat-a", np.array([b"a", b"b", b"cd"], dtype=object))
    yield ("object-concat-b", np.array([b"ab", b"", b"cd"], dtype=object))
    yield ("object-utf8", np.array(["☃", "café"], dtype=object))

    # StringDType a.k.a. NpyStrings
    if HAS_NPYSTRINGS:
        yield ("npystr", np.array(["a", "bb", "ccc"], dtype="T"))
        yield ("npystr-2d", np.array([["a", "bb"], ["", "dddd"]], dtype="T"))
        yield ("npystr-utf8", np.array(["☃", "café"], dtype="T"))


ALL_CHUNKS = list(_all_chunks())


@pytest.mark.parametrize(
    "chunk", [c for _, c in ALL_CHUNKS], ids=[i for i, _ in ALL_CHUNKS]
)
def test_matches_legacy(legacy_hash, chunk):
    assert hash_one(chunk) == legacy_hash(chunk)


def test_shape_sensitivity(legacy_hash):
    """Same bytes, different shape => different hash, in both implementations,
    and they agree with each other.
    """
    a = np.arange(6, dtype="i8").reshape(1, 6)
    b = np.arange(6, dtype="i8").reshape(6, 1)
    assert hash_one(a) == legacy_hash(a)
    assert hash_one(b) == legacy_hash(b)
    assert hash_one(a) != hash_one(b)


@pytest.mark.skipif(not HAS_NPYSTRINGS, reason="StringDType requires NumPy >=2.0")
def test_npystrings_object_equivalence(legacy_hash):
    """A StringDType array and the equivalent object-of-str array hash identically"""
    data = ["a", "bb", "", " cc "]
    t = np.array(data, dtype="T")
    o = np.array(data, dtype=object)
    assert hash_one(t) == hash_one(o) == legacy_hash(o) == legacy_hash(t)
