"""Test versioned_hdf5.hash.hash_slab.

See test_hash_legacy_compat.py for cross-checks against the legacy hash algorithm.
"""

import hashlib
import struct

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
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
        (10,),
    )
    assert rows_as_digests(ht)[0] == reference(slab)


def test_multiple_chunks_route_to_rows():
    """Each chunk's digest lands in the row named by hash_rows;
    unlisted rows stay untouched.
    """
    slab = np.arange(16, dtype="i4")  # 4 chunks
    ht = np.zeros((4, 4), dtype=np.uint64)
    # Hash chunks 0, 2 and 3 (skipping chunk 1), and (to prove hash_rows is honoured)
    # deliberately route chunk at offset 8 to row 2 and the one at offset 12 to row 3.
    hash_slab(
        slab,
        ht,
        np.array([0, 2, 3], dtype=np_hsize_t),
        np.array([0, 8, 12], dtype=np_hsize_t),
        np.array([[4], [4], [4]], dtype=np_hsize_t),
        (4,),
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
    hash_slab(slab, ht, rows, src_start, count, (3, 5))
    h0, h1 = rows_as_digests(ht)
    assert h0 == reference(slab[0:3, :])
    assert h1 == reference(slab[3:5, :])

    # Poison the uninitialised row and re-hash: the digests must be unchanged.
    slab[5] = 999
    ht2 = np.zeros((2, 4), dtype=np.uint64)
    hash_slab(slab, ht2, rows, src_start, count, (3, 5))
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
        (4, 5),
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
        (3, 4),
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
        (1, 3),
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
        (1,),
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
        (2,),
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
        (2,),
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
    hash_slab(slab, ht, rows, src_start, count, (1,))
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
        (2,),
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
        (2,),
    )
    assert rows_as_digests(ht)[0] == rows_as_digests(ht)[1]


VLEN_DTYPES = [
    "object",
    pytest.param(
        "T",
        marks=pytest.mark.skipif(
            not HAS_NPYSTRINGS, reason="StringDType requires NumPy >=2.0"
        ),
    ),
]


def hash_chunk(
    slab, src_start: int = 0, count: tuple[int, ...] | None = None, dtype=None
) -> bytes:
    """Hash a single chunk of ``slab`` with hash_slab and return its digest,
    double-checking it against the reference implementation.

    The chunk is ``slab[src_start:src_start + count[0], :count[1], ...]``;
    ``count`` defaults to the whole slab.
    """
    slab = np.asarray(slab, dtype=dtype)
    count = slab.shape if count is None else count
    ht = np.zeros((1, 4), dtype=np.uint64)
    hash_slab(
        slab,
        ht,
        np.array([0], dtype=np_hsize_t),
        np.array([src_start], dtype=np_hsize_t),
        np.array([count], dtype=np_hsize_t),
        count,
    )
    digest = rows_as_digests(ht)[0]
    idx = (slice(src_start, src_start + count[0]), *(slice(c) for c in count[1:]))
    assert digest == reference(slab[idx])
    return digest


@pytest.mark.parametrize("dtype", VLEN_DTYPES)
@pytest.mark.parametrize(
    ("a", "b"),
    [
        (["foo", "bar"], ["foob", "ar"]),
        (["a", "b"], ["ab", ""]),
        (["ab", ""], ["", "ab"]),
        (["ab", ""], ["ab"]),
        (["", "ab"], ["ab"]),
    ],
)
def test_no_collision_vlen_split_points(dtype, a, b):
    """Moving the boundaries between variable-width strings, without changing their
    concatenation, changes the hash.
    """
    assert "".join(a) == "".join(b)
    assert hash_chunk(a, dtype=dtype) != hash_chunk(b, dtype=dtype)


def test_no_collision_vlen_embedded_length_prefix():
    """A variable-width string that spells out its own length prefix can't be used to
    forge the element boundaries.
    """
    forged = b"x" + struct.pack("<Q", 1) + b"y"
    plain = hash_chunk([b"x", b"y"], dtype=object)
    assert plain != hash_chunk([forged], dtype=object)
    # Same shape as the plain chunk, to also rule out the shape suffix doing the work
    assert plain != hash_chunk([forged, b""], dtype=object)


def test_no_collision_vlen_empty_and_nul():
    """Empty strings, and strings made of NUL bytes, are all distinguished from each
    other and from their own repetitions.
    """
    cases = [
        [b""],
        [b"", b""],
        [b"", b"", b""],
        [b"\x00"],
        [b"\x00", b""],
        [b"", b"\x00"],
        [b"\x00\x00"],
        # Unlike fixed-width bytes, a trailing NUL is significant for VLEN
        [b"a"],
        [b"a\x00"],
    ]
    digests = {hash_chunk(case, dtype=object) for case in cases}
    assert len(digests) == len(cases)


# ------------------------------------------------------------------------------
# Non-contiguous input geometries
#
# hash_slab may be handed any NumPy view as a slab: transposed, stepped along any
# axis, broadcast, reversed, or combinations thereof. In all cases the digest must
# match the C-contiguous materialization of the chunk.
# ------------------------------------------------------------------------------


def make_ht(n):
    return np.zeros((n, 4), dtype=np.uint64)


def hash_single_chunk(slab, count=None):
    """Hash the whole of ``slab`` (or its ``count``-trimmed prefix) as one chunk."""
    count = slab.shape if count is None else count
    ht = make_ht(1)
    hash_slab(
        slab,
        ht,
        np.array([0], dtype=np_hsize_t),
        np.array([0], dtype=np_hsize_t),
        np.array([count], dtype=np_hsize_t),
        count,
    )
    return rows_as_digests(ht)[0]


def test_transposed_2d():
    slab = np.arange(30, dtype="i8").reshape(5, 6).T  # F-contiguous
    assert not slab.flags.c_contiguous
    assert hash_single_chunk(slab) == reference(slab)


def test_transposed_3d():
    a = np.arange(60, dtype="f4").reshape(3, 4, 5)
    for perm in [(1, 0, 2), (2, 1, 0), (0, 2, 1), (2, 0, 1)]:
        slab = a.transpose(perm)
        assert hash_single_chunk(slab) == reference(slab)


def test_fortran_order():
    slab = np.asfortranarray(np.arange(24, dtype="i2").reshape(4, 6))
    assert slab.flags.f_contiguous
    assert not slab.flags.c_contiguous
    assert hash_single_chunk(slab) == reference(slab)


def test_step_inner_axis():
    # Non-contiguous along the innermost axis
    slab = np.arange(40, dtype="i8").reshape(4, 10)[:, ::2]
    assert slab.strides[-1] != slab.dtype.itemsize
    assert hash_single_chunk(slab) == reference(slab)


def test_step_inner_axis_3d():
    slab = np.arange(120, dtype="i8").reshape(4, 5, 6)[:, ::3, ::2]
    assert slab.strides[-1] != slab.dtype.itemsize
    assert hash_single_chunk(slab) == reference(slab)


def test_step_outer_axis():
    # Contiguous along the innermost axis only; rows are contiguous runs
    slab = np.arange(60, dtype="i8").reshape(6, 10)[::2]
    assert slab.strides[-1] == slab.dtype.itemsize
    assert not slab.flags.c_contiguous
    assert hash_single_chunk(slab) == reference(slab)


def test_step_middle_axis():
    slab = np.arange(120, dtype="i8").reshape(4, 5, 6)[:, ::2]
    assert slab.strides[-1] == slab.dtype.itemsize
    assert not slab.flags.c_contiguous
    assert hash_single_chunk(slab) == reference(slab)


def test_trailing_axes_contiguous_3d():
    # Contiguous along axes 1 and 2, but not along axis 0
    slab = np.arange(180, dtype="i8").reshape(6, 5, 6)[::2]
    assert slab.strides == (60 * 8, 6 * 8, 8)
    assert hash_single_chunk(slab) == reference(slab)


def test_reversed_axis_0():
    slab = np.arange(30, dtype="i8").reshape(5, 6)[::-1]
    assert slab.strides[0] < 0
    assert hash_single_chunk(slab) == reference(slab)


def test_reversed_all_axes():
    slab = np.arange(60, dtype="i8").reshape(3, 4, 5)[::-1, ::-1, ::-1]
    assert all(s < 0 for s in slab.strides)
    assert hash_single_chunk(slab) == reference(slab)


def test_broadcast_inner_axis():
    # Zero stride on axis 0, contiguous innermost axis
    slab = np.broadcast_to(np.arange(1, 5, dtype="i8"), (3, 4))
    assert slab.strides == (0, 8)
    assert hash_single_chunk(slab) == reference(slab)


def test_broadcast_full_2d():
    slab = np.broadcast_to(np.array(7, dtype="i4"), (3, 5))
    assert slab.strides == (0, 0)
    assert hash_single_chunk(slab) == reference(slab)


def test_broadcast_full_3d():
    slab = np.broadcast_to(1.5, (2, 3, 4))
    assert hash_single_chunk(slab) == reference(slab)


def test_broadcast_1d():
    slab = np.broadcast_to(np.uint8(3), (7,))
    assert slab.strides == (0,)
    assert hash_single_chunk(slab) == reference(slab)


def test_expand_dims():
    slab = np.arange(12, dtype="i8").reshape(4, 3)[:, np.newaxis, :]
    assert hash_single_chunk(slab) == reference(slab)


def test_transposed_then_stepped():
    slab = np.arange(60, dtype="i8").reshape(5, 12).T[::3]
    assert slab.strides[-1] != slab.dtype.itemsize
    assert hash_single_chunk(slab) == reference(slab)


def test_strided_slab_multichunk():
    """Multiple chunks, with edge trimming and hash row permutation, carved out of a
    stepped (outer axis) slab: the chunk axis-0 offset must use the real stride.
    """
    slab = np.arange(80, dtype="i8").reshape(8, 10)[::2]  # shape (4, 10)
    # 2 chunks of 2 rows each, second one edge-trimmed to 1 row and 7 columns
    src_start = [0, 2]
    counts = [(2, 10), (1, 7)]
    ht = make_ht(2)
    hash_slab(
        slab,
        ht,
        np.array([1, 0], dtype=np_hsize_t),
        np.array(src_start, dtype=np_hsize_t),
        np.array(counts, dtype=np_hsize_t),
        (2, 10),
    )
    digests = rows_as_digests(ht)
    assert digests[1] == reference(slab[0:2, :10])
    assert digests[0] == reference(slab[2:3, :7])


def test_transposed_slab_multichunk():
    slab = np.arange(40, dtype="i4").reshape(4, 10).T  # shape (10, 4)
    ht = make_ht(2)
    hash_slab(
        slab,
        ht,
        np.array([0, 1], dtype=np_hsize_t),
        np.array([0, 5], dtype=np_hsize_t),
        np.array([(5, 4), (5, 3)], dtype=np_hsize_t),
        (5, 4),
    )
    digests = rows_as_digests(ht)
    assert digests[0] == reference(slab[0:5, :4])
    assert digests[1] == reference(slab[5:10, :3])


def test_object_strided_slab():
    slab = np.array([b"a", "bb", b"ccc", "dddd", "e", "ff"], dtype=object)[::2]
    assert hash_single_chunk(slab) == reference(slab)


# ------------------------------------------------------------------------------
# Hypothesis: randomized slab geometries
# ------------------------------------------------------------------------------


def _transformations(a: np.ndarray, rng):
    """Yield a handful of strided/broadcast views of ``a``."""
    yield "identity", a
    if a.ndim >= 2:
        yield "transpose", a.T
        yield "fortran", np.asfortranarray(a)
    yield "step0", a[:: 1 + rng.integers(1, 3)]
    if a.ndim >= 2:
        yield "step_inner", a[..., :: 1 + rng.integers(1, 3)]
        yield "step0_step_inner", a[::2, ..., ::2]
    if a.ndim >= 3:
        yield "step1", a[:, :: 1 + rng.integers(1, 3)]
    yield "reverse0", a[::-1]
    if a.ndim >= 2:
        yield "reverse_all", a[::-1, ..., ::-1]
    yield "expand0", a[np.newaxis]
    yield "expand_inner", a[..., np.newaxis]


@st.composite
def _slabs(draw):
    ndim = draw(st.integers(1, 3))
    shape = tuple(draw(st.integers(1, 5)) for _ in range(ndim))
    dtype = draw(st.sampled_from(["i8", "i4", "u1", "f8"]))
    a = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    name, slab = draw(st.sampled_from(list(_transformations(a, rng))))
    return name, slab


@pytest.mark.parametrize("dtype", VLEN_DTYPES)
def test_strided_single_chunk_object(dtype):
    slab = np.array(["a", "bb", "", "ccc", "dddd", "e"], dtype=dtype)[::2]
    assert hash_single_chunk(slab) == reference(slab)


@given(_slabs())
def test_hypothesis_single_chunk(slab_and_name):
    _, slab = slab_and_name
    digest = hash_single_chunk(slab)
    assert digest == reference(slab)


@given(_slabs(), st.data())
def test_hypothesis_multichunk(slab_and_name, data):
    """Random tiled chunks carved out of a randomly strided slab."""
    _, slab = slab_and_name
    ndim = slab.ndim
    # Chunks partition the slab's rows: draw the physical chunk width and how
    # many chunks fit, then trim the counts along the other axes randomly.
    chunk_size_0 = data.draw(st.integers(1, slab.shape[0]))
    nchunks = data.draw(
        st.integers(1, (slab.shape[0] + chunk_size_0 - 1) // chunk_size_0)
    )
    src_start = []
    counts = []
    for i in range(nchunks):
        start = i * chunk_size_0
        c0 = min(chunk_size_0, slab.shape[0] - start)
        count = [c0]
        for j in range(1, ndim):
            count.append(data.draw(st.integers(1, slab.shape[j])))
        src_start.append(start)
        counts.append(count)
    # hash_rows may repeat/permute freely
    hash_rows = np.arange(nchunks, dtype=np_hsize_t)[::-1].copy()
    ht = make_ht(nchunks)
    counts_arr = np.array(counts, dtype=np_hsize_t)
    hash_slab(
        slab,
        ht,
        hash_rows,
        np.array(src_start, dtype=np_hsize_t),
        counts_arr,
        tuple(counts_arr.max(axis=0)),
    )
    digests = rows_as_digests(ht)
    for i in range(nchunks):
        idx = (slice(src_start[i], src_start[i] + counts[i][0]),)
        idx += tuple(slice(c) for c in counts[i][1:])
        assert digests[hash_rows[i]] == reference(slab[idx])


GEOMETRIES = [(6,), (1, 6), (6, 1), (2, 3), (3, 2), (1, 1, 6), (1, 6, 1), (6, 1, 1)]


@pytest.mark.parametrize("dtype", ["i8", "u1", *VLEN_DTYPES])
def test_no_collision_same_data_different_geometry(dtype):
    """The same six values, arranged in different shapes, hash differently.

    Every reshape below is C-contiguous, so the data bytes are byte-for-byte the same
    in all cases and the shape suffix is the only discriminator.
    """
    if dtype in ("i8", "u1"):
        flat = np.arange(6, dtype=dtype)
    else:
        flat = np.array(["a", "bb", "", "d", "e", "ff"], dtype=dtype)
    digests = {hash_chunk(flat.reshape(shape)) for shape in GEOMETRIES}
    assert len(digests) == len(GEOMETRIES)


def test_no_collision_chunk_geometry_within_slab():
    """Chunks carved out of the same slab with the same number of identical elements,
    but different edge-trimmed shapes, hash differently.

    The data bytes are four zeros in every case, so the shape suffix is the only
    discriminator; (4, 1) and (2, 2) also go through the strided walk instead of the
    single-blob path.
    """
    slab = np.zeros((4, 4), dtype="i8")
    digests = {hash_chunk(slab, count=count) for count in [(1, 4), (4, 1), (2, 2)]}
    # A 1-dimensional slab of the same four zeros is different again
    digests.add(hash_chunk(np.zeros(4, dtype="i8"), count=(4,)))
    assert len(digests) == 4

    # Conversely, the offset of the chunk within the slab is not part of the digest
    assert hash_chunk(slab, count=(2, 2)) == hash_chunk(slab, 2, (2, 2))


def test_no_collision_shape_string_injection():
    """Data bytes that spell out a shape string can't be used to forge the shape
    suffix, no matter how they line up with the real one.
    """
    cases = [
        np.frombuffer(b"(4,)", dtype="u1"),  # b"(4,)" + b"(4,)"
        np.frombuffer(b"(8,)", dtype="u1"),  # b"(8,)" + b"(4,)"
        np.frombuffer(b"(1, 4)", dtype="u1"),  # b"(1, 4)" + b"(6,)"
        np.frombuffer(b"(6,)", dtype="u1").reshape(1, 4),  # b"(6,)" + b"(1, 4)"
        np.frombuffer(b"ab(2,)", dtype="u1"),  # b"ab(2,)" + b"(6,)"
        np.frombuffer(b"ab", dtype="u1"),  # b"ab" + b"(2,)"
    ]
    digests = {hash_chunk(case) for case in cases}
    assert len(digests) == len(cases)


@pytest.mark.parametrize("dtype", ["f4", "f8", "c8", "c16"])
def test_no_collision_signed_zero(dtype):
    """+0.0 and -0.0 compare equal, but they are not the same chunk."""
    a = np.zeros(4, dtype=dtype)
    b = -a
    assert (a == b).all()
    assert hash_chunk(a) != hash_chunk(b)


def test_nan_hashes_bitwise():
    """Chunks are deduplicated by raw bytes, not by value: two chunks of NaNs with
    the same bit pattern hash the same, even though they compare unequal
    """
    a = np.array([np.nan, 1.0])
    assert a[0] != a[0]
    assert hash_chunk(a) == hash_chunk(a.copy())


def test_no_collision_multichunk_geometry():
    """Chunks of different shapes hashed in a single hash_slab call don't contaminate
    each other; the shape suffix of one chunk must not leak into the next.
    """
    slab = np.zeros((6, 4), dtype="i8")
    starts = [0, 1, 2, 3]
    counts = [(1, 4), (1, 2), (1, 1), (3, 4)]
    ht = np.zeros((4, 4), dtype=np.uint64)
    hash_slab(
        slab,
        ht,
        np.array([0, 1, 2, 3], dtype=np_hsize_t),
        np.array(starts, dtype=np_hsize_t),
        np.array(counts, dtype=np_hsize_t),
        (4, 4),
    )
    digests = rows_as_digests(ht)
    assert digests == [
        hash_chunk(slab, start, count)
        for start, count in zip(starts, counts, strict=True)
    ]
    assert len(set(digests)) == 4


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # Same bits, different signedness or kind
        (np.array([-1, -2], "i4"), np.array([2**32 - 1, 2**32 - 2], "u4")),
        (np.array([0, 0], "i8"), np.array([0.0, 0.0], "f8")),
        (np.array([True, False]), np.array([1, 0], "u1")),
        # Same bits, different byte order, hence different values
        (np.array([1], ">i4"), np.array([1 << 24], "<i4")),
        # Same bits, different time unit, hence different instants
        (np.array([1, 2], "M8[s]"), np.array([1, 2], "M8[ns]")),
        (np.array([1, 2], "M8[s]"), np.array([1, 2], "m8[s]")),
        # Fixed-width strings vs. their raw buffers
        (np.array([b"a"], "S4"), np.array(["a"], "U1")),
        (np.array([b"ab"], "S2"), np.array([b"ab"], "V2")),
        # Structured vs. flat
        (
            np.array([(1, 2)], dtype=[("a", "i4"), ("b", "i4")]),
            np.array([2**33 + 1], "u8"),
        ),
        # A fixed-width string that spells out the length prefix of a VLEN one
        (
            np.array([b"ab"], dtype=object),
            np.array([struct.pack("<Q", 2) + b"ab"], dtype="S10"),
        ),
        (np.array([b""], dtype=object), np.array([0], "u8")),
        # Empty chunks hash their shape and nothing else
        (np.zeros((0, 3), dtype="i8"), np.zeros((0, 3), dtype=object)),
    ],
)
def test_known_collisions_reinterpreted_dtype(a, b):
    """The dtype is not part of the digest, so two chunks of the same shape with the
    same raw byte image collide even when they mean entirely different things.

    This is the only possible collision family and it is unreachable: a hash table is
    shared exclusively by the chunks of one raw_data dataset, and
    backend.check_compatible_dtypes() refuses to write to it anything but the dtype it
    was created with (bar the object/StringDType equivalence, which is deliberate; see
    test_vlen_encoding_equivalence). ``modify_metadata(dtype=...)`` rewrites the whole
    file, hash table included.
    """
    assert a.dtype != b.dtype
    assert hash_chunk(a) == hash_chunk(b)


def test_vlen_encoding_equivalence():
    """Variable-width strings are hashed by their UTF-8 bytes, so ``str``, ``bytes``
    and StringDType spellings of the same value collide *by design*.

    h5py hands back the contents of a VLEN string dataset as either ``bytes`` or
    ``str`` depending on how it is read, and the same dataset can be read as an
    object or a StringDType array.
    """
    digests = {
        hash_chunk(np.array(values, dtype=object))
        for values in (["é", "a"], [b"\xc3\xa9", b"a"], ["é", b"a"])
    }
    if HAS_NPYSTRINGS:
        digests.add(hash_chunk(np.array(["é", "a"], dtype="T")))
    assert len(digests) == 1
