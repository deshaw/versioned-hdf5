"""Vectorized SHA256 hasher for the chunks of a StagedChangesArray slab.

This is designed and optimized to hash many (potentially small) chunks of a slab
in a single call with an API that closely mirrors
:func:`versioned_hdf5.slicetools.read_many_slices`.

Output hashes are identical to the legacy :mod:`versioned_hdf5.hashtable`.
The invariant is enforced by tests/test_hash_legacy_compat.py.

The actual SHA256 is computed by the vendored sha256.c (public-domain B-Con
implementation), with no dependency on libcrypto or any other external library.
"""
import numpy as np
from numpy cimport npy_intp, NPY_MAXDIMS

from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from libc.stdio cimport snprintf

from versioned_hdf5.cytools cimport hsize_t

cdef extern from *:
    # libc defines htole64, but it is not available in MacOS. Let's implement our own.
    """
    #ifndef htole64
    #if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    static inline uint64_t htole64(uint64_t x) {return __builtin_bswap64(x);}
    #else
    static inline uint64_t htole64(uint64_t x) {return x;}
    #endif
    #endif
    """
    uint64_t htole64(uint64_t x)

cdef extern from "sha256.h":
    ctypedef struct SHA256_CTX:
        pass

    void sha256_init(SHA256_CTX* ctx) nogil
    void sha256_update(SHA256_CTX* ctx, const void* data, size_t cnt) nogil
    void sha256_final(SHA256_CTX* ctx, unsigned char* hash) nogil


cpdef void hash_slab(
    np.ndarray src,
    uint64_t[:, ::1] hash_table,
    hsize_t[::1] hash_rows,
    hsize_t[::1] src_start,
    hsize_t[:, ::1] count,
) except *:
    """Compute the SHA256 of one or more chunks of a slab and write them to a hash table.

    For each chunk i, this computes the SHA256 digest of the C-contiguous,
    edge-trimmed chunk::

        src[src_start[i, 0]      : src_start[i, 0]      + count[i, 0],
            src_start[i, 1]      : src_start[i, 1]      + count[i, 1],
            ...]

    exactly as if the chunk had been extracted from the virtual array and passed to
    :meth:`versioned_hdf5.hashtable.Hashtable.hash`, and writes the 32-byte digest into
    ``hash_table[hash_rows[i], :]`` (reinterpreted as 4 little-endian-on-x86
    ``hsize_t``).

    Uninitialised memory beyond the edge of a chunk (when ``count`` is smaller than the
    physical chunk size, e.g. on the last chunk along an axis) is never read.

    Parameters
    ----------
    src:
        The slab to read from; always a NumPy array (a staged slab, or the broadcasted
        full slab). Note that non-NumPy base slabs (e.g. backed by h5py) are never
        hashed here; their hashes are loaded from disk.
    hash_table:
        2D C-contiguous array of uint64 and shape ``(n, 4)``, modified in place.
        ``n`` is the number of chunks in the slab.
    hash_rows:
        1D array with one element per chunk to hash; the row of ``hash_table`` to write
        the digest of that chunk to.
    src_start:
        1D array with one element per chunk. ``src_start[i]`` is the offset along
        axis 0 of chunk i within its slab. Offsets on all other axes are always 0.
    count:
        2D array of shape ``(nchunks, ndim)`` with the edge-trimmed shape of each chunk,
        i.e. the actual number of valid points along each axis.
    """
    cdef hsize_t nchunks = count.shape[0]
    cdef hsize_t ndim = count.shape[1]
    cdef hsize_t i, j
    cdef hsize_t start
    cdef hsize_t itemsize = src.dtype.itemsize
    cdef hsize_t stride0
    cdef hsize_t offset
    cdef hsize_t total_bytes

    assert src.ndim == ndim
    assert hash_table.shape[1] == 4
    assert hash_rows.shape[0] == nchunks
    assert src_start.shape[0] == nchunks

    # Case 1: Object/StringDType — GIL always held (Python object iteration).
    if src.dtype.kind in ("O", "T"):
        for i in range(nchunks):
            offset = src_start[i]
            idx = [slice(offset, offset + count[i, 0])]
            for j in range(1, ndim):
                idx.append(slice(count[i, j]))
            chunk = src[tuple(idx)]
            _hash_object_chunk(
                chunk, &hash_table[hash_rows[i], 0], count[i], ndim
            )
        return

    # Case 2: full slab. This is a single chunk; it's ok to be suboptimal.
    if not src.flags.c_contiguous:
        src = np.ascontiguousarray(src)

    # Case 3: Non-object slab, at least C-contiguous along the innermost axis
    # (edge chunks may not be C-contiguous on axes[:-1])
    # release GIL for the loop.
    # Each chunk is a slice along axis 0; byte offset = start * stride0.
    with nogil:
        stride0 = itemsize
        for j in range(1, ndim):
            stride0 *= src.shape[j]

        for i in range(nchunks):
            offset = stride0 * src_start[i]
            total_bytes = stride0 * count[i, 0]
            _hash_chunk_from_ptr(
                <const unsigned char*>src.data + offset,
                &hash_table[hash_rows[i], 0],
                count[i],
                ndim,
                itemsize,
                src.strides,
                total_bytes,
            )


cdef void _hash_shape(SHA256_CTX* ctx, hsize_t* shape, int ndim) noexcept nogil:
    """Hash `str(tuple(shape))`.

    This is done with C snprintf to avoid any Python interaction.
    This is inefficient and convoluted but is necessary to generate
    hashes with DATA_VERSION=4.
    """
    cdef char shape_buf[4096]
    cdef int i
    cdef int nchars = 0
    cdef const char* fmt

    # Note: 0d arrays are not supported by HDF5
    for i in range(ndim):
        if ndim == 1:
            fmt = "(%llu,)"  # unsigned long long
        elif i == 0:
            fmt = "(%llu"
        elif i == ndim - 1:
            fmt = ", %llu)"
        else:
            fmt = ", %llu"

        nchars += snprintf(
            shape_buf + nchars,
            sizeof(shape_buf) - nchars,
            fmt,
            shape[i],
        )

    sha256_update(ctx, shape_buf, nchars)


cdef void _hash_chunk_from_ptr(
    const unsigned char* data_ptr,
    uint64_t* out,
    hsize_t[::1] shape,
    hsize_t ndim,
    hsize_t itemsize,
    npy_intp* strides,
    size_t total_bytes,
) noexcept nogil:
    """Hash a single chunk given a raw pointer and strides.
    The chunk must be not object dtype, not StringDType, not broadcasted, and
    C-contiguous at least along the innermost axis.
    """
    cdef SHA256_CTX ctx
    cdef hsize_t[NPY_MAXDIMS] outer_idx
    cdef hsize_t outer_total
    cdef hsize_t inner_size
    cdef hsize_t offset
    cdef hsize_t j
    cdef bint is_contiguous

    # Check C-contiguous: strides[j] == strides[j+1] * shape[j+1]
    is_contiguous = True
    for j in range(ndim - 1):
        if strides[j] != strides[j + 1] * shape[j + 1]:
            is_contiguous = False
            break

    sha256_init(&ctx)

    if is_contiguous:
        # Single contiguous blob
        sha256_update(&ctx, data_ptr, total_bytes)
    else:
        # Non-contiguous: walk in C-order, hash each row (innermost axis).
        # The innermost axis is contiguous for C-order arrays,
        # so each "row" (fixed indices on axes 0..ndim-2, all indices on
        # axis ndim-1) is a single contiguous run.
        inner_size = itemsize * shape[ndim - 1]

        # Iterate over outer dimensions (axes 0 .. ndim-2)
        outer_total = 1
        for j in range(ndim - 1):
            outer_idx[j] = 0
            outer_total *= shape[j]

        for outer in range(outer_total):
            # Compute byte offset for current outer position
            offset = 0
            for j in range(ndim - 1):
                offset += outer_idx[j] * strides[j]

            sha256_update(&ctx, data_ptr + offset, inner_size)

            # Advance outer indices
            for j in range(ndim - 2, -1, -1):
                outer_idx[j] += 1
                if outer_idx[j] < shape[j]:
                    break
                outer_idx[j] = 0

    _hash_shape(&ctx, &shape[0], ndim)
    sha256_final(&ctx, <unsigned char*>out)


cdef void _hash_object_chunk(
    np.ndarray data,
    uint64_t* out,
    hsize_t[::1] shape,
    hsize_t ndim,
) except *:
    """Hash a single chunk with object dtype or StringDType (variable-width strings).

    Object data forces us to hold the GIL throughout.
    """
    cdef SHA256_CTX ctx
    cdef bytes value_b
    cdef uint64_t nbytes_he
    cdef uint64_t nbytes_le

    sha256_init(&ctx)

    # Ensure StringDType and object strings produce the same hash
    # TODO speed this up with native C NumPy APIs
    if data.dtype.kind == "T":
        data = data.astype(object)

    for value in data.flat:
        if isinstance(value, str):
            value_b = value.encode("utf-8")
        elif isinstance(value, bytes):
            value_b = value
        else:
            raise ValueError(f"Object array contains unsupported type={type(value)}")

        nbytes_he = len(value_b)  # host-endian
        # On x86 and ARM, this is a no-op. On PowerPC, swap endianness
        nbytes_le = htole64(nbytes_he)  # little-endian
        sha256_update(&ctx, &nbytes_le, 8)
        sha256_update(&ctx, <const char*>value_b, nbytes_he)

    _hash_shape(&ctx, <hsize_t*>data.shape, data.ndim)
    sha256_final(&ctx, <unsigned char*>out)
