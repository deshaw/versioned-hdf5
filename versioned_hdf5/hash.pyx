"""Vectorized SHA256 hasher for the chunks of a StagedChangesArray slab.

This is designed and optimized to hash many (potentially small) chunks of a slab
in a single call with an API that closely mirrors
:func:`versioned_hdf5.slicetools.read_many_slices`.

Output hashes are identical to the legacy :mod:`versioned_hdf5.hashtable`.
The invariant is enforced by tests/test_hash_legacy_compat.py.

The actual SHA256 is computed by the interpreter's :func:`hashlib.sha256` (OpenSSL in
most CPython builds, CPython's own implementation otherwise), so there is no build-time
or run-time dependency on OpenSSL. See :class:`_Hasher` for how the per-chunk cost of
the Python calls is kept low.
"""
import hashlib

import numpy as np
from numpy cimport npy_intp, NPY_MAXDIMS

cimport cython
from cpython.buffer cimport PyBUF_WRITABLE
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_GET_SIZE
from cpython.unicode cimport PyUnicode_AsUTF8AndSize
from libc.stdint cimport uint64_t
from libc.stdio cimport snprintf
from libc.stdlib cimport free, realloc
from libc.string cimport memcmp, memcpy

from versioned_hdf5.cytools cimport hsize_t

cdef extern from "Python.h":
    bint PyUnicode_IS_ASCII(object o)

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


cdef enum:
    # Strided blocks of at least this many bytes are passed to hashlib one by one;
    # smaller ones cost less to copy to a contiguous buffer than a Python call each.
    DIRECT_MINSIZE = 512


# copy() of an initialised object is cheaper than the constructor
cdef object _sha256_init = hashlib.sha256()


@cython.auto_pickle(False)
cdef class _Hasher:
    """SHA256 of one chunk at a time, computed by hashlib.

    The instance itself is the read-only buffer passed to hashlib, pointing at whatever
    ``ptr`` and ``size`` say at the time of the call. hashlib releases the buffer before
    it returns, so no memoryview is allocated per chunk.

    The instance also owns a scratch buffer, where the chunks that are too fragmented
    in memory are assembled before they are hashed.
    """
    cdef object h  # the hashlib object of the current chunk
    cdef const void* ptr
    cdef Py_ssize_t size
    cdef unsigned char* scratch
    cdef Py_ssize_t capacity
    # str(shape) of the last chunk hashed; most chunks of a slab share it.
    cdef hsize_t[NPY_MAXDIMS] shape
    cdef int ndim
    cdef char[NPY_MAXDIMS * 22 + 2] shape_str  # ", " + 20 digits for each axis
    cdef int shape_len

    def __cinit__(self):
        self.h = _sha256_init.copy()
        self.ndim = -1

    def __dealloc__(self):
        free(self.scratch)

    def __getbuffer__(self, Py_buffer* buf, int flags):
        if flags & PyBUF_WRITABLE:
            raise BufferError("read-only buffer")
        buf.buf = <void*>self.ptr
        buf.obj = self
        buf.len = self.size
        buf.readonly = 1
        buf.itemsize = 1
        buf.format = NULL
        buf.ndim = 1
        buf.shape = &self.size
        buf.strides = NULL
        buf.suboffsets = NULL
        buf.internal = NULL

    def __releasebuffer__(self, Py_buffer* buf):
        pass

    cdef int reserve(self, Py_ssize_t size) except -1:
        """Ensure that the scratch buffer holds at least size bytes, preserving its
        contents.
        """
        cdef void* scratch
        if size > self.capacity:
            size = max(size, 2 * self.capacity)
            scratch = realloc(self.scratch, size)
            if scratch == NULL:
                raise MemoryError()
            self.scratch = <unsigned char*>scratch
            self.capacity = size
        return 0

    cdef int update(self, const void* ptr, Py_ssize_t size) except -1:
        """Hash size bytes at ptr, without copying them."""
        self.ptr = ptr
        self.size = size
        self.h.update(self)
        return 0

    cdef int digest(self, const hsize_t* shape, int ndim, uint64_t* out) except -1:
        """Hash ``str(tuple(shape))``, write the 32-byte digest of the chunk to out and
        start a new chunk.

        The shape is formatted with C snprintf to avoid any Python interaction.
        This is convoluted but is necessary to generate hashes with DATA_VERSION=4.
        """
        cdef int i
        cdef int nchars = 1
        cdef bytes digest

        if ndim != self.ndim or memcmp(shape, self.shape, ndim * sizeof(hsize_t)):
            # Note: 0d arrays are not supported by HDF5
            self.shape_str[0] = b"("
            for i in range(ndim):
                nchars += snprintf(
                    self.shape_str + nchars,
                    sizeof(self.shape_str) - nchars,
                    "%llu, ",
                    <unsigned long long>shape[i],
                )
            nchars -= 1 if ndim == 1 else 2  # "(1,)" vs. "(1, 2)"
            self.shape_str[nchars] = b")"
            self.shape_len = nchars + 1
            self.ndim = ndim
            memcpy(self.shape, shape, ndim * sizeof(hsize_t))

        self.update(self.shape_str, self.shape_len)
        digest = self.h.digest()
        memcpy(out, PyBytes_AS_STRING(digest), 32)
        self.h = _sha256_init.copy()
        return 0


cpdef void hash_slab(
    np.ndarray src,
    uint64_t[:, ::1] hash_table,
    hsize_t[::1] hash_rows,
    hsize_t[::1] src_start,
    hsize_t[:, ::1] count,
    chunk_size,  # tuple[int, ...]
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
        full slab). It may be strided: chunks that are not C-contiguous are
        hashed one contiguous block at a time, or through a chunk-sized scratch buffer
        if the blocks are small.
        Note that non-NumPy base slabs (e.g. backed by h5py) are never hashed here;
        their hashes are loaded from disk.
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
    chunk_size:
        The physical (untrimmed) chunk shape, e.g. the maximum that can be contained on
        any one row of ``count``.
    """
    cdef hsize_t nchunks = count.shape[0]
    cdef int ndim = count.shape[1]
    cdef hsize_t i
    cdef int j, n_outer
    cdef Py_ssize_t itemsize = src.dtype.itemsize
    cdef Py_ssize_t stride0, block, nblocks
    cdef const unsigned char* data
    cdef _Hasher hasher = _Hasher()

    assert src.ndim == ndim
    assert hash_table.shape[1] == 4
    assert hash_rows.shape[0] == nchunks
    assert src_start.shape[0] == nchunks
    assert isinstance(chunk_size, tuple)
    assert len(chunk_size) == ndim

    if src.dtype.kind in ("O", "T"):
        for i in range(nchunks):
            idx = [slice(src_start[i], src_start[i] + count[i, 0])]
            for j in range(1, ndim):
                idx.append(slice(count[i, j]))
            _hash_object_chunk(hasher, src[tuple(idx)], &hash_table[hash_rows[i], 0])
        return

    if src.size > 0 and not any((<object>src).strides):
        # The slab is broadcasted from a scalar, e.g. the full slab, so all chunks are
        # views of the same one. Materialize it once, instead of once per chunk.
        # This could also be a much larger user-provided broadcasted array, which we
        # don't want to materialize as contiguous all at once.
        src = np.full(chunk_size, fill_value=src.flat[0], dtype=src.dtype)
        stride0 = 0
    else:
        stride0 = src.strides[0]

    # hashlib needs the GIL, but releases it while it hashes 2 kiB or more at a time.
    for i in range(nchunks):
        # Each chunk is a slice along axis 0
        data = <const unsigned char*>src.data + stride0 * <Py_ssize_t>src_start[i]

        # Split the axes into n_outer leading ones and the trailing ones, which form
        # C-contiguous blocks of `block` bytes. This must be tested per chunk: edge
        # chunks with trimmed counts may be non-contiguous even on a C-contiguous slab.
        block = itemsize
        n_outer = ndim
        while n_outer > 0 and src.strides[n_outer - 1] == block:
            n_outer -= 1
            block *= count[i, n_outer]
        nblocks = 1
        for j in range(n_outer):
            nblocks *= count[i, j]

        if n_outer == 0:
            hasher.update(data, block)
        elif nblocks * block > 0:
            _hash_blocks(
                hasher, data, &count[i, 0], src.strides, n_outer, nblocks, block
            )

        hasher.digest(&count[i, 0], ndim, &hash_table[hash_rows[i], 0])


cdef int _hash_blocks(
    _Hasher hasher,
    const unsigned char* src,
    const hsize_t* shape,
    const npy_intp* strides,
    int ndim,
    Py_ssize_t nblocks,
    Py_ssize_t block,
) except -1:
    """Hash, in C order, the non-empty array of blocks of `block` contiguous bytes
    each, with the given shape and strides, that starts at src.

    Small blocks are first deep-copied to the scratch buffer, like
    np.ascontiguousarray() would.
    """
    cdef hsize_t[NPY_MAXDIMS] idx
    cdef Py_ssize_t n_inner = shape[ndim - 1]
    cdef npy_intp stride_inner = strides[ndim - 1]
    cdef bint direct = block >= DIRECT_MINSIZE
    cdef unsigned char* dst = NULL
    cdef Py_ssize_t i
    cdef int k

    if not direct:
        hasher.reserve(nblocks * block)
        dst = hasher.scratch
    for k in range(ndim - 1):
        idx[k] = 0

    while True:
        if direct:
            for i in range(n_inner):
                hasher.update(src + i * stride_inner, block)
        else:
            for i in range(n_inner):
                memcpy(dst, src + i * stride_inner, block)
                dst += block

        # Advance the indices of the leading axes
        for k in range(ndim - 2, -1, -1):
            src += strides[k]
            idx[k] += 1
            if idx[k] < shape[k]:
                break
            src -= strides[k] * <npy_intp>shape[k]
            idx[k] = 0
        else:
            break

    if not direct:
        hasher.update(hasher.scratch, nblocks * block)
    return 0


cdef int _hash_object_chunk(_Hasher hasher, np.ndarray data, uint64_t* out) except -1:
    """Hash a single chunk with object dtype or StringDType (variable-width strings).

    The chunk is serialized to the scratch buffer of the hasher, as hashing each string
    on its own would cost two Python calls per string.
    """
    cdef const char* value_ptr
    cdef Py_ssize_t nbytes
    cdef Py_ssize_t pos = 0
    cdef uint64_t nbytes_le

    # Ensure StringDType and object strings produce the same hash
    # TODO speed this up with native C NumPy APIs
    if data.dtype.kind == "T":
        data = data.astype(object)

    for value in data.flat:
        if isinstance(value, str) and not PyUnicode_IS_ASCII(value):
            # Don't call PyUnicode_AsUTF8AndSize: it would cache the encoded string
            # for as long as the str object lives.
            value = value.encode("utf-8")

        if isinstance(value, str):
            # ASCII: this is a pointer to the contents of the str object
            value_ptr = PyUnicode_AsUTF8AndSize(value, &nbytes)
        elif isinstance(value, bytes):
            value_ptr = PyBytes_AS_STRING(value)
            nbytes = PyBytes_GET_SIZE(value)
        else:
            raise ValueError(f"Object array contains unsupported type={type(value)}")

        hasher.reserve(pos + 8 + nbytes)
        # On x86 and ARM, this is a no-op. On PowerPC, swap endianness
        nbytes_le = htole64(nbytes)
        memcpy(hasher.scratch + pos, &nbytes_le, 8)
        memcpy(hasher.scratch + pos + 8, value_ptr, nbytes)
        pos += 8 + nbytes

    hasher.update(hasher.scratch, pos)
    return hasher.digest(<hsize_t*>data.shape, data.ndim, out)
