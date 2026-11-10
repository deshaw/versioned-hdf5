"""Vectorized SHA256 hasher for the chunks of a StagedChangesArray slab.

This is designed and optimized to hash many (potentially small) chunks of a slab
in a single call with an API that closely mirrors
:func:`versioned_hdf5.slicetools.read_many_slices`.

Output hashes are identical to the legacy :mod:`versioned_hdf5.hashtable`.
The invariant is enforced by tests/test_hash_legacy_compat.py.

The actual SHA256 is computed by the interpreter's :func:`hashlib.sha256` (OpenSSL in
most CPython builds, CPython's own implementation otherwise), so there is no build-time
or run-time dependency on OpenSSL. See :class:`_Sha256` for how the per-chunk cost of
the Python calls is kept low.
"""
import hashlib

import numpy as np
from numpy cimport npy_intp, NPY_MAXDIMS

cimport cython
from cpython.buffer cimport PyBUF_WRITABLE
from cpython.bytes cimport PyBytes_AS_STRING
from libc.stdint cimport uint64_t
from libc.stdio cimport snprintf
from libc.stdlib cimport free, malloc
from libc.string cimport memcmp, memcpy

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


cdef enum:
    # hashlib releases the GIL to hash a buffer of at least this many bytes
    # (HASHLIB_GIL_MINSIZE in CPython's Modules/hashlib.h). A hash object that received
    # such a buffer through update() takes a mutex in every later call; the constructor
    # hashes it without that switch.
    GIL_MINSIZE = 2048
    # Inputs below this size are copied into a buffer of BUF_SIZE bytes and passed on
    # together; larger ones cost less as a call of their own than as a copy.
    DIRECT_MINSIZE = 512
    BUF_SIZE = 65536


cdef object _sha256 = hashlib.sha256
# copy() of an initialised object is cheaper than the constructor for small inputs.
cdef object _sha256_init = hashlib.sha256()


@cython.auto_pickle(False)
cdef class _BufferView:
    """Read-only bytes at a pointer that the owner sets before each use.

    hashlib releases the buffer before it returns, so one instance serves every chunk
    and no memoryview is allocated per chunk.
    """
    cdef const unsigned char* ptr
    cdef Py_ssize_t size
    cdef Py_ssize_t shape[1]

    def __getbuffer__(self, Py_buffer* buf, int flags):
        if flags & PyBUF_WRITABLE:
            raise BufferError("read-only buffer")
        self.shape[0] = self.size
        buf.buf = <void*>self.ptr
        buf.obj = self
        buf.len = self.size
        buf.readonly = 1
        buf.itemsize = 1
        buf.format = NULL
        buf.ndim = 1
        buf.shape = self.shape
        buf.strides = NULL
        buf.suboffsets = NULL
        buf.internal = NULL

    def __releasebuffer__(self, Py_buffer* buf):
        pass


@cython.auto_pickle(False)
cdef class _Sha256:
    """SHA256 of a sequence of byte ranges, computed by hashlib.

    An input of DIRECT_MINSIZE bytes or more goes to hashlib without a copy; if it is the
    first one and has GIL_MINSIZE bytes or more, it is the argument of the constructor.
    Smaller inputs are copied into a buffer and passed on together, so a small chunk
    with its shape costs three Python calls: copy(), update() and digest().
    """
    cdef object h  # the hashlib object; None until the first call to hashlib
    cdef _BufferView view
    cdef unsigned char* buf
    cdef Py_ssize_t nbuf
    # str(shape) of the last chunk hashed; most chunks of a slab share it.
    cdef hsize_t shape[NPY_MAXDIMS]
    cdef int ndim
    cdef char shape_str[4096]
    cdef int shape_len

    def __cinit__(self):
        self.buf = <unsigned char*>malloc(BUF_SIZE)
        if self.buf == NULL:
            raise MemoryError()
        self.view = _BufferView()
        self.h = None
        self.nbuf = 0
        self.ndim = -1

    def __dealloc__(self):
        free(self.buf)

    cdef int _send(self, const unsigned char* ptr, Py_ssize_t size) except -1:
        self.view.ptr = ptr
        self.view.size = size
        if self.h is not None:
            self.h.update(self.view)
        elif size >= GIL_MINSIZE:
            self.h = _sha256(self.view)
        else:
            self.h = _sha256_init.copy()
            self.h.update(self.view)
        return 0

    cdef int _flush(self) except -1:
        if self.nbuf:
            self._send(self.buf, self.nbuf)
            self.nbuf = 0
        return 0

    cdef int update(self, const void* ptr, Py_ssize_t size) except -1:
        if size >= DIRECT_MINSIZE:
            self._flush()
            self._send(<const unsigned char*>ptr, size)
        elif size:
            if self.nbuf + size > BUF_SIZE:
                self._flush()
            memcpy(self.buf + self.nbuf, ptr, size)
            self.nbuf += size
        return 0

    cdef int finish(self, uint64_t* out) except -1:
        """Write the 32-byte digest to out and start over."""
        cdef bytes digest
        self._flush()
        if self.h is None:
            self.h = _sha256_init.copy()
        digest = self.h.digest()
        memcpy(out, PyBytes_AS_STRING(digest), 32)
        self.h = None
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
        full slab). Slabs that are not C-contiguous along the innermost axis are
        hashed through a chunk-sized scratch buffer; other axes may be strided.
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
    cdef hsize_t ndim = count.shape[1]
    cdef hsize_t i, j
    cdef hsize_t start
    cdef hsize_t itemsize = src.dtype.itemsize
    cdef hsize_t stride0
    cdef hsize_t offset
    cdef hsize_t n_contig
    cdef np.ndarray scratch
    cdef _Sha256 hasher = _Sha256()

    assert src.ndim == ndim
    assert hash_table.shape[1] == 4
    assert hash_rows.shape[0] == nchunks
    assert src_start.shape[0] == nchunks
    assert isinstance(chunk_size, tuple)
    assert len(chunk_size) == ndim

    if src.dtype.kind in ("O", "T"):
        # Case 1: Object/StringDType — GIL always held (Python object iteration).
        for i in range(nchunks):
            offset = src_start[i]
            idx = [slice(offset, offset + count[i, 0])]
            for j in range(1, ndim):
                idx.append(slice(count[i, j]))
            chunk = src[tuple(idx)]
            _hash_object_chunk(
                hasher, chunk, &hash_table[hash_rows[i], 0], count[i], ndim
            )
        return

    cdef bint is_broadcasted_scalar = True
    for j in range(ndim):
        if src.strides[j] != 0:
            is_broadcasted_scalar = False
            break

    # hashlib needs the GIL, but releases it while it hashes 2 kiB or more at a time.
    if is_broadcasted_scalar and src.size > 0:
        # Case 2: the full slab is broadcasted from scalar. However, this could also be
        # a much larger user-provided broadcasted array, which we don't want to
        # materialize as contiguous all at once.
        # Special case: an empty slab also has all-zero strides, but a None base;
        # fall through to Case 3 instead.
        scratch = np.full(chunk_size, fill_value=src.flat[0], dtype=src.dtype)
        for i in range(nchunks):
            _hash_chunk_from_ptr(
                hasher,
                <const unsigned char*>scratch.data,
                &hash_table[hash_rows[i], 0],
                count[i],
                ndim,
                ndim,
                itemsize,
                scratch.strides,
            )

    elif src.strides[ndim - 1] != itemsize:
        # Case 3: the innermost axis is strided. Deep-copy each chunk into a chunk-sized
        # scratch buffer and hash it as a contiguous blob.
        scratch = np.empty(chunk_size, dtype=src.dtype)
        for i in range(nchunks):
            _copy_chunk(
                <const unsigned char*>src.data + src.strides[0] * src_start[i],
                <unsigned char*>scratch.data,
                count[i],
                ndim,
                itemsize,
                src.strides,
            )
            _hash_chunk_from_ptr(
                hasher,
                <const unsigned char*>scratch.data,
                &hash_table[hash_rows[i], 0],
                count[i],
                ndim,
                ndim,
                itemsize,
                scratch.strides,
            )

    else:
        # Case 4 (general, most common case): Non-object slab, at least C-contiguous
        # along the innermost axis. The other axes may be strided (e.g. stepped or
        # transposed slabs): chunks are then hashed one trailing-contiguous hyperplane
        # at a time. Each chunk is a slice along axis 0; byte offset = start * stride0.
        stride0 = src.strides[0]

        for i in range(nchunks):
            offset = stride0 * src_start[i]
            # Count the trailing axes -1..-n that form one C-contiguous run. This must
            # be tested per chunk: edge chunks with trimmed counts may be non-contiguous
            # even on a C-contiguous slab.
            n_contig = 1
            for j in range(ndim - 2, -1, -1):
                if src.strides[j] != src.strides[j + 1] * count[i, j + 1]:
                    break
                n_contig += 1

            _hash_chunk_from_ptr(
                hasher,
                <const unsigned char*>src.data + offset,
                &hash_table[hash_rows[i], 0],
                count[i],
                ndim,
                n_contig,
                itemsize,
                src.strides,
            )


cdef int _hash_shape(_Sha256 hasher, hsize_t* shape, int ndim) except -1:
    """Hash `str(tuple(shape))`.

    This is done with C snprintf to avoid any Python interaction.
    This is inefficient and convoluted but is necessary to generate
    hashes with DATA_VERSION=4.
    """
    cdef char* shape_buf = hasher.shape_str
    cdef int i
    cdef int nchars = 0
    cdef const char* fmt

    if ndim == hasher.ndim and memcmp(shape, hasher.shape, ndim * sizeof(hsize_t)) == 0:
        return hasher.update(shape_buf, hasher.shape_len)

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
            sizeof(hasher.shape_str) - nchars,
            fmt,
            shape[i],
        )

    memcpy(hasher.shape, shape, ndim * sizeof(hsize_t))
    hasher.ndim = ndim
    hasher.shape_len = nchars
    return hasher.update(shape_buf, nchars)


cdef void _copy_chunk(
    const unsigned char* src_ptr,
    unsigned char* dst,
    hsize_t[::1] shape,
    hsize_t ndim,
    hsize_t itemsize,
    npy_intp* strides,
) noexcept nogil:
    """Deep-copy an edge-trimmed chunk into a C-contiguous scratch buffer.

    nogil, C-friendly equivalent of np.ascontiguousarray().
    """
    cdef hsize_t[NPY_MAXDIMS] outer_idx
    cdef hsize_t outer_total = 1
    cdef hsize_t inner = shape[ndim - 1]
    cdef hsize_t outer, k
    cdef npy_intp offset
    cdef const unsigned char* src_row

    for k in range(ndim - 1):
        outer_idx[k] = 0
        outer_total *= shape[k]

    for outer in range(outer_total):
        offset = 0
        for k in range(ndim - 1):
            offset += outer_idx[k] * strides[k]
        src_row = src_ptr + offset

        for k in range(inner):
            memcpy(dst, src_row + k * strides[ndim - 1], itemsize)
            dst += itemsize

        # Advance the indices of the leading axes
        for k in range(ndim - 2, -1, -1):
            outer_idx[k] += 1
            if outer_idx[k] < shape[k]:
                break
            outer_idx[k] = 0


cdef int _hash_chunk_from_ptr(
    _Sha256 hasher,
    const unsigned char* data_ptr,
    uint64_t* out,
    hsize_t[::1] shape,
    hsize_t ndim,
    hsize_t n_contig,
    hsize_t itemsize,
    npy_intp* strides,
) except -1:
    """Hash a single chunk given a raw pointer and strides.
    The chunk must be not object dtype, not StringDType, not broadcasted, and
    C-contiguous at least along the innermost axis.
    """
    cdef hsize_t[NPY_MAXDIMS] outer_idx
    cdef hsize_t outer_total
    cdef hsize_t n_outer
    cdef hsize_t block_size = itemsize
    cdef hsize_t offset
    cdef hsize_t j

    if n_contig == ndim:
        # Single contiguous blob
        for j in range(ndim):
            block_size *= shape[j]
        hasher.update(data_ptr, block_size)
    else:
        # The trailing n_contig axes are one contiguous run; hash the chunk as
        # prod(shape[:n_outer]) contiguous blocks of block_size bytes each,
        # walking the leading axes in C-order.
        n_outer = ndim - n_contig
        outer_total = 1
        for j in range(ndim - 1, n_outer - 1, -1):
            block_size *= shape[j]
        for j in range(n_outer):
            outer_idx[j] = 0
            outer_total *= shape[j]

        for outer in range(outer_total):
            # Compute byte offset for current block
            offset = 0
            for j in range(n_outer):
                offset += outer_idx[j] * strides[j]

            hasher.update(data_ptr + offset, block_size)

            # Advance the indices of the leading axes
            for j in range(n_outer - 1, -1, -1):
                outer_idx[j] += 1
                if outer_idx[j] < shape[j]:
                    break
                outer_idx[j] = 0

    _hash_shape(hasher, &shape[0], ndim)
    return hasher.finish(out)


cdef void _hash_object_chunk(
    _Sha256 hasher,
    np.ndarray data,
    uint64_t* out,
    hsize_t[::1] shape,
    hsize_t ndim,
) except *:
    """Hash a single chunk with object dtype or StringDType (variable-width strings).

    Object data forces us to hold the GIL throughout.
    """
    cdef bytes value_b
    cdef uint64_t nbytes_he
    cdef uint64_t nbytes_le

    # Ensure StringDType and object strings produce the same hash
    # TODO speed this up with native C NumPy APIs
    if data.dtype.kind == "T":
        data = data.astype(object)

    for value in data.flat:
        if isinstance(value, str):
            value_b = value.encode("utf-8")
        elif isinstance(value, bytes):
            # Explicitly cast np.bytes_ to its parent class bytes. Cython's
            # auto-coercion rejects subclasses and would raise TypeError without an
            # explicit cast.
            value_b = <bytes>value
        else:
            raise ValueError(f"Object array contains unsupported type={type(value)}")

        nbytes_he = len(value_b)  # host-endian
        # On x86 and ARM, this is a no-op. On PowerPC, swap endianness
        nbytes_le = htole64(nbytes_he)  # little-endian
        hasher.update(&nbytes_le, 8)
        hasher.update(<const char*>value_b, nbytes_he)

    _hash_shape(hasher, <hsize_t*>data.shape, data.ndim)
    hasher.finish(out)
