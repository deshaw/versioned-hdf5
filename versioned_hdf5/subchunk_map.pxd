# This file allows cimport'ing the functions declared below from other Cython modules

from versioned_hdf5.cytools cimport hsize_t


cdef class IndexChunkMapper:
    cdef readonly hsize_t[:] chunk_indices
    cdef readonly hsize_t chunk_size
    cdef readonly hsize_t dset_size
    cdef readonly hsize_t n_chunks
    cdef readonly hsize_t last_chunk_size

    cpdef tuple[object, object | None] read_many_slices_params(self)

    cpdef object chunks_indexer(self)
    cpdef object whole_chunks_idxidx(self)

    # Private methods. Cython complains if we don't export _everything_.
    cdef tuple[hsize_t, hsize_t] _chunk_start_stop(
        self, hsize_t chunk_idx
    ) noexcept nogil


cpdef hsize_t[:, :, :] read_many_slices_params_nd(
    transfer_type,  # TransferType Python enum
    list[IndexChunkMapper] mappers,
    hsize_t[:, :] chunk_idxidx,
    hsize_t[:] src_slab_offsets,
    hsize_t[:] dst_slab_offsets,
)


# At the moment of writing, Cython enums are unusable in pure python mode:
# https://github.com/cython/cython/issues/4252
# The below is for user reference only and not actually used in the code.

cdef enum ReadManySlicesColumn:
    """Axis 1 of the return value of IndexChunkMapper.read_many_slices_params"""
    chunk_sub_start = 0
    value_sub_start = 1
    count = 2
    chunk_sub_stride = 3
    value_sub_stride = 4


cdef enum ReadManySlicesNDColumn:
    """Axis 1 of the return value of read_many_slices_params_nd"""
    src_start = 0
    dst_start = 1
    count_ = 2  # Two enums can't define the same label
    src_stride = 3
    dst_stride = 4
