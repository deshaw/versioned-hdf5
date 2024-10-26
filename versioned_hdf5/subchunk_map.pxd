# This file allows cimport'ing the functions declared below from other Cython modules

from .cytools cimport hsize_t


cdef class IndexChunkMapper:
    cdef readonly hsize_t[:] chunk_indices
    cdef readonly hsize_t chunk_size
    cdef readonly hsize_t dset_size

    cpdef tuple[object, object, object] chunk_submap(self, hsize_t chunk_idx)

    # Private methods. Cython complains if we don't export _everything_.
    cdef tuple[hsize_t, hsize_t] _chunk_start_stop(
        self, hsize_t chunk_idx
    ) noexcept nogil
