# This file allows cimport'ing the functions declared below from other Cython modules

from cython cimport Py_ssize_t


cdef class IndexChunkMapper:
    cdef readonly Py_ssize_t[:] chunk_indices
    cdef readonly Py_ssize_t chunk_size
    cdef readonly Py_ssize_t dset_size

    cpdef tuple[object, object] chunk_submap(
        self,
        Py_ssize_t chunk_start_idx,
        Py_ssize_t chunk_stop_idx,
        bint shift,
    )

    # Private methods. Cython complains if we don't export _everything_.
    cdef tuple[Py_ssize_t, Py_ssize_t] _chunk_start_stop(
        self, Py_ssize_t a, Py_ssize_t b
    )
    cdef tuple[object, object, object] chunk_submap_compat(self, Py_ssize_t chunk_idx)

# Utility functions
cpdef Py_ssize_t ceil_a_over_b(Py_ssize_t a, Py_ssize_t b)
