# This file allows cimport'ing the functions declared below from other Cython modules

from libc.stdint cimport uint64_t

cimport numpy as np

from versioned_hdf5.cytools cimport hsize_t

cpdef void hash_slab(
    np.ndarray src,
    uint64_t[:, ::1] hash_table,
    hsize_t[::1] hash_rows,
    hsize_t[::1] src_start,
    hsize_t[:, ::1] count,
) except *