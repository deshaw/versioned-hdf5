# This file allows cimport'ing the functions and types declared below from other
# Cython modules

from libc.stdint cimport uint64_t

# Centralized definition of hsize_t, in accordance with libhd5:
# https://github.com/HDFGroup/hdf5/blob/6b43197b0817596f47670c6b55d26ff7f86d6bd9/src/H5public.h#L301
#
# versioned_hdf5 uses the same datatype for indexing as libhdf5. Notably, this differs
# from numpy's npy_intp (same as Py_ssize_t, ssize_t, and signed long), which is only 32
# bit wide on 32 bit platforms, to allow indexing datasets with >=2**31 points per axis
# on disk, as long as you don't load them in memory all at once.
#
# Note that hsize_t is unsigned, which can lead to integer underflows.
#
# The definition of hsize_t has changed over time in the libhdf5 headers, as noted
# below.
# C99 dictates that long long is *at least* 64 bit wide, while uint64_t is *exactly* 64
# bit wide. So the two definitions are de facto equivalent.
# The below definition is inconsequential, as it is overridden by whatever version of
# hdf5.h is installed thanks to the 'cdef extern from "hdf5.h"' block.
#
# h5py gets this wildly wrong, as it defines hsize_t as long long, which is signed, and
# confusingly also defines hssize_t as signed long long - which is an alias for
# long long:
# https://github.com/h5py/h5py/blob/eaa9d93cc7620f3e7d8cff6f3a739b7d9834aef7/h5py/api_types_hdf5.pxd#L21-L22

cdef extern from "hdf5.h":
    # ctypedef unsigned long long hsize_t  # hdf5 <=1.12
    ctypedef uint64_t hsize_t  # hdf5 >=1.14

cpdef hsize_t stop2count(hsize_t start, hsize_t stop, hsize_t step) noexcept nogil
cpdef hsize_t count2stop(hsize_t start, hsize_t count, hsize_t step) noexcept nogil
cpdef hsize_t ceil_a_over_b(hsize_t a, hsize_t b) noexcept nogil
cpdef hsize_t smallest_step_after(hsize_t x, hsize_t a, hsize_t m) noexcept nogil
