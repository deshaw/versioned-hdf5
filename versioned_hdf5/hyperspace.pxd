# This file allows cimport'ing the functions declared below from other Cython modules

from cython cimport Py_ssize_t


cpdef Py_ssize_t[:] empty_view(Py_ssize_t n)
cpdef Py_ssize_t[:] view_from_tuple(tuple[int, ...] t)
cpdef Py_ssize_t[:, :] fill_hyperspace(
    Py_ssize_t[:, :] obstacles, tuple[int, ...] shape
)
