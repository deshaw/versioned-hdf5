import sys
from functools import lru_cache

import cython
from h5py import h5s
from h5py._hl.base import phil
from ndindex import Slice, Tuple

from libc.stddef cimport ptrdiff_t, size_t
from libc.stdio cimport FILE, fclose
from libcpp.vector cimport vector


cdef FILE* fmemopen(void* buf, size_t size, const char* mode):
    raise NotImplementedError("fmemopen is not available on Windows")
if sys.platform != "win32":
    from posix.stdio cimport fmemopen

ctypedef ptrdiff_t ssize_t

cdef extern from "hdf5.h":
    # HDF5 types copied from h5py/api_types_hdf5.pxd
    ctypedef long int hid_t
    ctypedef int hbool_t
    ctypedef int herr_t
    ctypedef int htri_t
    ctypedef long long hsize_t
    ctypedef signed long long hssize_t
    ctypedef signed long long haddr_t
    ctypedef long int off_t

    cdef hid_t H5E_DEFAULT = 0
    cdef herr_t H5Eprint(hid_t stack_id, FILE* stream) nogil

    # virtual Dataset functions
    cdef hid_t H5Pget_virtual_vspace(hid_t dcpl_id, size_t index)
    cdef hid_t H5Pget_virtual_srcspace(hid_t dcpl_id, size_t index)

    ctypedef enum H5S_sel_type:
        H5S_SEL_ERROR = -1,  # Error
        H5S_SEL_NONE = 0,  # Nothing selected
        H5S_SEL_POINTS = 1,  # Sequence of points selected
        H5S_SEL_HYPERSLABS = 2,  # "New-style" hyperslab selection defined
        H5S_SEL_ALL = 3,  # Entire extent selected
        H5S_SEL_N = 4  # THIS MUST BE LAST

    cdef H5S_sel_type H5Sget_select_type(
        hid_t space_id
    ) except H5S_sel_type.H5S_SEL_ERROR
    cdef int H5Sget_simple_extent_ndims(hid_t space_id)
    cdef htri_t H5Sget_regular_hyperslab(
        hid_t spaceid,
        hsize_t* start,
        hsize_t* stride,
        hsize_t* count,
        hsize_t* block,
    )


def spaceid_to_slice(space) -> Tuple:
    """
    Convert an h5py spaceid object into an ndindex index

    The resulting index is always a Tuple index.
    """

    sel_type = space.get_select_type()

    if sel_type == h5s.SEL_ALL:
        return Tuple()
    elif sel_type == h5s.SEL_HYPERSLABS:
        slices = []
        starts, strides, counts, blocks = space.get_regular_hyperslab()
        for start, stride, count, block in zip(starts, strides, counts, blocks):
            slices.append(hyperslab_to_slice(start, stride, count, block))
        return Tuple(*slices)
    elif sel_type == h5s.SEL_NONE:
        return Tuple(
            Slice(0, 0),
        )
    else:
        raise NotImplementedError("Point selections are not yet supported")


@lru_cache(2048)
def hyperslab_to_slice(start, stride, count, block):
    if not (block == 1 or count == 1):
        raise NotImplementedError("Nontrivial blocks are not yet supported")
    end = start + (stride * (count - 1) + 1) * block
    stride = stride if block == 1 else 1
    return Slice(start, end, stride)


cdef _spaceid_to_slice(space_id: hid_t):
    """
    Helper function to read the data for `space_id` selection and
    convert it to a Tuple of slices.
    """
    sel_type: H5S_sel_type = H5Sget_select_type(space_id)

    if sel_type == H5S_sel_type.H5S_SEL_ALL:
        return Tuple()
    elif sel_type == H5S_sel_type.H5S_SEL_HYPERSLABS:
        slices: list = []

        rank: cython.int = H5Sget_simple_extent_ndims(space_id)
        if rank < 0:
            raise ValueError("Cannot determine rank of selection.")
        start_array: vector[hsize_t] = vector[hsize_t](rank)
        stride_array: vector[hsize_t] = vector[hsize_t](rank)
        count_array: vector[hsize_t] = vector[hsize_t](rank)
        block_array: vector[hsize_t] = vector[hsize_t](rank)

        ret: htri_t = H5Sget_regular_hyperslab(
            space_id,
            start_array.data(),
            stride_array.data(),
            count_array.data(),
            block_array.data(),
        )
        if ret < 0:
            raise HDF5Error()

        i: cython.int
        start: hsize_t
        end: hsize_t
        stride: hsize_t
        count: hsize_t
        block: hsize_t
        for i in range(rank):
            start = start_array[i]
            stride = stride_array[i]
            count = count_array[i]
            block = block_array[i]
            if not (block == 1 or count == 1):
                raise NotImplementedError("Nontrivial blocks are not yet supported")
            end = start + (stride * (count - 1) + 1) * block
            stride = stride if block == 1 else 1
            slices.append(Slice(start, end, stride))

        return Tuple(*slices)
    elif sel_type == H5S_sel_type.H5S_SEL_NONE:
        return Tuple(
            Slice(0, 0),
        )
    else:
        raise NotImplementedError("Point selections are not yet supported")

cpdef build_data_dict(dcpl, raw_data_name: str):
    """
    Function to build the "data_dict" of a versioned virtual dataset.

    All virtual datasets created by versioned-hdf5 should have chunks in
    exactly one raw dataset `raw_data_name` in the same file.
    This function blindly assumes this is the case.

    :param dcpl: the dataset creation property list of the versioned dataset
    :param raw_data_name: the name of the corresponding raw dataset
    :return: a dictionary mapping the `Tuple` of the virtual dataset chunk
        to a `Slice` in the raw dataset.
    """
    data_dict: dict = {}

    with phil:
        dcpl_id: hid_t = dcpl.id
        virtual_count: size_t = dcpl.get_virtual_count()
        j: size_t

        for j in range(virtual_count):
            vspace_id: hid_t = H5Pget_virtual_vspace(dcpl_id, j)
            if vspace_id == -1:
                raise HDF5Error()
            srcspace_id: hid_t = H5Pget_virtual_srcspace(dcpl_id, j)
            if srcspace_id == -1:
                raise HDF5Error()

            vspace_slice_tuple = _spaceid_to_slice(vspace_id)
            srcspace_slice_tuple = _spaceid_to_slice(srcspace_id)
            # the slice into the raw_data (srcspace_slice_tuple) is only
            # on the first axis
            data_dict[vspace_slice_tuple] = srcspace_slice_tuple.args[0]

    return data_dict


cdef Exception HDF5Error():
    """Generate a RuntimeError with the HDF5 error message.

    This function must be invoked only after a HDF5 function returned an error code.
    """
    if sys.platform == "win32":
        # No fmemopen available
        return RuntimeError("HDF5 error")

    cdef char buf[20000]
    cdef FILE* stream = fmemopen(buf, sizeof(buf), "w")
    if stream == NULL:
        return MemoryError("fmemopen() failed")
    with phil:
        H5Eprint(H5E_DEFAULT, stream)
    fclose(stream)
    msg = buf.decode("utf-8", errors="replace")
    return RuntimeError(msg)
