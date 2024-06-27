from functools import lru_cache

from ndindex import Slice, Tuple
from h5py import h5d, h5s
from h5py._hl.base import phil

import cython
from libc.stddef cimport size_t, ptrdiff_t
ctypedef ptrdiff_t ssize_t
from libc.stdlib cimport malloc, free
from libc.string cimport strlen, strncmp
from libcpp.vector cimport vector

ctypedef enum H5S_sel_type:
    H5S_SEL_ERROR = -1,  #Error
    H5S_SEL_NONE = 0,  #Nothing selected
    H5S_SEL_POINTS = 1,  #Sequence of points selected
    H5S_SEL_HYPERSLABS = 2,  #"New-style" hyperslab selection defined
    H5S_SEL_ALL = 3,  #Entire extent selected
    H5S_SEL_N = 4  #/*THIS MUST BE LAST

cdef extern from "hdf5.h":
    # HDF5 types
    ctypedef long int hid_t
    ctypedef int herr_t
    ctypedef int htri_t
    ctypedef long long hsize_t

    # virtual Dataset functions
    cdef herr_t H5Pget_virtual_count(hid_t dcpl_id, size_t *count) except <herr_t>-1
    cdef ssize_t H5Pget_virtual_dsetname(hid_t dcpl_id, size_t index, char *name, size_t size) except <ssize_t>-1
    cdef ssize_t H5Pget_virtual_filename(hid_t dcpl_id, size_t index, char *name, size_t size) except <ssize_t>-1
    cdef hid_t H5Pget_virtual_vspace(hid_t dcpl_id, size_t index) except <hid_t>-1
    cdef hid_t H5Pget_virtual_srcspace(hid_t dcpl_id, size_t index) except <hid_t>-1

    # TODO: this function actually returns an H5S_sel_type enum, but compilation fails
    #       when that's specified and we have an "except" clause. This looks like a bug in Cython?
    #       https://github.com/cython/cython/issues/6275
    # cdef H5S_sel_type H5Sget_select_type(hid_t space_id) except <H5S_sel_type> -1
    cdef int H5Sget_select_type(hid_t space_id) except <int>-1

    cdef int H5Sget_simple_extent_ndims(hid_t space_id) except <int> -1
    cdef htri_t H5Sget_regular_hyperslab(hid_t spaceid, hsize_t* start, hsize_t* stride, hsize_t* count, hsize_t* block) except <htri_t>-1

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
    sel_type: H5S_sel_type = <H5S_sel_type> H5Sget_select_type(space_id)

    if sel_type == H5S_sel_type.H5S_SEL_ALL:
        return Tuple()
    elif sel_type == H5S_sel_type.H5S_SEL_HYPERSLABS:
        slices: list = []

        rank: cython.int = H5Sget_simple_extent_ndims(space_id)
        start_array: vector[hsize_t] = vector[hsize_t](rank)
        stride_array: vector[hsize_t] = vector[hsize_t](rank)
        count_array: vector[hsize_t] = vector[hsize_t](rank)
        block_array: vector[hsize_t] = vector[hsize_t](rank)

        H5Sget_regular_hyperslab(space_id, start_array.data(), stride_array.data(),
                                 count_array.data(), block_array.data())
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

cpdef build_data_dict(dcpl, shape: tuple, chunks: tuple, raw_data_name: str):
    data_dict: dict = {}

    is_virtual: bool = dcpl.get_layout() == h5d.VIRTUAL

    if not is_virtual:
        # A dataset created with only a fillvalue will be nonvirtual,
        # since create_virtual_dataset makes a nonvirtual dataset when
        # there are no virtual sources.
        data_dict = {}
    # Same as dataset.get_virtual_sources
    elif 0 in shape:
        # Work around https://github.com/h5py/h5py/issues/1660
        empty_idx = Tuple().expand(shape)
        data_dict = {empty_idx: Slice()}
    else:
        data_dict = {}

        with phil:
            dcpl_id: hid_t = dcpl.id

            virtual_count: size_t = dcpl.get_virtual_count()
            j: size_t

            raw_data_name_bytes: bytes = raw_data_name.encode('utf8')
            # this a reference to the internal buffer of raw_data_name, do not free!
            raw_data_str: cython.p_char = raw_data_name_bytes

            filename_buf_len: ssize_t = 2
            filename_buf: cython.p_char = <char *>malloc(filename_buf_len)
            if not filename_buf:
                raise MemoryError('could not allocate filename_buf')

            try:
                dataset_buf_len: ssize_t = strlen(raw_data_str) + 1
                dataset_buf: cython.p_char = <char *>malloc(dataset_buf_len)
                if not dataset_buf:
                    raise MemoryError('could not allocate dataset_buf')

                try:
                    for j in range(virtual_count):
                        H5Pget_virtual_filename(dcpl_id, j, filename_buf, filename_buf_len)
                        if strncmp(filename_buf, ".", filename_buf_len) != 0:
                            raise ValueError('Virtual dataset filename mismatch, expected "."')

                        H5Pget_virtual_dsetname(dcpl_id, j, dataset_buf, dataset_buf_len)
                        if strncmp(dataset_buf, raw_data_str, dataset_buf_len) != 0:
                            raise ValueError(f'Virtual dataset name mismatch, expected {raw_data_name}')

                        vspace_id: hid_t = H5Pget_virtual_vspace(dcpl_id, j)
                        srcspace_id: hid_t = H5Pget_virtual_srcspace(dcpl_id, j)

                        vspace_slice_tuple = _spaceid_to_slice(vspace_id)
                        srcspace_slice_tuple = _spaceid_to_slice(srcspace_id)
                        # the slice into the raw_data (srcspace_slice_tuple) is only on the first axis
                        data_dict[vspace_slice_tuple] = srcspace_slice_tuple.args[0]
                finally:
                    free(dataset_buf)
            finally:
                free(filename_buf)

    return data_dict