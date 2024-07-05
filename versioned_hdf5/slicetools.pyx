import itertools
import typing
from functools import lru_cache

from h5py import h5s, h5i
from h5py._hl.base import phil
from ndindex import Slice, Tuple, ndindex, Integer, IntegerArray, BooleanArray, ChunkSize
import numpy as np

import cython
from libc.stddef cimport size_t, ptrdiff_t
ctypedef ptrdiff_t ssize_t
from libc.stdlib cimport malloc, free
from libc.string cimport strlen, strncmp
from libcpp.vector cimport vector

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

    # virtual Dataset functions
    cdef ssize_t H5Pget_virtual_dsetname(hid_t dcpl_id, size_t index, char *name, size_t size)
    cdef ssize_t H5Pget_virtual_filename(hid_t dcpl_id, size_t index, char *name, size_t size)
    cdef hid_t H5Pget_virtual_vspace(hid_t dcpl_id, size_t index)
    cdef hid_t H5Pget_virtual_srcspace(hid_t dcpl_id, size_t index)

    ctypedef enum H5S_sel_type:
        H5S_SEL_ERROR = -1,  #Error
        H5S_SEL_NONE = 0,  #Nothing selected
        H5S_SEL_POINTS = 1,  #Sequence of points selected
        H5S_SEL_HYPERSLABS = 2,  #"New-style" hyperslab selection defined
        H5S_SEL_ALL = 3,  #Entire extent selected
        H5S_SEL_N = 4  #/*THIS MUST BE LAST

    cdef H5S_sel_type H5Sget_select_type(hid_t space_id) except H5S_sel_type.H5S_SEL_ERROR

    cdef int H5Sget_simple_extent_ndims(hid_t space_id)
    cdef htri_t H5Sget_regular_hyperslab(hid_t spaceid, hsize_t* start, hsize_t* stride, hsize_t* count, hsize_t* block)

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
            raise ValueError('Cannot determine rank of selection.')
        start_array: vector[hsize_t] = vector[hsize_t](rank)
        stride_array: vector[hsize_t] = vector[hsize_t](rank)
        count_array: vector[hsize_t] = vector[hsize_t](rank)
        block_array: vector[hsize_t] = vector[hsize_t](rank)

        ret: htri_t = H5Sget_regular_hyperslab(space_id, start_array.data(), stride_array.data(),
                                               count_array.data(), block_array.data())
        if ret < 0:
            raise ValueError('Cannot determine hyperslab selection.')

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
    exactly one raw dataset `raw_data_name` in the same file. This function will
    check that this is the case and return a dictionary mapping the `Tuple` of
    the chunk in the virtual dataset to a `Slice` in the raw dataset.

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
                    if H5Pget_virtual_filename(dcpl_id, j, filename_buf, filename_buf_len) < 0:
                        raise ValueError('Could not get virtual filename')
                    if strncmp(filename_buf, ".", filename_buf_len) != 0:
                        raise ValueError('Virtual dataset filename mismatch, expected "."')

                    if H5Pget_virtual_dsetname(dcpl_id, j, dataset_buf, dataset_buf_len) < 0:
                        raise ValueError('Could not get virtual dsetname')
                    if strncmp(dataset_buf, raw_data_str, dataset_buf_len) != 0:
                        raise ValueError(f'Virtual dataset name mismatch, expected {raw_data_name}')

                    vspace_id: hid_t = H5Pget_virtual_vspace(dcpl_id, j)
                    if vspace_id == -1:
                        raise ValueError('Could not get vspace_id')
                    srcspace_id: hid_t = H5Pget_virtual_srcspace(dcpl_id, j)
                    if srcspace_id == -1:
                        raise ValueError('Could not get srcspace_id')

                    vspace_slice_tuple = _spaceid_to_slice(vspace_id)
                    srcspace_slice_tuple = _spaceid_to_slice(srcspace_id)
                    # the slice into the raw_data (srcspace_slice_tuple) is only on the first axis
                    data_dict[vspace_slice_tuple] = srcspace_slice_tuple.args[0]
            finally:
                free(dataset_buf)
        finally:
            free(filename_buf)

    return data_dict


@cython.cfunc
@cython.returns(cython.Py_ssize_t)
def _ceiling(a: cython.Py_ssize_t, b: cython.Py_ssize_t):
    """
    Returns ceil(a/b)
    """
    return -(-a//b)

@cython.cfunc
@cython.returns(cython.Py_ssize_t)
def _min(a: cython.Py_ssize_t, b: cython.Py_ssize_t):
    return min(a, b)

@cython.cfunc
@cython.returns(cython.Py_ssize_t)
def _smallest(x: cython.Py_ssize_t, a: cython.Py_ssize_t, m: cython.Py_ssize_t):
    """
    Gives the smallest integer >= x that equals a (mod m)

    Assumes x >= 0, m >= 1, and 0 <= a < m.
    """
    n: cython.Py_ssize_t = _ceiling(x - a, m)
    return a + n*m

@cython.ccall
def subindex_chunk_slice(c_start: cython.Py_ssize_t, c_stop: cython.Py_ssize_t,
                         i_start: cython.Py_ssize_t, i_stop: cython.Py_ssize_t, i_step: cython.Py_ssize_t):
    common: cython.Py_ssize_t = i_start % i_step

    lcm: cython.Py_ssize_t = i_step
    start: cython.Py_ssize_t = max(c_start, i_start)

    # Get the smallest lcm multiple of common that is >= start
    start = _smallest(start, common, lcm)
    # Finally, we need to shift start so that it is relative to index
    start = (start - i_start)//i_step

    stop: cython.Py_ssize_t = _ceiling((_min(c_stop, i_stop) - i_start), i_step)
    stop = 0 if stop < 0 else stop

    step: cython.Py_ssize_t = lcm//i_step

    return Slice(start, stop, step)


@cython.ccall
def subindex_slice_chunk(s_start: cython.Py_ssize_t, s_stop: cython.Py_ssize_t, s_step: cython.Py_ssize_t,
                         c_start: cython.Py_ssize_t, c_stop: cython.Py_ssize_t):
    common: cython.Py_ssize_t = s_start % s_step

    lcm: cython.Py_ssize_t = s_step
    start: cython.Py_ssize_t = max(s_start, c_start)

    # Get the smallest lcm multiple of common that is >= start
    start = _smallest(start, common, lcm)
    # Finally, we need to shift start so that it is relative to index
    start = (start - c_start)

    stop: cython.Py_ssize_t = _min(s_stop, c_stop) - c_start
    stop = 0 if stop < 0 else stop

    step: cython.Py_ssize_t = lcm

    return Slice(start, stop, step)

def as_subchunk_map(chunk_size: ChunkSize, idx, shape: tuple) -> typing.Generator[typing.Tuple[
    typing.Tuple[Slice, ...],
    typing.Tuple[slice | np.ndarray, ...],
    typing.Tuple[slice | np.ndarray, ...]]]:
    """
    Computes the chunk selection assignment. In particular, given a `chunk_size`
    it returns triple (chunk_slices, arr_subidxs, chunk_subidxs) such that for a
    chunked Dataset `ds` we can translate selections like

    >> ds[idx]

    into selecting from the individual chunks of `ds` as

    >> arr = np.ndarray(output_shape)
    >> for chunk, arr_idx_raw, index_raw in as_subchunk_map(ds.chunk_size, idx, ds.shape):
    ..     arr[arr_idx_raw] = ds.data_dict[chunk][index_raw]

    Similarly, assignments like

    >> ds[idx] = arr

    can be translated into

    >> for chunk, arr_idx_raw, index_raw in as_subchunk_map(ds.chunk_size, idx, ds.shape):
    ..     ds.data_dict[chunk][index_raw] = arr[arr_idx_raw]

    :param chunk_size: the `ChunkSize` of the Dataset
    :param idx: the "index" to read from / write to the Dataset
    :param shape: the shape of the Dataset
    :return: a generator of `(chunk, arr_idx_raw, index_raw)` tuples
    """
    assert isinstance(chunk_size, ChunkSize)
    if isinstance(idx, Tuple):
        pass
    elif isinstance(idx, tuple):
        idx = Tuple(*idx)
    else:
        idx = Tuple(ndindex(idx))
    assert isinstance(shape, tuple)

    if any(dim < 0 for dim in shape):
        raise ValueError("shape dimensions must be non-negative")

    if len(shape) != len(chunk_size):
        raise ValueError("chunks dimensions must equal the array dimensions")

    if idx.isempty(shape):
        # abort early for empty index
        return

    idx_len: cython.Py_ssize_t = len(idx.args)

    prefix_chunk_size = chunk_size[:idx_len]
    prefix_shape = shape[:idx_len]

    suffix_chunk_size = chunk_size[idx_len:]
    suffix_shape = shape[idx_len:]

    chunk_subindexes = []

    n: int
    i: Slice | IntegerArray | BooleanArray | Integer
    s: int

    chunk_idx: cython.Py_ssize_t
    chunk_start: cython.Py_ssize_t
    chunk_stop: cython.Py_ssize_t

    # Process the prefix of the axes which idx selects on
    for n, i, d in zip(prefix_chunk_size, idx.args, prefix_shape):
        i = i.reduce((d,))

        # Compute chunk_idxs, e.g., chunk_idxs == (2, 4) for chunk sizes (100, 1000)
        # would correspond to chunk (slice(200, 300), slice(4000, 5000)).
        chunk_subindexes_for_axis: list = []
        if isinstance(i, Slice):
            if i.step <= 0:
                raise NotImplementedError(f'Slice step must be positive not {i.step}')

            start: int = i.start
            stop: int = i.stop
            step: int = i.step

            if step > n:
                chunk_idxs = tuple((start + k * step) // n for k in range((stop - start + step - 1) // step))
            else:
                chunk_idxs = tuple(range(start // n, (stop + n - 1) // n))

            for chunk_idx in chunk_idxs:
                chunk_start = chunk_idx * n
                chunk_stop = min((chunk_idx + 1) * n, d)

                chunk_subindexes_for_axis.append((
                    Slice(chunk_start, chunk_stop, 1),
                    (subindex_chunk_slice(chunk_start, chunk_stop,
                                          start, stop, step)
                     .raw),
                    (subindex_slice_chunk(start, stop, step,
                                          chunk_start, chunk_stop)
                     # Make chunk_slice canonical.
                     .reduce(d)
                     .raw),
                ))
        elif isinstance(i, IntegerArray):
            assert i.ndim == 1
            chunk_idxs = np.unique(i.array // n)

            for chunk_idx in chunk_idxs:
                chunk_start = chunk_idx * n
                chunk_stop = min((chunk_idx + 1) * n, d)
                i_chunk_mask = (chunk_start <= i.array) & (i.array < chunk_stop)
                chunk_subindexes_for_axis.append((
                    Slice(chunk_start, chunk_stop, 1),
                    i_chunk_mask,
                    i.array[i_chunk_mask] - chunk_start,
                ))
        elif isinstance(i, BooleanArray):
            if i.ndim != 1:
                raise NotImplementedError('boolean mask index must be 1-dimensional')
            if i.shape != (d,):
                raise IndexError(f'boolean index did not match indexed array; dimension is {d}, '
                                 f'but corresponding boolean dimension is {i.shape[0]}')

            # pad i.array to be a multiple of n and group into chunks
            mask = np.pad(i.array, (0, n - (d % n)), 'constant', constant_values=(False,))
            mask = mask.reshape((mask.shape[0] // n, n))

            # count how many elements were selected in each chunk
            chunk_selected_counts = np.sum(mask, axis=1, dtype=np.intp)

            # compute offsets based on selected counts which will be used to build the masks for each chunk
            chunk_selected_offsets = np.zeros(len(chunk_selected_counts) + 1, dtype=np.intp)
            chunk_selected_offsets[1:] = np.cumsum(chunk_selected_counts)

            # chunk_idxs for the chunks which are not empty
            chunk_idxs = np.flatnonzero(chunk_selected_counts)

            for chunk_idx in chunk_idxs:
                chunk_start = chunk_idx * n
                chunk_stop = min((chunk_idx + 1) * n, d)
                chunk_subindexes_for_axis.append((
                    Slice(chunk_start, chunk_stop, 1),
                    np.concatenate([np.zeros(chunk_selected_offsets[chunk_idx], dtype=bool),
                                    np.ones(chunk_selected_counts[chunk_idx], dtype=bool),
                                    np.zeros(chunk_selected_offsets[-1] - chunk_selected_offsets[chunk_idx + 1], dtype=bool)]),
                    np.flatnonzero(i.array[chunk_start:chunk_stop]),
                ))
        elif isinstance(i, Integer):
            chunk_idx = i.raw // n
            chunk_start = chunk_idx * n
            chunk_stop = min((chunk_idx + 1) * n, d)
            chunk_subindexes_for_axis.append((
                Slice(chunk_start, chunk_stop, 1),
                (),
                i.raw - chunk_start,
            ))
        else:
            raise NotImplementedError(f"index type {type(i)} not supported")

        chunk_subindexes.append(chunk_subindexes_for_axis)

    # Handle the remaining suffix axes on which we did not select, we still need to break
    # them up into chunks.
    for n, d in zip(suffix_chunk_size, suffix_shape):
        chunk_idxs = tuple(range((d + n - 1) // n))
        chunk_slices = tuple(Slice(chunk_idx * n, min((chunk_idx + 1) * n, d), 1)
                             for chunk_idx in chunk_idxs)
        chunk_subindexes.append([(chunk_slice, chunk_slice.raw, ())
                                 for chunk_slice in chunk_slices])

    # Now combine the chunk_slices and subindexes for each dimension into tuples
    # across all dimensions.
    for p in itertools.product(*chunk_subindexes):
        chunk_slices, arr_subidxs, chunk_subidxs = zip(*p)

        # skip dimensions which were sliced away
        arr_subidxs = tuple(arr_subidx for arr_subidx in arr_subidxs
                            if not isinstance(arr_subidx, tuple) or arr_subidx != ())

        # skip suffix dimensions
        chunk_subidxs = chunk_subidxs[:idx_len]

        yield Tuple(*chunk_slices), arr_subidxs, chunk_subidxs
