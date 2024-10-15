# cython: linetrace=True
import sys
from functools import lru_cache

import cython
import h5py

cimport numpy as np

import numpy as np
from cython import void
from h5py import h5s
from h5py._hl.base import phil
from h5py.h5t import py_create
from ndindex import Slice, Tuple
from numpy.typing import ArrayLike

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
    ctypedef long long hsize_t  # as per C99, uint64 or wider
    ctypedef signed long long hssize_t  # as per C99, int64 or wider
    ctypedef signed long long haddr_t
    ctypedef long int off_t

    cdef herr_t H5Dread(
        hid_t dset_id,
        hid_t mem_type_id,
        hid_t mem_space_id,
        hid_t file_space_id,
        hid_t dxpl_id,
        void* buf,
    ) nogil

    cdef hid_t H5E_DEFAULT = 0
    cdef herr_t H5Eprint(hid_t stack_id, FILE* stream) nogil

    cdef hid_t H5I_INVALID_HID = -1

    cdef hid_t H5P_DEFAULT = 0
    # virtual Dataset functions
    cdef hid_t H5Pget_virtual_vspace(hid_t dcpl_id, size_t index) nogil
    cdef hid_t H5Pget_virtual_srcspace(hid_t dcpl_id, size_t index) nogil

    ctypedef enum H5S_sel_type:
        H5S_SEL_ERROR = -1,  # Error
        H5S_SEL_NONE = 0,  # Nothing selected
        H5S_SEL_POINTS = 1,  # Sequence of points selected
        H5S_SEL_HYPERSLABS = 2,  # "New-style" hyperslab selection defined
        H5S_SEL_ALL = 3,  # Entire extent selected
        H5S_SEL_N = 4  # THIS MUST BE LAST

    ctypedef enum H5S_seloper_t:
        H5S_SELECT_NOOP = -1,
        H5S_SELECT_SET = 0,
        H5S_SELECT_OR,
        H5S_SELECT_AND,
        H5S_SELECT_XOR,
        H5S_SELECT_NOTB,
        H5S_SELECT_NOTA,
        H5S_SELECT_APPEND,
        H5S_SELECT_PREPEND,
        H5S_SELECT_INVALID    # Must be the last one

    cdef H5S_sel_type H5Sget_select_type(
        hid_t space_id
    ) except H5S_sel_type.H5S_SEL_ERROR nogil
    cdef herr_t H5Sclose(hid_t space_id) nogil
    cdef int H5Sget_simple_extent_ndims(hid_t space_id) nogil
    cdef hid_t H5Screate_simple	(
        int rank,
        const hsize_t* dims,
        const hsize_t* maxdims,
    ) nogil
    cdef htri_t H5Sget_regular_hyperslab(
        hid_t spaceid,
        hsize_t* start,
        hsize_t* stride,
        hsize_t* count,
        hsize_t* block,
    ) nogil
    cdef herr_t H5Sselect_hyperslab(
        hid_t space_id,
        H5S_seloper_t op,
        const hsize_t* start,
        const hsize_t* stride,
        const hsize_t* count,
        const hsize_t* block,
    ) nogil


np.import_array()

# Numpy equivalents of Cython types
np_hsize_t = np.ulonglong
np_hssize_t = np.longlong
np_haddr_t = np.longlong

NP_GE_200 = np.lib.NumpyVersion(np.__version__) >= "2.0.0"


def spaceid_to_slice(space: h5s.SpaceID) -> Tuple:
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


cpdef _spaceid_to_slice(space_id: hid_t):
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
    elif sel_type == H5S_sel_type.H5S_SEL_POINTS:
        raise NotImplementedError("Point selections are not yet supported")
    else:
        raise HDF5Error()


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


cpdef void read_many_slices(
    src: np.ndarray | h5py.Dataset,
    np.ndarray dst,
    src_start: ArrayLike,
    dst_start: ArrayLike,
    count: ArrayLike,
    src_stride: ArrayLike | None = None,
    dst_stride: ArrayLike | None = None,
    fast: bool | None = None,
):
    """Transfer multiple n-dimensional slices of data from src to dst::

        dst[dst_idx0] = src[src_idx0]
        dst[dst_idx1] = src[src_idx1]
        ...

    where all indices are tuples of slices.

    Parameters
    ----------
    src: np.ndarray | h5py.Dataset
        The source data to read from
    dst: np.ndarray
        The destination array to write to.
        It must be C-contiguous and of the same dtype and dimensionality as src.
    src_start: array-like
        The starting coordinates, a.k.a. offsets, of the slices in src.
        It must be a 2D array or array-like (e.g. list of lists) with the same number of
        rows as there are selections and as many columns as there are dimensions in src.
    dst_start: array-like
        The starting coordinates of the slices in dst.
        Same format as src_start.
    count: array-like
        The number of elements to read and write in each dimension.
        Same format as src_start.
    src_stride: array-like, optional
        The stride, a.k.a. step, of the slices when reading from src, in points.
        Note that this differs from numpy.ndarray.strides, which is in bytes.
        Same format as src_start. If omitted, default to 1.
    dst_stride: array-like, optional
        The stride of the slices when writing to dst.
        Same format as src_start. If omitted, default to 1.
    fast: bool, optional
        If True, use the hdf5 C API directly to read the data if possible. This is when
        src is a h5py dataset with a simple data type (see h5py.Dataset._fast_read_ok).
        If False, use pure Python numpy syntax to slice src and dst.
        If omitted, determine automatically. It's recommended to omit this flag unless
        you're running a unit test or benchmark.

    For example, if src and dst are 2D and strides are omitted or always 1,
    the areas of data to be copied will be::

        dst[
            dst_start[i, 0]:dst_start[i, 0] + count[i, 0],
            dst_start[i, 1]:dst_start[i, 1] + count[i, 1],
        ] = src[
            src_start[i, 0]:src_start[i, 0] + count[i, 0],
            src_start[i, 1]:src_start[i, 1] + count[i, 1],
        ]

    where i is the row of each of the coordinates arrays.

    Coordinates arrays can alternatively be 1D arrays, with one point per dimension;
    they will be broadcasted against the other arrays.
    For example, you may use a 1D array for count if you want all selected areas to have
    the same shape.

    Scalar selections
    -----------------
    If dst has less dimensions than src, you should present a reshaped view of dst
    to this function.

    e.g. to execute::

        dst[2:3, 4:6] = src[6:7, 8, 9:11]

    where dst.ndim == 2 and src.ndim == 3, you should call::

        read_many_slices(
            src,
            dst[:, None, :],
            src_start=[[6, 8, 9]],
            dst_start=[[2, 0, 4]],
            count=[[1, 1, 2]],
        )

    Note that the same hack is not possible when src is a h5py.Dataset and has less
    dimensions than dst, as src[:, None, :] would load the whole dataset into memory.

    TODO support src.ndim < dst.ndim. It would take some extra logic
    and an extra parameter to inform where the extra dimensions are.

    Performance notes
    -----------------
    When reading from h5py, this function calls::

        H5Sselect_hyperslab(file_id, H5S_SELECT_SET, ...)
        H5Sselect_hyperslab(mem_id, H5S_SELECT_SET, ...)
        H5DRead(...)

    once per input row; see _read_many_slices_fast.

    This is much faster than calling H5Sselect_hyperslab(..., H5S_SELECT_OR, ...) once
    per input row followed by a single call to H5DRead, a.k.a. hyperslab fusion,
    due to a known issue in libhdf5:
    https://forum.hdfgroup.org/t/union-of-non-consecutive-hyperslabs-is-very-slow/5062

    The downside of this approach is that if your selections access the same chunk more
    than once, they have to be contiguous or else the chunk may fall out of the
    internal cache of libhdf5. This can also happen with contiguous selections that span
    multiple chunks.
    """
    # Begin input validation and preprocessing

    ndim = dst.ndim
    if ndim < 1:
        raise ValueError("must operate on at least one dimension")
    if dst.dtype != src.dtype or ndim != src.ndim or not dst.flags.writeable:
        raise ValueError(
            "dst must be a writeable numpy array of the same dtype and dimensionality "
            "as src"
        )
    if dst.size == 0 or not all(src.shape):
        return

    cdef bint bfast = False
    if fast is not False:
        if isinstance(src, h5py.Dataset):
            with phil:
                bfast = src._fast_read_ok
        if fast and not bfast:
            raise ValueError("fast transfer is not possible with this source")

    if bfast and not dst.flags.c_contiguous:
        # TODO we could support non-C-contiguous arrays by doing
        # arithmetics with starts and strides
        raise NotImplementedError("dst must be C-contiguous for fast mode")

    src_start = _preproc_many_slices_idx(src_start, ndim, bfast)
    dst_start = _preproc_many_slices_idx(dst_start, ndim, bfast)
    count = _preproc_many_slices_idx(count, ndim, bfast)

    if src_stride is None:
        src_stride = np.ones(ndim, dtype=np_hsize_t)
    else:
        src_stride = _preproc_many_slices_idx(src_stride, ndim, bfast)
    if dst_stride is None:
        dst_stride = np.ones(ndim, dtype=np_hsize_t)
    else:
        dst_stride = _preproc_many_slices_idx(dst_stride, ndim, bfast)

    for arr in (src_start, dst_start, count, src_stride, dst_stride):
        if arr.size == 0:
            return

    # dummy makes sure to coerce into the correct number of dimensions when
    # all indices are 1D arrays.
    # This raises if the arrays have mismatched number of rows or dimensions, or if
    # there's a wrong number of columns...
    dummy = np.empty((1, ndim))
    _, src_start, dst_start, count, src_stride, dst_stride = np.broadcast_arrays(
        dummy, src_start, dst_start, count, src_stride, dst_stride
    )
    # ...*unless* ndim=1, in which case we need to test explicitly
    if src_start.shape[1] != ndim:
        raise ValueError("Coordinates arrays must have as many columns as src.ndim")

    cdef np.ndarray src_shape = np.array(src.shape, dtype=np_hsize_t)
    cdef np.npy_intp[1] ndim_ptr = {ndim}
    cdef np.ndarray dst_shape = np.PyArray_SimpleNewFromData(
        1, ndim_ptr, np.NPY_INTP, <void*>dst.shape
    ).astype(np_hsize_t)  # On 32 bit platforms, sizeof(hsize_t) == 8; sizeof(int) == 4

    clipped_count = _clip_count(
        src_shape,
        dst_shape,
        src_start,
        dst_start,
        count,
        src_stride,
        dst_stride,
    )

    # End of input validation and preprocessing

    if bfast:
        _read_many_slices_fast(
            src,
            dst,
            <hsize_t*>dst_shape.data,
            src_start,
            dst_start,
            clipped_count,
            src_stride,
            dst_stride,
        )
    else:
        _read_many_slices_slow(
            src,
            dst,
            src_start,
            dst_start,
            clipped_count,
            src_stride,
            dst_stride,
        )


cdef np.ndarray _preproc_many_slices_idx(obj: ArrayLike, hsize_t ndim, bint fast):
    """Helper of read_many_slices. Ensure that obj is a numpy array of hsize_t with
    either 1 or 2 dimensions, and that it's C-contiguous at least along the
    innermost dimension.
    """
    # np.asarray with unsigned dtype raises in numpy>=2 if presented with a Python list
    # containing negative numbers, but can quietly cause an integer underflow if arr is
    # already a numpy array with signed dtype and negative numbers in it.
    # TODO https://github.com/numpy/numpy/issues/25396
    if not NP_GE_200:
        obj = np.asarray(obj)
    if isinstance(obj, np.ndarray) and obj.dtype.kind != "u" and (obj < 0).any():
        raise OverflowError("index out of bounds for uint64")
    cdef np.ndarray arr = np.asarray(obj, dtype=np_hsize_t)

    if arr.ndim not in (1, 2):
        raise ValueError("Coordinates arrays must have 1 or 2 dimensions")

    # The array must be contiguous along the columns in order to be passed to
    # H5Sselect_hyperslab. So a view like a[:2] is OK, but a[::2] must be made
    # contiguous first. Note that, to save a bit of time, this deep-copy happens before
    # we broadcast 1D arrays to add rows.
    if fast and arr.strides[arr.ndim - 1] != sizeof(hsize_t):
        return np.ascontiguousarray(arr)
    else:
        return arr


@cython.cdivision(True)
cpdef hsize_t stop2count(hsize_t start, hsize_t stop, hsize_t step) noexcept nogil:
    """Given a start:stop:step slice or range, return the number of elements yielded.

    This is functionally identical to::

        len(range(start, stop, step))

    Doesn't assume that stop >= start. Assumes that step >= 1.
    """
    # Note that hsize_t is unsigned so stop - start could underflow.
    if stop <= start:
        return 0
    return (stop - start - 1) // step + 1


cpdef hsize_t count2stop(hsize_t start, hsize_t count, hsize_t step) noexcept nogil:
    """Inverse of stop2count.

    When count == 0 or when step>1, multiple stops can yield the same count.
    This function returns the smallest stop >= start.
    """
    if count == 0:
        return start
    return start + (count - 1) * step + 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cdef hsize_t[:, :] _clip_count(
    const hsize_t[:] src_shape,
    const hsize_t[:] dst_shape,
    const hsize_t[:, :] src_start,
    const hsize_t[:, :] dst_start,
    const hsize_t[:, :] count,
    const hsize_t[:, :] src_stride,
    const hsize_t[:, :] dst_stride,
):
    """Helper of read_many_slices. Clip count to prevent writing out of bounds.
    This function also neuters src_start beyond src.shape or dst_start beyond dst.shape.
    Finally, it also ensures that strides are >= 1.

    TODO refactor as a @cython.ufunc.
    Cython ufuncs don't support hsize_t (long long) at the time of writing.
    """
    nslices, ndim = count.shape[:2]
    cdef hsize_t[:, :] out = np.empty((nslices, ndim), dtype=np_hsize_t)

    with nogil:
        for i in range(nslices):
            for j in range(ndim):
                if src_stride[i, j] == 0 or dst_stride[i, j] == 0:
                    raise ValueError("Strides must be strictly greater than zero")

                out[i, j] = min(
                    count[i, j],
                    stop2count(src_start[i, j], src_shape[j], src_stride[i, j]),
                    stop2count(dst_start[i, j], dst_shape[j], dst_stride[i, j]),
                )

    return out


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.infer_types(True)
cdef void _read_many_slices_fast (
    src: h5py.Dataset,
    np.ndarray dst,
    const hsize_t* dst_shape,
    const hsize_t[:, :] src_start,
    const hsize_t[:, :] dst_start,
    const hsize_t[:, :] count,
    const hsize_t[:, :] src_stride,
    const hsize_t[:, :] dst_stride,
):
    """Implements read_many_slices data transfer when src is a h5py.Dataset with simple
    extents only (h5py.Dataset._fast_read_ok).

    Performance notes
    -----------------
    See docstring of read_many_slices
    """
    nslices, ndim = src_start.shape[:2]

    with phil:
        # h5py internals. See h5py/dataset.py and h5py/_selector.pyx.
        dset_id: hid_t = src.id.id
        # must remain in scope for the id to remain valid
        space = src.id.get_space()
        file_space_id: hid_t = space.id
        # must remain in scope for the id to remain valid
        mem_type = py_create(src.dtype)
        mem_type_id: hid_t = mem_type.id

        # libhdf5 is not thread safe (read comment about locking in h5py/_objects.pyx),
        # so we are wrapping everything in a reentrant lock (`with phil`).
        # We can however release the GIL to allow unrelated code to run in parallel.
        with nogil:
            mem_space_id = H5Screate_simple(ndim, dst_shape, NULL)
            if mem_space_id == H5I_INVALID_HID:
                raise HDF5Error()

            for i in range(nslices):
                for j in range(ndim):
                    if count[i, j] == 0:
                        break
                else:
                    # count > 0 along all axes
                    err = H5Sselect_hyperslab(
                        file_space_id,
                        H5S_SELECT_SET,
                        &src_start[i, 0],
                        &src_stride[i, 0],
                        &count[i, 0],
                        NULL,
                    )
                    if err < 0:
                        raise HDF5Error()

                    err = H5Sselect_hyperslab(
                        mem_space_id,
                        H5S_SELECT_SET,
                        &dst_start[i, 0],
                        &dst_stride[i, 0],
                        &count[i, 0],
                        NULL,
                    )
                    if err < 0:
                        raise HDF5Error()

                    err = H5Dread(
                        dset_id,
                        mem_type_id,
                        mem_space_id,
                        file_space_id,
                        H5P_DEFAULT,
                        dst.data,
                    )
                    if err < 0:
                        raise HDF5Error()

            err = H5Sclose(mem_space_id)
            if err < 0:
                raise HDF5Error()


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.infer_types(True)
cdef void _read_many_slices_slow (
    src: np.ndarray | h5py.Dataset,
    np.ndarray dst,
    const hsize_t[:, :] src_start,
    const hsize_t[:, :] dst_start,
    const hsize_t[:, :] count,
    const hsize_t[:, :] src_stride,
    const hsize_t[:, :] dst_stride,
):
    """Implements read_many_slices data transfer when fast transfer cannot be performed.

    This happens when:
    1. src is a h5py.Dataset but h5py.Dataset._fast_read_ok returns False.
       This is to avoid replicating the complex machinery found in
       h5py.Dataset.__getitem__.
    2. src is a numpy array.

    Performance notes
    -----------------
    There has been experimentation to bypass the Python slicing machinery by hacking the
    numpy C API. While functionally successful, there was no material performance gain.

    An alternative approach would be to manually loop over the strides at least for
    basic item sizes (1, 2, 4, and 8 bytes). This has not been attempted yet.
    """
    nslices, ndim = src_start.shape[:2]

    for i in range(nslices):
        for j in range(ndim):
            if count[i, j] == 0:
                break
        else:
            # count > 0 along all axes
            src_idx = []
            dst_idx = []
            for j in range(ndim):
                src_start_ij = src_start[i, j]
                dst_start_ij = dst_start[i, j]
                count_ij = count[i, j]
                src_stride_ij = src_stride[i, j]
                dst_stride_ij = dst_stride[i, j]

                src_stop_ij = count2stop(src_start_ij, count_ij, src_stride_ij)
                dst_stop_ij = count2stop(dst_start_ij, count_ij, dst_stride_ij)

                src_idx.append(slice(src_start_ij, src_stop_ij, src_stride_ij))
                dst_idx.append(slice(dst_start_ij, dst_stop_ij, dst_stride_ij))

            dst[tuple(dst_idx)] = src[tuple(src_idx)]
