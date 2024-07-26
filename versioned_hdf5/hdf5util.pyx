from libc.stddef cimport size_t


cdef extern from "hdf5.h":
    ctypedef long int hid_t
    ctypedef int htri_t
    ctypedef int H5Z_filter_t
    ctypedef int herr_t

    cdef htri_t H5Zfilter_avail(H5Z_filter_t id)

    # From preprocessor DEFINE
    cdef int H5Z_FILTER_MAX
    cdef int H5Z_FILTER_NONE
    cdef int H5Z_FILTER_DEFLATE
    cdef int H5Z_FILTER_SHUFFLE
    cdef int H5Z_FILTER_FLETCHER32
    cdef int H5Z_FILTER_SZIP
    cdef int H5Z_FILTER_NBIT
    cdef int H5Z_FILTER_SCALEOFFSET
    cdef int H5Z_FILTER_RESERVED

KNOWN_FILTERS = {
    H5Z_FILTER_NONE: 'H5Z_FILTER_NONE',
    H5Z_FILTER_DEFLATE: 'H5Z_FILTER_DEFLATE',
    H5Z_FILTER_SHUFFLE: 'H5Z_FILTER_SHUFFLE',
    H5Z_FILTER_FLETCHER32: 'H5Z_FILTER_FLETCHER32',
    H5Z_FILTER_SZIP: 'H5Z_FILTER_SZIP',
    H5Z_FILTER_NBIT: 'H5Z_FILTER_NBIT',
    H5Z_FILTER_SCALEOFFSET: 'H5Z_FILTER_SCALEOFFSET',
    H5Z_FILTER_RESERVED: 'H5Z_FILTER_RESERVED',
}

class FilterInfo:
    """Container for holding info about various HDF5 filters."""
    def __init__(self, int_id: int, *, name: int | None = None):
        self.id: int = int_id
        self.name: str = name if name else KNOWN_FILTERS.get(int_id, "Unknown")

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.name} [ID: {self.id}]"


cpdef get_available_filters():
    """Get information about the available HDF5 filters.

    HDF5 does not provide a way of querying available filters as of 1.14.4,
    other than looping through all filter IDs up to H5Z_FILTER_MAX. Furthermore,
    no filter names or description are available to the user unless the filter
    is already part of a pipeline in a dataset creation property list. For a
    select few builtin filters, we know what these filters are and can provide
    their names, but for any additional filters this is not possible. So this
    function only provides basic information about what filters are available.

    Returns
    -------
    List[Filter]
        List of available filters, their associated IDs, and names (if possible)
    """
    filters: list[FilterInfo] = []
    for i in range(H5Z_FILTER_MAX):
        print("Fetching available filters: ", i, "/", H5Z_FILTER_MAX, end="\r")
        result: htri_t = H5Zfilter_avail(i)
        if result == 0:
            continue
        elif result < 0:
            raise ValueError(f"Unspecified error determining if filter {i} is available.")

        filters.append(FilterInfo(i))

    return filters
