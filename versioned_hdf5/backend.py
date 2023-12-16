from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
from h5py import Dataset, File, Group, VirtualLayout, VirtualSource, h5s
from h5py._hl.filters import guess_chunk
from ndindex import ChunkSize, Slice, Tuple, ndindex
from numpy.testing import assert_array_equal

from .hashtable import Hashtable
from .slicetools import (
    get_shape,
    get_vchunk_overlap,
    partition,
    spaceid_to_slice,
    to_raw_index,
    to_slice_tuple,
)
from .utils import AppendChunk, WriteChunk

DEFAULT_CHUNK_SIZE = 2**12
DATA_VERSION = 4
# data_version 2 has broken hashtables, always need to rebuild
# data_version 3 hash collisions for string arrays which, when concatenated, give the same string
CORRUPT_DATA_VERSIONS = frozenset([2, 3])

if TYPE_CHECKING:
    from .wrappers import InMemoryDataset


class SplitResult:
    """Object which stores the result of splitting a dataset across the last chunk."""

    def __init__(
        self,
        arr_to_append: np.ndarray,
        arr_to_write: np.ndarray,
        new_raw_last_chunk: Tuple,
        new_raw_last_chunk_data: np.ndarray,
    ):
        """Init SplitResult.

        Parameters
        ----------
        arr_to_append : np.ndarray
            Array to be appended to the dataset
        arr_to_write : np.ndarray
            Array to be written into new chunks
        new_raw_last_chunk : Tuple
            Slice of the raw data containing the last chunk
        new_raw_last_chunk_data : np.ndarray

        """
        self.arr_to_append = arr_to_append
        self.arr_to_write = arr_to_write
        self.new_raw_last_chunk = new_raw_last_chunk
        self.new_raw_last_chunk_data = new_raw_last_chunk_data

    def has_append_data(self) -> bool:
        """Check whether there is data to append.

        Returns
        -------
        bool
            True if there is data to append, False otherwise
        """
        return self.arr_to_append.size > 0

    def has_write_data(self) -> bool:
        """Check whether there is data to write into a new chunk.

        Returns
        -------
        bool
            True if there is data to write into a new chunk, False otherwise
        """
        return self.arr_to_write.size > 0

    def get_additional_rchunks_needed(self, chunk_size: int) -> int:
        """Compute the number of additional chunks needed in the raw dataset.

        Parameters
        ----------
        chunk_size : int
            Number of elements in each chunk (along axis 0)

        Returns
        -------
        int
            Number of additional chunks of size chunk_size needed to store the data
            that doesn't fit in the remaining free space in the dataset
        """
        return int(self.arr_to_write.shape[0] / chunk_size + 0.5)

    def get_new_vshape(self, old_vshape: tuple[int, ...]) -> tuple[int, ...]:
        """Get the new shape of the virtual dataset post-append.

        Parameters
        ----------
        old_vshape : tuple[int, ...]
            Previous shape of the dataset

        Returns
        -------
        tuple[int, ...]
            Shape of the dataset post-append
        """
        return (
            (old_vshape[0] + self.arr_to_append.shape[0] + self.arr_to_write.shape[0]),
            *old_vshape[1:],
        )

    def get_new_raw_shape(self, raw_data: Dataset) -> tuple:
        """Get the new shape for the raw data post-append.

        Parameters
        ----------
        raw_data : Dataset
            Raw data which is to be appended to

        Returns
        -------
        tuple
            Shape of the raw data post-append
        """
        chunk_size = tuple(raw_data.attrs["chunks"])[0]
        return (
            raw_data.shape[0]
            + self.get_additional_rchunks_needed(chunk_size) * chunk_size,
            *raw_data.shape[1:],
        )

    def get_new_last_vchunk(self, slices: Dict[Tuple, Tuple]) -> Tuple:
        """Get the last slice of the virtual dataset post-append.

        Parameters
        ----------
        slices : Dict[Tuple, Tuple]
            Mapping between {slices in virtual dataset: slices in raw dataset} which
            make up the virtual dataset of the previous version.

        Returns
        -------
        Tuple
            Slices of the new version of the virtual dataset post-append
        """
        last_vchunk = list(slices)[-1]
        dim0 = last_vchunk.args[0]
        return Tuple(
            Slice(dim0.start, dim0.stop + self.arr_to_append.shape[0]),
            *last_vchunk.args[1:],
        )

    def get_new_last_rchunk(
        self, raw_data: Dataset, slices: Dict[Tuple, Tuple]
    ) -> Tuple:
        """Get the last slice of the raw dataset post-append.

        Parameters
        ----------
        raw_data : Dataset
            Raw dataset where data will be written
        slices : Dict[Tuple, Tuple]
            Mapping between {slices in virtual dataset: slices in raw dataset} which
            make up the virtual dataset of the previous version.

        Returns
        -------
        Tuple
            Last slice of the of the raw dataset post-append
        """
        chunk_size = tuple(raw_data.attrs["chunks"])[0]
        last_chunk_start = raw_data.shape[0] - chunk_size
        last_chunk_length = len(self.get_new_last_vchunk(slices).args[0])
        last_chunk_end = last_chunk_start + last_chunk_length
        return Tuple(
            Slice(
                last_chunk_start,
                last_chunk_end,
            ),
            *[Slice(None, None) for _ in raw_data.shape[1:]],
        )

    def get_append_rchunk_slice(self, slices: Dict[Tuple, Tuple]) -> Tuple:
        """Get the slice into the raw data where the new data will be appended into.

        Parameters
        ----------
        slices : Dict[Tuple, Tuple]
            Mapping between {slices in virtual dataset: slices in raw dataset} which
            make up the virtual dataset of the previous version.

        Returns
        -------
        Tuple
            Slice of the raw dataset where the data is to be appended
        """
        last_rchunk = list(slices.values())[-1]
        dim0 = last_rchunk.args[0]
        return Tuple(
            Slice(dim0.stop, dim0.stop + self.arr_to_append.shape[0]),
            *last_rchunk.args[1:],
        )


def normalize_dtype(dtype):
    return np.array([], dtype=dtype).dtype


def get_chunks(shape, dtype, chunk_size):
    # TODO: Implement this
    if len(shape) > 1:
        raise NotImplementedError
    return (chunk_size,)


def initialize(f):
    import datetime

    from .versions import TIMESTAMP_FMT

    version_data = f.create_group("_version_data")
    versions = version_data.create_group("versions")
    versions.create_group("__first_version__")
    versions.attrs["current_version"] = "__first_version__"
    ts = datetime.datetime.now(datetime.timezone.utc)
    versions["__first_version__"].attrs["timestamp"] = ts.strftime(TIMESTAMP_FMT)
    versions.attrs["data_version"] = DATA_VERSION


def create_base_dataset(
    f,
    name,
    *,
    shape=None,
    data=None,
    dtype=None,
    chunks=True,
    compression=None,
    compression_opts=None,
    fillvalue=None,
):
    # Validate shape (based on h5py._hl.dataset.make_new_dset)
    if shape is None:
        if data is None:
            if dtype is None:
                raise TypeError("One of data, shape or dtype must be specified")
            raise NotImplementedError("empty datasets are not yet implemented")
        shape = data.shape
    else:
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if data is not None and (np.prod(shape, dtype=np.ulonglong) != np.prod(data.shape, dtype=np.ulonglong)):
            raise ValueError("Shape tuple is incompatible with data")

    ndims = len(shape)
    if isinstance(chunks, int) and not isinstance(chunks, bool):
        chunks = (chunks,)
    if chunks in [True, None]:
        if ndims == 0:
            # Chunks are not allowed for scalar datasets; keeping original
            # behavior here
            chunks = (DEFAULT_CHUNK_SIZE,)
        elif ndims == 1:
            chunks = guess_chunk(shape, None, data.dtype.itemsize)
        else:
            raise NotImplementedError(
                "chunks must be specified for multi-dimensional datasets"
            )
    group = f["_version_data"].create_group(name)

    if dtype is None:
        # https://github.com/h5py/h5py/issues/1474
        dtype = data.dtype
    dtype = normalize_dtype(dtype)
    if dtype.metadata and (
        "vlen" in dtype.metadata or "h5py_encoding" in dtype.metadata
    ):
        # h5py string dtype
        # (https://h5py.readthedocs.io/en/2.10.0/strings.html). Setting the
        # fillvalue in this case doesn't work
        # (https://github.com/h5py/h5py/issues/941).
        if fillvalue not in [0, "", b"", None]:
            raise ValueError(
                "Non-default fillvalue not supported for variable length strings"
            )
        fillvalue = None
    dataset = group.create_dataset(
        "raw_data",
        shape=(0,) + chunks[1:],
        chunks=tuple(chunks),
        maxshape=(None,) + chunks[1:],
        dtype=dtype,
        compression=compression,
        compression_opts=compression_opts,
        fillvalue=fillvalue,
    )
    dataset.attrs["chunks"] = chunks
    return write_dataset(f, name, data, chunks=chunks)


def write_dataset(
    f,
    name,
    data,
    chunks=None,
    dtype=None,
    compression=None,
    compression_opts=None,
    fillvalue=None,
):
    if name not in f["_version_data"]:
        return create_base_dataset(
            f,
            name,
            data=data,
            dtype=dtype,
            chunks=chunks,
            compression=compression,
            compression_opts=compression_opts,
            fillvalue=fillvalue,
        )

    ds = f["_version_data"][name]["raw_data"]
    if isinstance(chunks, int) and not isinstance(chunks, bool):
        chunks = (chunks,)
    if chunks is None:
        chunks = tuple(ds.attrs["chunks"])
    else:
        if chunks != tuple(ds.attrs["chunks"]):
            raise ValueError(
                "Chunk size specified but doesn't match already existing chunk size"
            )

    if dtype is not None:
        if dtype != ds.dtype:
            raise ValueError("dtype specified but doesn't match already existing dtype")

    if (
        compression
        and compression != ds.compression
        or compression_opts
        and compression_opts != ds.compression_opts
    ):
        raise ValueError(
            "Compression options can only be specified for the first version of a dataset"
        )
    if fillvalue is not None and fillvalue != ds.fillvalue:
        dtype = ds.dtype
        if dtype.metadata and (
            "vlen" in dtype.metadata or "h5py_encoding" in dtype.metadata
        ):
            # Variable length string dtype. The ds.fillvalue will be None in
            # this case (see create_virtual_dataset() below)
            pass
        else:
            raise ValueError(f"fillvalues do not match ({fillvalue} != {ds.fillvalue})")
    if data.dtype != ds.dtype:
        raise ValueError(f"dtypes do not match ({data.dtype} != {ds.dtype})")

    # TODO: Handle more than one dimension
    old_shape = ds.shape
    slices = {}
    slices_to_write = {}
    chunk_size = chunks[0]

    with Hashtable(f, name) as hashtable:
        if len(data.shape) != 0:
            for data_slice in ChunkSize(chunks).indices(data.shape):
                data_s = data[data_slice.raw]
                data_hash = hashtable.hash(data_s)

                if data_hash in hashtable:
                    hashed_slice = hashtable[data_hash]
                    slices[data_slice] = hashed_slice

                    _verify_new_chunk_reuse(
                        raw_data=ds,
                        new_data=data,
                        data_hash=data_hash,
                        hashed_slice=hashed_slice,
                        chunk_being_written=data_s,
                        slices_to_write=slices_to_write,
                    )
                else:
                    idx = hashtable.largest_index
                    raw_slice = Slice(
                        idx * chunk_size, idx * chunk_size + data_s.shape[0]
                    )
                    slices[data_slice] = raw_slice
                    hashtable[data_hash] = raw_slice
                    slices_to_write[raw_slice] = data_slice

                    # Keep track of the last index written to in the raw dataset
                    # for this chunk; future appends are simplified by this
                    ds.attrs["last_element"] = raw_slice.stop

            ds.resize((old_shape[0] + len(slices_to_write) * chunk_size,) + chunks[1:])
            for raw_slice, data_slice in slices_to_write.items():
                data_s = data[data_slice.raw]
                idx = Tuple(raw_slice, *[slice(0, i) for i in data_s.shape[1:]])
                ds[idx.raw] = data[data_slice.raw]
    return slices


def _verify_new_chunk_reuse(
    raw_data: np.ndarray,
    new_data: np.ndarray,
    data_hash: bytes,
    hashed_slice: Tuple,
    chunk_being_written: Tuple,
    slices_to_write: Optional[Dict[Slice, Tuple]] = None,
    data_to_write: Optional[Dict[Slice, np.ndarray]] = None,
):
    """Check that the data corresponding to the slice in the hashtable matches the data
    that is going to be written.

    Raises a ValueError if the data reference by the hashed slice doesn't match the
    underlying raw data.

    This function retrieves a reused chunk of data either from the ``slices_to_write``,
    if the data has not yet been written to the file, or from the ``raw_data`` that has
    already been written.

    Parameters
    ----------
    raw_data : np.ndarray
        Raw data that already exists in the file
    new_data : np.ndarray
        New data that we are writing
    data_hash : bytes
        Hash of the new data chunk
    hashed_slice : Tuple
        Slice that is stored in the hash table for the given data_hash. This is a slice
        into the raw_data for the dataset; however if the data has not yet been written
        it may not point to a valid region in raw_data (but in that case it _would_
        point to a slice in ``slices_to_write``)
    chunk_being_written : np.ndarray
        New data chunk to be written
    slices_to_write : Optional[Dict[Slice, Tuple]]
        Dict of slices which will be written. Maps slices that will exist in the
        raw_data once the write is complete to slices of the dataset that is being
        written.
    data_to_write : Optional[Dict[Slice, Tuple]]
        Dict of arrays which will be written as chunks. Maps slices that will exist in
        the raw_data once the write is complete to chunks of the dataset that is being
        written. If ``data_to_write`` is specified, ``slices_to_write`` must be None.
    """
    if slices_to_write is not None and hashed_slice in slices_to_write:
        # The hash table contains a slice we will write but haven't yet; grab the
        # chunk from the new data being written
        reused_chunk = new_data[slices_to_write[hashed_slice].raw]
    elif data_to_write is not None and hashed_slice in data_to_write:
        # The hash table contains a slice we will write but haven't yet; grab the
        # chunk from the data_to_write dict, which stores the data that will be written
        # for the given hashed slice.
        reused_chunk = data_to_write[hashed_slice]
    else:
        # The hash table contains a slice that was written in a previous
        # write operation; grab that chunk from the existing raw data
        reused_slice = Tuple(
            hashed_slice, *[slice(0, size) for size in new_data.shape[1:]]
        )
        reused_chunk = raw_data[reused_slice.raw]

    # In some cases type coersion can happen during the write process even if the dtypes
    # are the same - for example, if the raw_data.dtype == dtype('O'), but the elements
    # are bytes, and chunk_being_written.dtype == dtype('O'), but the elements are
    # utf-8 strings. For this case, when the raw_data is changed, e.g.
    #      raw_data[some_slice] = chunk_being_written[another_slice]
    # the data that gets written is bytes. So in certain cases, just calling
    # assert_array_equal doesn't work. Instead, we convert each element to a bytestring
    # first.
    if reused_chunk.dtype == "O" and chunk_being_written.dtype == "O":
        to_be_written = _convert_to_bytes(chunk_being_written)
    else:
        to_be_written = chunk_being_written

    assert_array_equal(
        reused_chunk,
        to_be_written,
        err_msg=(
            f"Hash {data_hash} of existing data chunk {reused_chunk} "
            f"matches the hash of new data chunk {chunk_being_written}, "
            "but data does not."
        ),
    )


def _convert_to_bytes(arr: np.ndarray) -> np.ndarray:
    """Convert each element in the array to bytes.

    Each element in the array is assumed to be the same type, even if the input is an
    object dtype array.

    Parameters
    ----------
    arr : np.ndarray
        Array to be converted; no conversion is done if the elements are already bytes

    Returns
    -------
    np.ndarray
        Object dtype array filled with elements of type bytes
    """
    if len(arr) > 0 and isinstance(arr[0], bytes):
        return arr
    else:
        return np.vectorize(lambda i: bytes(i, encoding="utf-8"))(arr)


def write_dataset_chunks(
    f: File,
    name: str,
    data_dict: Dict[Tuple, Union[Tuple, np.ndarray]],
    shape: Optional[tuple] = None,
) -> Dict[Tuple, Tuple]:
    """Write chunks in data_dict to the raw data.

    data_dict should be a dictionary mapping chunk_size index to either an
    array for that chunk, or a slice into the raw data for that chunk

    Parameters
    ----------
    f : File

    name : str

    data_dict : Dict[Tuple, Union[Slice, np.ndarray]]
        Mapping between indices in the virtual dataset and either

            1. Slices along the first dimension of the raw dataset
               (other dimensions are implicit, and depend on the
               shape of the virtual dataset); or
            2. A new numpy array to write to the raw data for the
               slice in the virtual dataset

    shape : Optional[tuple]


    Returns
    -------
    Dict[Tuple, Tuple]
        Mapping between slices in the virtual dataset to slices in the raw dataset
    """
    if name not in f["_version_data"]:
        raise NotImplementedError(
            "Use write_dataset() if the dataset does not yet exist"
        )

    raw_data = f["_version_data"][name]["raw_data"]
    chunks = tuple(raw_data.attrs["chunks"])
    chunk_size = chunks[0]

    if shape is None:
        shape = tuple(
            max(c.args[i].stop for c in data_dict) for i in range(len(chunks))
        )

    with Hashtable(f, name) as hashtable:
        slices = {i: None for i in data_dict}

        # Mapping from slices in the raw dataset after this write is complete to ndarray
        # chunks of the new data which will be written
        data_to_write = {}
        for chunk, data_s in data_dict.items():
            if isinstance(data_s, (slice, tuple, Tuple, Slice)):
                slices[chunk] = ndindex(data_s)
            elif isinstance(data_s, AppendChunk):
                slices[chunk] = data_s.write_to_raw(hashtable, raw_data)
            elif isinstance(data_s, WriteChunk):
                slices.update(data_s.write_to_raw(hashtable, raw_data))
            else:
                if data_s.dtype != raw_data.dtype:
                    raise ValueError(
                        f"dtypes do not match ({data_s.dtype} != {raw_data.dtype})"
                    )

                data_hash = hashtable.hash(data_s)

                if data_hash in hashtable:
                    hashed_slice = hashtable[data_hash]
                    slices[chunk] = hashed_slice

                    _verify_new_chunk_reuse(
                        raw_data=raw_data,
                        new_data=data_s,
                        data_hash=data_hash,
                        hashed_slice=hashed_slice,
                        chunk_being_written=data_s,
                        data_to_write=data_to_write,
                    )

                else:
                    idx = hashtable.largest_index
                    raw_slice = Slice(
                        idx * chunk_size, idx * chunk_size + data_s.shape[0]
                    )
                    slices[chunk] = raw_slice
                    hashtable[data_hash] = raw_slice
                    data_to_write[raw_slice] = data_s
                    raw_data.attrs["last_element"] = raw_slice.stop

    assert None not in slices.values()
    old_shape = raw_data.shape
    raw_data.resize((old_shape[0] + len(data_to_write) * chunk_size,) + chunks[1:])
    for raw_slice, data_s in data_to_write.items():
        c = (raw_slice.raw,) + tuple(slice(0, i) for i in data_s.shape[1:])
        raw_data[c] = data_s

    # TODO: should this return Tuple (not Slice) elements?
    return slices


def create_virtual_dataset(
    f, version_name, name, shape, slices, attrs=None, fillvalue=None
):
    from h5py._hl.selections import select
    from h5py._hl.vds import VDSmap

    raw_data = f["_version_data"][name]["raw_data"]
    raw_data_shape = raw_data.shape

    # Reduce the raw chunks, and strip out any extra dimensions (if any rchunks
    # are Tuples). The shape of the raw data corresponding to a given virtual
    # chunk is implied by the shape of the virtual chunk
    new_slices = {}
    for vchunk, rchunk in slices.items():
        reduced_rchunk = rchunk.reduce()
        if isinstance(reduced_rchunk, Tuple):
            new_slices[vchunk] = reduced_rchunk.args[0]
        else:
            new_slices[vchunk] = reduced_rchunk
    slices = new_slices

    if len(raw_data) == 0:
        layout = VirtualLayout(shape=(0,), dtype=raw_data.dtype)
    else:
        # Chunks in the raw dataset are expanded along the first dimension only.
        # Since the chunks are pointed to by virtual datasets, it doesn't make
        # sense to expand the chunks in the raw dataset along multiple dimensions
        # (the true layout of the chunks in the raw dataset is irrelevant).
        for c, s in slices.items():
            if len(c.args[0]) != len(s):
                raise ValueError(f"Inconsistent slices dictionary ({c.args[0]}, {s})")

        # h5py 3.3 changed the VirtualLayout code so that it no longer uses
        # sources. See https://github.com/h5py/h5py/pull/1905.
        layout = VirtualLayout(shape, dtype=raw_data.dtype)
        layout_has_sources = hasattr(layout, "sources")
        if not layout_has_sources:
            from h5py import _selector

            layout._src_filenames.add(b".")
            space = h5s.create_simple(shape)
            selector = _selector.Selector(space)

        for c, s in slices.items():
            if c.isempty():
                continue
            # idx = Tuple(s, *Tuple(*[slice(0, i) for i in shape[1:]]).as_subindex(Tuple(*c.args[1:])).args)
            S = [Slice(0, len(c.args[i])) for i in range(1, len(shape))]
            idx = Tuple(s, *S)
            # assert c.newshape(shape) == vs[idx.raw].shape, (c, shape, s)

            # This is equivalent to
            #
            # layout[c.raw] = vs[idx.raw]
            #
            # but faster because vs[idx.raw] does a deepcopy(vs), which is
            # slow. We need different versions for h5py 2 and 3 because the
            # virtual sources code was rewritten.
            if not layout_has_sources:
                key = idx.raw
                vs_sel = select(raw_data.shape, key, dataset=None)

                sel = selector.make_selection(c.raw)
                layout.dcpl.set_virtual(
                    sel.id, b".", raw_data.name.encode("utf-8"), vs_sel.id
                )

            else:
                vs_sel = select(raw_data_shape, idx.raw, None)
                layout_sel = select(shape, c.raw, None)
                layout.sources.append(
                    VDSmap(layout_sel.id, ".", raw_data.name, vs_sel.id)
                )

    dtype = raw_data.dtype
    if dtype.metadata and (
        "vlen" in dtype.metadata or "h5py_encoding" in dtype.metadata
    ):
        # Variable length string dtype
        # (https://h5py.readthedocs.io/en/2.10.0/strings.html). Setting the
        # fillvalue in this case doesn't work
        # (https://github.com/h5py/h5py/issues/941).
        if fillvalue not in [0, "", b"", None]:
            raise ValueError(
                "Non-default fillvalue not supported for variable length strings"
            )
        fillvalue = None

    # Note that due to a bug in Group.create_virtual_dataset, empty virtual datasets are
    # not actually virtual. See https://github.com/h5py/h5py/issues/1660
    # for the relevant discussion.
    virtual_data = f["_version_data/versions"][version_name].create_virtual_dataset(
        name, layout, fillvalue=fillvalue
    )

    if attrs:
        for k, v in attrs.items():
            virtual_data.attrs[k] = v
    virtual_data.attrs["raw_data"] = raw_data.name
    virtual_data.attrs["chunks"] = raw_data.chunks
    return virtual_data


class WriteOperation:
    """Base class for dataset manipulations."""

    def apply(
        self,
        f: File,
        name: str,
        version: str,
        slices: Dict[Tuple, Tuple],
        shape: tuple[int, ...],
    ) -> tuple[Dict[Tuple, Tuple], tuple[int, ...]]:
        """Apply the operation to the chunks in memory.

        Parameters
        ----------
        f : File
            File containing the dataset to write to
        name : str
            Name of the dataset
        version : str
            Version of the dataset to write to
        slices : Dict[Tuple, Tuple]
            Mapping between {slices in virtual dataset: slices in raw dataset}
            in the virtual datset initially
        shape: tuple[int, ...]
            Current shape of the data

        Returns
        -------
        tuple[Dict[Tuple, Tuple], tuple[int, ...]]
            Mapping between {slices in virtual dataset: slices in raw dataset}
            operated on by this function; and shape of the current dataset
        """
        raise NotImplementedError

    def write_raw(
        self, f, name, chunks, shape
    ) -> tuple[Dict[Tuple, Tuple], tuple[int, ...]]:
        return write_dataset_chunks(f, name, chunks, shape), shape


class ResizeOperation(WriteOperation):
    def __init__(
        self,
        shape: tuple[int, ...],
        chunks: ChunkSize,
        fillvalue: Union[int, float, str],
        dtype: np.dtype,
    ):
        self.shape = shape
        self.chunks = chunks
        self.fillvalue = fillvalue
        self.dtype = dtype

    def apply(
        self,
        f: File,
        name: str,
        _version: str,
        slices: Dict[Tuple, Tuple],
        _shape: tuple[int, ...],
    ) -> tuple[Dict[Tuple, Tuple], tuple[int, ...]]:
        """Reshape the existing slices to the new shape.

        Does not work like numpy's reshape - data is not shuffled to fit the new shape.
        Instead, data in the existing shape that falls outside the new shape is lost,
        and new elements in the new shape are filled with `self.fillvalue`.

        Parameters
        ----------
        f : File
            File containing the dataset to write to
        name : str
            Name of the dataset
        _version : str
            Version of the dataset to write to; unused (this is provided by the
            slices dict)
        slices : Dict[Tuple, Tuple]
            Mapping between {slices in virtual dataset: slices in raw dataset}
            in the virtual datset initially
        _shape: tuple[int, ...]
            Current shape of the data; unused (old shape is irrelevant, we are
            reshaping here)

        Returns
        -------
        tuple[Dict[Tuple, Tuple], tuple[int, ...]]
            Mapping between {slices in virtual dataset: slices in raw dataset}
            which were written by this function; and shape of the current dataset
        """
        new_shape_index = Tuple(*[Slice(0, i) for i in self.shape])
        raw_data: Dataset = f["_version_data"][name]["raw_data"]  # type: ignore

        # Keep a copy of the old slices; it will be needed later to compute
        # the chunks modified by the reshape operation
        current_slices = slices.copy()

        new_slices = {}
        for vchunk in partition(new_shape_index, self.chunks):
            # If the new virtual chunk is in the old set of slices, just use the same
            # raw data the virtual chunk is already mapped to. Pop it out of the slices
            # dict so that we don't need to iterate over it when computing parts of the
            # dataset with the new shape.
            if vchunk in current_slices:
                new_slices[vchunk] = current_slices.pop(vchunk)

            # Otherwise compute the overlap of the new vchunk and the old vchunks, and
            # then compute what the raw data should be.
            else:
                new_slices[vchunk] = arr_from_chunks(
                    current_slices,
                    raw_data,
                    vchunk,
                    self.shape,
                )

        return new_slices, self.shape

    def __repr__(self) -> str:
        return f"ResizeOperation: {self.shape}"


class SetOperation(WriteOperation):
    """Operation which indexes the dataset to write data."""

    def __init__(self, index: Tuple, arr: np.ndarray):
        """Initialize a SetOperation.

        Parameters
        ----------
        index : Tuple
            Virtual dataset index where ``arr`` is to be written
        arr : np.ndarray
            Array to write to the dataset
        """
        self.index = index
        self.arr = arr

    def __repr__(self):
        return f"SetOperation: {self.index} = {self.arr}"

    def apply(
        self,
        f: File,
        name: str,
        _version: str,
        slices: Dict[Tuple, Tuple],
        shape: tuple[int, ...],
    ) -> tuple[Dict[Tuple, Tuple], tuple[int, ...]]:
        """Write the stored data to the dataset in chunks.

        Parameters
        ----------
        f : File
            File containing the dataset to write to
        name : str
            Name of the dataset
        version : str
            Version of the dataset to write to; unused (this is provided by the
            slices dict)
        slices : Dict[Tuple, Tuple]
            Mapping between {slices in virtual dataset: slices in raw dataset}
            in the virtual datset initially
        shape: tuple[int, ...]
            Current shape of the data

        Returns
        -------
        tuple[Dict[Tuple, Tuple], tuple[int, ...]]
            Mapping between {slices in virtual dataset: slices in raw dataset}
            which were written by this function; and shape of the current dataset
        """
        # Ensure each element of the index is a slice. Reduce the index
        # onto the shape of the current dataset to remove NoneType or Eliipsis
        # index shape args.
        index = to_slice_tuple(ndindex(self.index).expand(shape))

        # If the shape of the array doesn't match the shape of the
        # index to assign the array to, broadcast it first.
        index_shape = tuple(len(dim) for dim in index.args)
        if self.arr.shape != index_shape:
            arr = np.broadcast_to(self.arr, index_shape)
        else:
            arr = self.arr

        raw_data: Dataset = f["_version_data"][name]["raw_data"]

        new_chunks = get_updated_chunks(
            slices,
            index,
            arr,
            shape,
            raw_data,
        )

        return new_chunks, shape


class AppendOperation(WriteOperation):
    """Operation which appends data to a dataset."""

    def __init__(self, value: np.ndarray):
        """Initialize a WriteOperation.

        Parameters
        ----------
        value : np.ndarray
            Array to append to the dataset
        """
        self.value = value

    def __repr__(self):
        return f"AppendOperation: {self.value}"

    def apply(
        self,
        f: File,
        name: str,
        version: str,
        slices: Dict[Tuple, Tuple],
        shape: tuple[int, ...],
    ) -> tuple[Dict[Tuple, Tuple], tuple[int, ...]]:
        """Append data to the raw dataset.

        1. Split the data into a part which can fit in the last chunk, and a part which
           can't.
        2. Compute the hash of the last virtual chunk with the new data appended.
        3. Add either the hashed raw slice (if in the hashtable, and therefore the raw data)
           or the new slice to the slice dict.
        4. If the data to append doesn't fit in the unused space in the last chunk, write
           any additional data into new chunk(s).

        Parameters
        ----------
        f : File
            File where the data is to be written
        version : str
            Version of the version to appending data to
        name : str
            Name of the dataset to append data to
        arr : np.ndarray
            Data to append to the dataset
        slices : Dict[Tuple, Tuple]
            Slices of the virtual dataset that is being appended to; maps {slices in
            virtual dataset: slices in raw dataset}. If only one append
            operation is being carried out, this is the value returned by.
            `get_previous_version_slices(f, version_name, name)`. If multiple WriteOperation
            operations are being carried out, these slices represent the result of all the
            write operations applied to the virtual dataset
        shape : tuple[int, ...]
            Shape of the dataset pre-append

        Returns
        -------
        Tuple[Dict[Tuple, Tuple], tuple[int, ...]]
            Mapping between {slices in virtual dataset: slices in raw dataset}
            which were written by this function; and shape of the current dataset
        """
        if not slices:
            raise ValueError("Cannot append to empty dataset.")

        raw_data: Dataset = f["_version_data"][name]["raw_data"]

        if raw_data.dtype != self.value.dtype:
            raise ValueError(
                f"dtypes of raw data ({raw_data.dtype}) does not match data to append "
                f"({self.value.dtype})"
            )

        # Get the slices from the previous version; they are reused here
        last_virtual_slice = list(sorted(slices, key=lambda obj: obj.args[0].start))[-1]

        # Split the data to append into a part which fits in the last
        # chunk of the raw data, and the part that doesn't
        split = split_across_unused(f, name, self.value)

        # If there's empty space in the last chunk of the raw data, append as much
        # data as will fit
        if split.has_append_data():
            with Hashtable(f, name) as hashtable:
                # Get the new virtual and raw last chunks
                vchunk = split.get_new_last_vchunk(slices)
                rchunk = split.get_new_last_rchunk(raw_data, slices)

                # Get the indices to write the new data into
                append_slice = split.get_append_rchunk_slice(slices)

                # Remove the last chunk of the virtual dataset; we are
                # replacing it with the chunk containing the appended data
                del slices[last_virtual_slice]

                # If the hash of the data is already in the hash table,
                # just reuse the hashed slice. Otherwise, update the
                # hash table with the new data hash and write the data
                # to the raw dataset.
                data_hash = hashtable.hash(split.new_raw_last_chunk_data)
                if data_hash in hashtable:
                    slices[vchunk] = hashtable[data_hash]
                else:
                    # Update the slices mapping
                    slices[vchunk] = rchunk

                    # Update the hashtable
                    hashtable[data_hash] = rchunk

                    # Write the data
                    raw_data[append_slice.raw] = split.arr_to_append

                    # Keep track of the last index written to in the raw dataset;
                    # future appends are simplified by this
                    raw_data.attrs["last_element"] = rchunk.args[0].stop

        if split.has_write_data():
            if split.has_append_data():
                last_virtual_index = (
                    last_virtual_slice.args[0].stop + split.arr_to_append.shape[0]
                )
            else:
                last_virtual_index = last_virtual_slice.args[0].stop

            virtual_slice_to_write = Tuple(
                Slice(
                    last_virtual_index,
                    last_virtual_index + split.arr_to_write.shape[0],
                ),
                *[Slice(None, None) for _ in last_virtual_slice.args[1:]],
            )

            slices.update(
                write_to_dataset(
                    f,
                    version,
                    name,
                    virtual_slice_to_write,
                    split.arr_to_write,
                )
            )

        return slices, split.get_new_vshape(shape)

    def apply2(
        self,
        f: File,
        name: str,
        version: str,
        slices: Dict[Tuple, Tuple],
        shape: tuple[int, ...],
    ) -> tuple[Dict[Tuple, Tuple], tuple[int, ...]]:
        """Append data to the raw dataset.

        1. Split the data into a part which can fit in the last chunk, and a part which
           can't.
        2. Compute the hash of the last virtual chunk with the new data appended.
        3. Add either the hashed raw slice (if in the hashtable, and therefore the raw data)
           or the new slice to the slice dict.
        4. If the data to append doesn't fit in the unused space in the last chunk, write
           any additional data into new chunk(s).

        Parameters
        ----------
        f : File
            File where the data is to be written
        version : str
            Version of the version to appending data to
        name : str
            Name of the dataset to append data to
        arr : np.ndarray
            Data to append to the dataset
        slices : Dict[Tuple, Tuple]
            Slices of the virtual dataset that is being appended to; maps {slices in
            virtual dataset: slices in raw dataset}. If only one append
            operation is being carried out, this is the value returned by.
            `get_previous_version_slices(f, version_name, name)`. If multiple WriteOperation
            operations are being carried out, these slices represent the result of all the
            write operations applied to the virtual dataset
        shape : tuple[int, ...]
            Shape of the dataset pre-append

        Returns
        -------
        Tuple[Dict[Tuple, Tuple], tuple[int, ...]]
            Mapping between {slices in virtual dataset: slices in raw dataset}
            which were written by this function; and shape of the current dataset
        """
        if not slices:
            raise ValueError("Cannot append to empty dataset.")

        raw_data: Dataset = f["_version_data"][name]["raw_data"]

        if raw_data.dtype != self.value.dtype:
            raise ValueError(
                f"dtypes of raw data ({raw_data.dtype}) does not match data to append "
                f"({self.value.dtype})"
            )

        # Get the slices from the previous version; they are reused here
        last_virtual_slice = list(sorted(slices, key=lambda obj: obj.args[0].start))[-1]

        # Split the data to append into a part which fits in the last
        # chunk of the raw data, and the part that doesn't
        split = split_across_unused(f, name, self.value)

        # If there's empty space in the last chunk of the raw data, append as much
        # data as will fit
        if split.has_append_data():
            # Get the new virtual and raw last chunks
            vchunk = split.get_new_last_vchunk(slices)
            # rchunk = split.get_new_last_rchunk(raw_data, slices)

            # Get the indices to write the new data into
            append_slice = split.get_append_rchunk_slice(slices)

            # Remove the last chunk of the virtual dataset; we are
            # replacing it with the chunk containing the appended data
            del slices[last_virtual_slice]

            slices[vchunk] = AppendChunk(
                target_raw_index=append_slice,
                target_raw_data=split.arr_to_append,
                raw_last_chunk=split.new_raw_last_chunk,
                raw_last_chunk_data=split.new_raw_last_chunk_data,
                # rchunk=rchunk,
            )

        if split.has_write_data():
            if split.has_append_data():
                last_virtual_index = (
                    last_virtual_slice.args[0].stop + split.arr_to_append.shape[0]
                )
            else:
                last_virtual_index = last_virtual_slice.args[0].stop

            virtual_slice_to_write = Tuple(
                Slice(
                    last_virtual_index,
                    last_virtual_index + split.arr_to_write.shape[0],
                ),
                *[Slice(None, None) for _ in last_virtual_slice.args[1:]],
            )

            slices[virtual_slice_to_write] = split.arr_to_write

        return slices, split.get_new_vshape(shape)


def write_dataset_operations(
    f: File,
    version_name: str,
    name: str,
    dataset: "InMemoryDataset",
) -> tuple[Dict[Tuple, Tuple], tuple[int, ...]]:
    chunks, shape = write_operations(f, version_name, name, dataset._operations)
    result = write_dataset_chunks(f, name, chunks, shape)
    dataset._operations.clear()
    return result, shape


def write_operations(
    f: File, version_name: str, name: str, operations: List[WriteOperation]
) -> tuple[Dict[Tuple, Tuple], tuple[int, ...]]:
    """Carry out a sequence of write operations on the file.

    If no operations are pending, just return the previous version slices and shape.

    Parameters
    ----------
    f : File
        File to write
    version_name : str
        Version name of the data to be written
    name : str
        Name of the dataset to be written
    operations : List[WriteOperation]
        List of write operations to perform

    Returns
    -------
    tuple[Dict[Tuple, Tuple], tuple[int, ...]]
        (Slices map, shape of virtual dataset post-write)

        The slices map is a mapping from {virtual dataset slice: raw dataset slice}.
        The virtual dataset is created elsewhere using the slices return here.
    """
    if name not in f["_version_data"]:
        raise NotImplementedError(
            "Use write_dataset() if the dataset does not yet exist"
        )

    slices = get_previous_version_slices(f, version_name, name)
    shape = get_previous_version_shape(f, version_name, name)

    for operation in operations:
        slices, shape = operation.apply(f, name, version_name, slices, shape)

    sorted_slices = dict(sorted(list(slices.items()), key=lambda s: s[0].args[0].start))

    return sorted_slices, shape


def last_raw_used_chunk(f: File, name: str) -> Tuple:
    """Get the part of the last chunk in the raw dataset used by a virtual dataset.

    Parameters
    ----------
    f : File
        File for which the used part of the last chunk is to be retrieved
    name : str
        Name of the dataset

    Returns
    -------
    Tuple
        Slice of the last chunk in the raw dataset that is used by a virtual dataset
    """
    raw_data: Dataset = f["_version_data"][name]["raw_data"]
    chunks: tuple = raw_data.attrs["chunks"]
    if "last_element" in raw_data.attrs:
        last_chunk_start = raw_data.shape[0] - chunks[0]
        return Tuple(
            Slice(
                last_chunk_start,
                last_chunk_start + raw_data.attrs["last_element"],
            ),
            *(Slice(0, i) for i in raw_data.shape[1:]),
        )
    raise ValueError("Cannot find the last written element in the raw data.")


def get_space_remaining(f: File, name: str) -> int:
    """Get the number of unused elements in the last raw chunk along axis 0.

    Parameters
    ----------
    f : File
        File for which the remaining space in the last raw chunk is to be retrieved
    name : str
        Name of the dataset

    Returns
    -------
    int
        Number of unused elements in the last chunk
    """
    raw_data: Dataset = f["_version_data"][name]["raw_data"]
    last_element: Optional[int] = raw_data.attrs.get("last_element", None)
    if last_element is None:
        raise ValueError("Cannot find the last written element in the raw data.")

    return raw_data.shape[0] - last_element


def write_to_dataset(
    f: File,
    version_name: str,
    name: str,
    virtual_slice: Tuple,
    data: Union[np.ndarray, Tuple],
) -> Dict[Tuple, Tuple]:
    """Write data into new chunks at the end of the dataset.

    Parameters
    ----------
    f : File
        File where data should be written
    version_name : str
        Version name for which data is to be written
    name : str
        Name of the dataset being modified
    virtual_slice : Tuple
        Slice of the virtual dataset the data is to be written into
    data : Union[np.ndarray, Tuple]
        Data to be written. If it is a Tuple, this is a slice of the raw dataset.

    Returns
    -------
    Dict[Tuple, Tuple]
        Mapping between {slices in virtual dataset: slices in raw dataset} which were
        written by this function.
    """
    raw_data: Dataset = f["_version_data"][name]["raw_data"]
    chunk_size = raw_data.chunks[0]

    if isinstance(data, np.ndarray) and raw_data.dtype != data.dtype:
        raise ValueError(
            f"dtype of raw data ({raw_data.dtype}) does not match data to append "
            f"({data.dtype})"
        )

    slices: Dict[Tuple, Tuple] = {}
    with Hashtable(f, name) as hashtable:
        for data_slice, vchunk in zip(
            partition(data, raw_data.chunks),
            partition(virtual_slice, raw_data.chunks),
        ):
            arr = data[data_slice.raw]
            data_hash = hashtable.hash(arr)

            if data_hash in hashtable:
                slices[vchunk] = hashtable[data_hash]
            else:
                new_chunk_axis_size = raw_data.shape[0] + len(data_slice.args[0])

                rchunk = Tuple(
                    Slice(
                        raw_data.shape[0],
                        new_chunk_axis_size,
                    ),
                    *[Slice(None, None) for _ in raw_data.shape[1:]],
                )

                # Resize the dataset to include a new chunk
                raw_data.resize(raw_data.shape[0] + chunk_size, axis=0)

                # Map the virtual chunk to the raw data chunk
                slices[vchunk] = rchunk

                # Map the data hash to the raw data chunk
                hashtable[data_hash] = rchunk

                # Set the value of the raw data chunk to the chunk of the new data being written
                raw_data[rchunk.raw] = data[data_slice.raw]

                # Update the last element attribute of the raw dataset
                raw_data.attrs["last_element"] = rchunk.args[0].stop

    return slices


def get_previous_version_slices(
    f: File,
    version_name: str,
    name: str,
) -> Dict[Tuple, Tuple]:
    """Get the slices from the previous version.

    Ensures the slices are sorted in the order they appear along axis 0 of the virtual
    dataset.

    Parameters
    ----------
    f : File
        File where the slices are to be retrieved
    version_name : str
        Name of the version for which the previous version slices are to be retrieved
    name : str
        Name of the dataset

    Returns
    -------
    Dict[Tuple, Tuple]
        Mapping between {indices in virtual dataset: indices in raw dataset} which
        make up the virtual dataset of the previous version.
    """
    prev_version = get_previous_version(f, version_name, name)
    slices = []

    # Due to a bug in Group.create_virtual_dataset, empty virtual datasets are not actually
    # virtual. See https://github.com/h5py/h5py/issues/1660 for the relevant discussion.
    # Here we first check if it's virtual to avoid calling virtual_sources if it isn't.
    if prev_version.is_virtual:
        for source in prev_version.virtual_sources():
            slices.append(
                (spaceid_to_slice(source.vspace), spaceid_to_slice(source.src_space))
            )

    return dict(sorted(slices, key=lambda s: s[0].args[0].start))


def get_previous_version(
    f: File,
    version_name: str,
    name: str,
) -> Dataset:
    versions: Group = f["_version_data"]["versions"]
    prev_version_name: str = versions[version_name].attrs["prev_version"]
    return versions[prev_version_name][name]


def get_previous_version_shape(
    f: File,
    version_name: str,
    name: str,
) -> tuple[int, ...]:
    prev_version = get_previous_version(f, version_name, name)
    if prev_version.shape is not None:
        return prev_version.shape

    raise ValueError("Shape of a dataset is None")


def split_across_unused(
    f: File,
    name: str,
    arr: np.ndarray,
) -> SplitResult:
    """Split arr into a part that fits into unused space, and the part that doesn't.

    If there is any space remaining in the last chunk in the raw_data, arr will be
    broken into a piece that fits in that last chunk. If arr is larger than the
    remaining unused space, there will also be a piece that needs to be written into
    a new chunk.

    Parameters
    ----------
    f : File
        File to operate on
    name : str
        Name of the dataset to split arr into the last chunk of
    arr : np.ndarray
        Numpy array to try to cram into the final chunk

    Returns
    -------
    SplitWhatFitsResult
        Result of what fits into the last chunk of the raw data
    """
    raw_data: Dataset = f["_version_data"][name]["raw_data"]

    # Find the last used chunk of the raw data, and the empty space still left in the
    # last used chunk
    raw_last_used_chunk = last_raw_used_chunk(f, name)

    # Append as many as we can; that's either the amount that fits in the remaining
    # space, or the size of the array to be appended, whichever is smaller
    n_to_append = min(get_space_remaining(f, name), arr.shape[0])

    # Break the data into the part that will fit in the last raw chunk,
    # and the part that will need to be written to new chunks.
    vslice_to_append = Tuple(
        Slice(None, n_to_append),
        *[Slice(None, None) for _ in arr.shape[1:]],
    )
    vslice_remaining = Tuple(
        Slice(n_to_append, None),
        *[Slice(None, None) for _ in arr.shape[1:]],
    )
    arr_to_append = arr[vslice_to_append.raw]
    arr_to_write = arr[vslice_remaining.raw]

    # Compute the new size of the last used chunk
    new_raw_last_chunk = Tuple(
        Slice(
            raw_last_used_chunk.args[0].start,
            raw_last_used_chunk.args[0].stop + arr_to_append.shape[0],
        ),
        *[Slice(None, None) for _ in raw_data.shape[1:]],
    )

    # Since we're appending to the last chunk, we need to grab the last
    # chunk, append the data, then hash it.
    new_raw_last_chunk_data = np.concatenate(
        (
            raw_data[raw_last_used_chunk.raw],
            arr_to_append,
        )
    )

    return SplitResult(
        arr_to_append,
        arr_to_write,
        new_raw_last_chunk,
        new_raw_last_chunk_data,
    )


def get_updated_chunks(
    slices: Dict[Tuple, Tuple],
    index: Tuple,
    arr: np.ndarray,
    shape: tuple[int, ...],
    raw_data: Dataset,
) -> Dict[Tuple, Union[Tuple, np.ndarray]]:
    """Get the new chunks of the virtual dataset after arr is written to index.

    Parameters
    ----------
    slices : Dict[Tuple, Union[Tuple, np.ndarray]]
        Mapping between existing {slices in virtual dataset: slices in raw dataset}.
        Must be of size chunk_size along axis 0.
    index : Tuple
        (Contiguous) virtual indices for which arr is to be set
    arr : np.ndarray
        Values to set at `index` indices in the virtual dataset
    raw_data : Dataset
        Raw data

    Returns
    -------
    Dict[Tuple, Union[Tuple, np.ndarray]]
        Mapping between {slices in virtual dataset: slices in raw dataset}
        _after_ the data is written (which will be done elsewhere, in
        write_dataset_chunks). These slices are of size chunk_size.
    """
    new_chunks = {}

    # Get new chunks that `index` touches that aren't in the slices dict
    chunker = ChunkSize(raw_data.chunks)
    for vchunk in chunker.as_subchunks(index, shape):
        if vchunk not in slices:
            arr_overlap, relative_overlap = get_vchunk_overlap(vchunk, index)

            # Overwrite the part of the chunk that `index` touches with the
            # corresponding part of `arr`
            data = np.full(
                get_shape(vchunk, shape),
                fill_value=raw_data.fillvalue,
                dtype=raw_data.dtype,
            )
            data[relative_overlap.raw] = arr[arr_overlap.raw]
            new_chunks[vchunk] = data

    # Handle preexisting slices
    for vchunk, rchunk in slices.items():
        arr_overlap, relative_overlap = get_vchunk_overlap(vchunk, index)

        if arr_overlap.isempty():
            # There's no overlap between the index being set and this chunk;
            # reuse the existing raw slice or data
            new_chunks[vchunk] = rchunk
        else:
            if isinstance(rchunk, Tuple):
                # The index to set (partially) overlaps a chunk that points
                # to the raw dataset; copy the raw data, then overwrite the
                # indices that overlap with the new data
                overlap_data = raw_data[rchunk.raw]
                overlap_data[relative_overlap.raw] = arr[arr_overlap.raw]
                new_chunks[vchunk] = overlap_data
            elif isinstance(rchunk, np.ndarray):
                # The index to set (partially) overlaps a chunk that was
                # set in a different InMemoryDataset.__setitem__ call;
                # take the previous __setitem__ data, and overwrite the
                # indices that overlap with the new data
                overlap_data = rchunk
                overlap_data[relative_overlap.raw] = arr[arr_overlap.raw]
                new_chunks[vchunk] = overlap_data
            elif isinstance(rchunk, AppendChunk):
                # The index to set (partially) overlaps a chunk that was
                # appended to in a different InMemoryDataset.append call;
                # For certain cases where the __setitem__ hits only the
                # indices that are being appended, it might be possible
                # to modify the AppendChunk in place. However this behavior
                # will be harder to maintain and probably not necessary. For
                # now just blow out the AppendChunk and write a new chunk.
                overlap_data = rchunk.new_raw_last_chunk_data
                overlap_data[relative_overlap.raw] = arr[arr_overlap.raw]
                breakpoint()
                new_chunks[vchunk] = overlap_data
            elif isinstance(rchunk, WriteChunk):
                raise NotImplementedError

    return new_chunks


def arr_from_chunks(chunks, raw_data, index, shape) -> np.ndarray:
    """Get the resulting array that spans `index` from the given chunks.

          +---index.as_subindex(vchunk)----+  <-- Referenced with respect to beginning of vchunk
          |                                |
          +------------ vchunk ------------+
          |                                |
        +------------ arr -------------------------+
        |                                          |
        +-+-------- index -----------------+-------+
          |                                |
          +----vchunk.as_subindex(index)---+  <-- Referenced with respect to beginning of index


    Parameters
    ----------
    chunks :

    raw_data :

    index :

    shape :


    Returns
    -------
    np.ndarray
        Note that if `chunks` doesn't cover all parts of `index`, unset values will
        be filled with `raw_data.fillvalue`, NOT what is actually in the raw data
    """
    index = to_slice_tuple(index)
    arr = np.full(get_shape(index, shape), raw_data.fillvalue, dtype=raw_data.dtype)

    for vchunk, rchunk in chunks.items():
        arr_overlap, relative_overlap = get_vchunk_overlap(vchunk, index)

        # |__________|         vchunk
        #         |________|   index
        #         ^  ^
        # overlap of index and vchunk in vspace; find overlap in rspace,
        # set the subset of arr indices to the rspace subset that overlaps
        if not arr_overlap.isempty():
            if isinstance(rchunk, Tuple):
                # Ensure the raw chunk is a Tuple of Slice instances
                rchunk = to_slice_tuple(rchunk)
                raw_index = to_raw_index(rchunk, relative_overlap)
                arr[arr_overlap.raw] = raw_data[raw_index.raw]
            elif isinstance(rchunk, np.ndarray):
                arr[arr_overlap.raw] = rchunk[relative_overlap.raw]
            elif isinstance(rchunk, AppendChunk):
                arr[arr_overlap.raw] = rchunk.new_raw_last_chunk_data[
                    relative_overlap.raw
                ]

    return arr
