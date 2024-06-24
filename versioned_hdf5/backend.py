import logging
import os
from typing import Dict, Optional, Union

import numpy as np
from h5py import Dataset, File, VirtualLayout, VirtualSource, h5s
from h5py._hl.filters import guess_chunk
from ndindex import ChunkSize, Slice, Tuple, ndindex
from numpy.testing import assert_array_equal

from .hashtable import Hashtable
from .slicetools import spaceid_to_slice

DEFAULT_CHUNK_SIZE = 2**12
DATA_VERSION = 4
# data_version 2 has broken hashtables, always need to rebuild
# data_version 3 hash collisions for string arrays which, when concatenated, give the same string
CORRUPT_DATA_VERSIONS = frozenset([2, 3])


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
        if data is not None and (
            np.prod(shape, dtype=np.ulonglong)
            != np.prod(data.shape, dtype=np.ulonglong)
        ):
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

    validate_reused_chunks = os.environ.get(
        "ENABLE_CHUNK_REUSE_VALIDATION", "false"
    ).lower() in ("1", "true")

    with Hashtable(f, name) as hashtable:
        old_chunks = hashtable.largest_index
        chunks_reused = 0

        if len(data.shape) != 0:
            for data_slice in ChunkSize(chunks).indices(data.shape):
                data_s = data[data_slice.raw]
                data_hash = hashtable.hash(data_s)

                if data_hash in hashtable:
                    hashed_slice = hashtable[data_hash]
                    slices[data_slice] = hashed_slice

                    if validate_reused_chunks:
                        _verify_new_chunk_reuse(
                            raw_dataset=ds,
                            new_data=data,
                            data_hash=data_hash,
                            hashed_slice=hashed_slice,
                            chunk_being_written=data_s,
                            slices_to_write=slices_to_write,
                        )

                    chunks_reused += 1

                else:
                    idx = hashtable.largest_index
                    raw_slice = Slice(
                        idx * chunk_size, idx * chunk_size + data_s.shape[0]
                    )
                    slices[data_slice] = raw_slice
                    hashtable[data_hash] = raw_slice
                    slices_to_write[raw_slice] = data_slice

            ds.resize((old_shape[0] + len(slices_to_write) * chunk_size,) + chunks[1:])
            for raw_slice, data_slice in slices_to_write.items():
                data_s = data[data_slice.raw]
                idx = Tuple(raw_slice, *[slice(0, i) for i in data_s.shape[1:]])
                ds[idx.raw] = data[data_slice.raw]

        new_chunks = hashtable.largest_index

    logging.debug(
        "  %s: " "New chunks written: %d; " "Number of chunks reused: %d",
        name,
        new_chunks - old_chunks,
        chunks_reused,
    )

    return slices


def _verify_new_chunk_reuse(
    raw_dataset: Dataset,
    new_data: np.ndarray,
    data_hash: bytes,
    hashed_slice: Slice,
    chunk_being_written: np.ndarray,
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
    raw_dataset : Dataset
        Raw Dataset that already exists in the file
    new_data : np.ndarray
        New data that we are writing
    data_hash : bytes
        Hash of the new data chunk
    hashed_slice : Slice
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
            hashed_slice, *[slice(0, size) for size in chunk_being_written.shape[1:]]
        )
        reused_chunk = raw_dataset[reused_slice.raw]

    # In some cases type coercion can happen during the write process even if the dtypes
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

    try:
        assert_array_equal(reused_chunk, to_be_written)
    except AssertionError as e:
        raise ValueError(
            f"Hash {data_hash} of existing data chunk {reused_chunk} "
            f"matches the hash of new data chunk {chunk_being_written}, "
            "but data does not."
        ) from e


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


def write_dataset_chunks(f, name, data_dict):
    """
    data_dict should be a dictionary mapping chunk_size index to either an
    array for that chunk, or a slice into the raw data for that chunk

    """
    if name not in f["_version_data"]:
        raise NotImplementedError(
            "Use write_dataset() if the dataset does not yet exist"
        )

    raw_data = f["_version_data"][name]["raw_data"]
    chunks = tuple(raw_data.attrs["chunks"])
    chunk_size = chunks[0]

    validate_reused_chunks = os.environ.get(
        "ENABLE_CHUNK_REUSE_VALIDATION", "false"
    ).lower() in ("1", "true")

    with Hashtable(f, name) as hashtable:
        old_chunks = hashtable.largest_index
        chunks_reused = 0

        slices = {i: None for i in data_dict}
        chunks_appended_to = set()

        # Mapping from slices in the raw dataset after this write is complete to ndarray
        # chunks of the new data which will be written
        data_to_write = {}
        for chunk, data_s in data_dict.items():
            if isinstance(data_s, (slice, tuple, Tuple, Slice)):
                slices[chunk] = ndindex(data_s)
            elif isinstance(data_s, AppendData):
                # Write chunks to append to the raw data without writing a new chunk
                if data_s.array.dtype != raw_data.dtype:
                    raise ValueError(
                        f"dtypes do not match ({data_s.array.dtype} != {raw_data.dtype})"
                    )

                # The extant raw indices are of shape `chunks`; chunks at the edge of
                # multidimensional datasets can have `fillvalue` elements in the parts
                # of the chunk that are outside the area referenced by the virtual
                # dataset. Therefore the array to append can differ in shape along
                # axes >1, and must be expanded to fill those other dimensions that
                # are not along the append axis in order to concatenate.
                raw_chunk_data = raw_data[data_s.extant_rindex.raw]

                # Pad out the array to append. Use the shape along axis 0 of the
                # array to append (since that can be anything), but for the other
                # dimensions use the shape of the underlying data chunk.
                array_padded = np.full(
                    (data_s.array.shape[0], *raw_chunk_data.shape[1:]),
                    fill_value=raw_data.fillvalue,
                )
                arr_index = Tuple(*[Slice(0, dim) for dim in data_s.array.shape])
                array_padded[arr_index.raw] = data_s.array

                # Calculate a new hash for this chunk using the extant data and the
                # data to be appended
                concatenated_raw_chunk = np.concatenate((raw_chunk_data, array_padded))

                # Ensure that the new raw chunk data has the correct dimensions
                # along every axis except for the chunking axis (it's fine to have
                # partially full raw chunks)
                assert concatenated_raw_chunk.shape[1:] == chunks[1:]

                # This is the location where the data to append will be written; first
                # compute it here to check whether the raw space is referenced by any
                # other version of the dataset.
                raw_chunk = Tuple(
                    Slice(
                        data_s.extant_rindex.stop,
                        data_s.extant_rindex.stop + data_s.array.shape[0],
                    ),
                    *[Slice(0, dim) for dim in data_s.array.shape[1:]],
                )

                # Multidimensional datasets can have multiple chunks which reference the
                # same underlying chunk. If that happens, only one AppendChunk can fill
                # the remaining space in the underlying chunk. If a chunk has already
                # been appended to, bail out and write a new chunk.
                if data_s.extant_rindex in chunks_appended_to or _is_occupied_space(
                    f, name, raw_chunk
                ):
                    chunks_reused += update_hashtable(
                        raw_data,
                        concatenated_raw_chunk,
                        hashtable,
                        slices,
                        chunk,
                        chunk_size,
                        data_to_write,
                        validate_reused_chunks,
                    )
                    continue

                # Expand the extant raw index to include the concatenated data
                raw_slice = Slice(
                    data_s.extant_rindex.start,
                    data_s.extant_rindex.start + concatenated_raw_chunk.shape[0],
                )

                chunks_appended_to.add(data_s.extant_rindex)
                data_hash = hashtable.hash(concatenated_raw_chunk)
                if data_hash in hashtable:
                    hashed_slice = hashtable[data_hash]
                    slices[chunk] = hashed_slice

                    _verify_new_chunk_reuse(
                        raw_data=raw_data,
                        new_data=data_s.array,
                        data_hash=data_hash,
                        hashed_slice=hashed_slice,
                        chunk_being_written=data_s.array,
                        data_to_write=data_to_write,
                    )

                    chunks_reused += 1
                else:
                    # Expand the extant raw index to include the concatenated data
                    raw_slice = Slice(
                        data_s.extant_rindex.start,
                        data_s.extant_rindex.start + concatenated_raw_chunk.shape[0],
                    )
                    slices[chunk] = raw_slice
                    hashtable[data_hash] = raw_slice

                    # Decrement the number of stored chunks because we aren't actually
                    # writing a new chunk.
                    hashtable.largest_index -= 1

                    # Write the data here, because if you add it to data_to_write it
                    # will actually be written into a new chunk, changing the shape
                    # of the raw data.
                    raw_data[raw_chunk.raw] = data_s.array

            else:
                chunks_reused += update_hashtable(
                    raw_data,
                    data_s,
                    hashtable,
                    slices,
                    chunk,
                    chunk_size,
                    data_to_write,
                    validate_reused_chunks,
                )

        new_chunks = hashtable.largest_index

    assert None not in slices.values()
    old_shape = raw_data.shape
    raw_data.resize((old_shape[0] + len(data_to_write) * chunk_size,) + chunks[1:])
    for raw_slice, data_s in data_to_write.items():
        c = (raw_slice.raw,) + tuple(slice(0, i) for i in data_s.shape[1:])
        raw_data[c] = data_s

    logging.debug(
        "  %s: " "New chunks written: %d; " "Number of chunks reused: %d",
        name,
        new_chunks - old_chunks,
        chunks_reused,
    )

    return slices


def _is_occupied_space(
    f: File,
    name: str,
    index: Slice,
) -> bool:
    """Check whether the target virtual index references unused space.

    Parameters
    ----------
    f : File
        File containing the various versions of the dataset
    name : str
        Name of the dataset to check
    target_vindex : Tuple
        Target virtual index to be checked for existing data

    Returns
    -------
    bool
        True if the space referenced by target_vindex is occupied, False
        otherwise
    """
    for version in f["_version_data/versions"]:
        # Ignore the root version or the current (uncommitted) version
        if version == "__first_version__" or name not in f["_version_data/versions"]:
            continue

        for vindex, rindex in get_data_map(f, name, version).items():
            assert len(vindex.args[0]) != len(rindex.args[0])

            try:
                index.as_subindex(rindex)
            except ValueError:
                return True

    return False


def update_hashtable(
    raw_data: Dataset,
    data_s: np.ndarray,
    hashtable: Hashtable,
    slices: Dict[Tuple, Slice],
    chunk: Tuple,
    chunk_size: int,
    data_to_write: Dict[Slice, np.ndarray],
    validate_reused_chunks: bool,
) -> int:
    """Update the hashtable with the new data hash, if necessary.

    If the hash of `data_s` already exists in the `hashtable`, write the hashed slice
    into the `slices`. Otherwise, write a new entry into the `hashtable` for the
    hashed data, then update `slices`, and update `data_to_write` with the data
    to be written.

    The `hashtable`, `slices`, and `data_to_write` are all mutated by this
    function.

    Parameters
    ----------
    raw_data : Dataset
        Underlying raw dataset
    data_s : np.ndarray
        Chunk of data to possibly write to the underlying dataset
    hashtable : Hashtable
        Hashtable containing a mapping of raw data to slices in the raw data
    slices : Dict[Tuple, Slice]
        Mapping between virtual indices to underlying Slices in the raw data
    chunk : Tuple
        Virtual index being written to
    chunk_size : int
        Size of the chunks of the dataset along the first axis
    data_to_write : Dict[Slice, np.ndarray]
        Mapping of slices in the raw dataset to data to be written to those
        slices
    validate_reused_chunks : bool
        If True, check that chunks that are reused have data which matches the
        data to be written

    Returns
    -------
    int
        The number of chunks reused from the underlying raw dataset
    """
    if data_s.dtype != raw_data.dtype:
        raise ValueError(f"dtypes do not match ({data_s.dtype} != {raw_data.dtype})")

    data_hash = hashtable.hash(data_s)

    if data_hash in hashtable:
        raw_slice = hashtable[data_hash]
        slices[chunk] = raw_slice

        if validate_reused_chunks:
            _verify_new_chunk_reuse(
                raw_dataset=raw_data,
                new_data=data_s,
                data_hash=data_hash,
                hashed_slice=raw_slice,
                chunk_being_written=data_s,
                data_to_write=data_to_write,
            )

        reused_chunks = 1

    else:
        idx = hashtable.largest_index
        raw_slice = Slice(idx * chunk_size, idx * chunk_size + data_s.shape[0])
        slices[chunk] = raw_slice
        hashtable[data_hash] = raw_slice
        data_to_write[raw_slice] = data_s
        reused_chunks = 0

    return reused_chunks


def create_virtual_dataset(
    f, version_name, name, shape, slices, attrs=None, fillvalue=None
):
    from h5py._hl.selections import select
    from h5py._hl.vds import VDSmap

    raw_data = f["_version_data"][name]["raw_data"]
    raw_data_shape = raw_data.shape
    slices = {c: s.reduce() for c, s in slices.items()}

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

    virtual_data = f["_version_data/versions"][version_name].create_virtual_dataset(
        name, layout, fillvalue=fillvalue
    )

    if attrs:
        for k, v in attrs.items():
            virtual_data.attrs[k] = v
    virtual_data.attrs["raw_data"] = raw_data.name
    virtual_data.attrs["chunks"] = raw_data.chunks

    return virtual_data


class AppendData:
    """Container for data to be appended.

    Attributes
    ----------
    target_vindex : Indices of the virtual dataset which the data to be appended should appear
    array : Data to append (this is what is written to the raw dataset)
    extant_vindex : Virtual indices of the data which already exists in the chunk where the
         data is to be written
    extant_rindex : Raw indices of the data which already exists in the chunk where the data
         is to be written
    """

    def __init__(
        self,
        target_vindex: Tuple,
        array: np.ndarray,
        extant_vindex: Tuple,
        extant_rindex: Slice,
    ):
        """Instantiate an AppendData instance.

        Parameters
        ----------
        target_vindex : Tuple
             Indices of the virtual dataset which the data to be appended should appear
        array : np.ndarray
             Data to append (this is what is written to the raw dataset)
        extant_vindex : Tuple
             Virtual indices of the data which already exists in the chunk where the
             data is to be written
        extant_rindex : Slice
             Raw indices of the data which already exists in the chunk where the data
             is to be written
        """
        self.target_vindex = target_vindex
        self.array = array
        self.extant_vindex = extant_vindex
        self.extant_rindex = extant_rindex

    def __repr__(self):
        return f"""
        AppendChunk:
            array to append: {self.array}
            target virtual index: {self.target_vindex}
            extant data virtual index: {self.extant_vindex}
            extant data raw index: {self.extant_rindex}"""


def get_data_map(f: File, name: str, version: str) -> Dict[Tuple, Tuple]:
    """Get the Dict which maps virtual dataset sources to raw data slices.

    Parameters
    ----------
    f : File
        File containing the chunks to get for the given version
    name : str
        Name of the dataset
    version : str
        Version of the dataset for which the data map is to be retrieved

    Returns
    -------
    Dict[ndindex.Tuple, ndindex.Tuple]
        Mapping between the dataset's virtual slices and the slices of the
        underlying raw data
    """
    slices = {}
    for vs in f["_version_data/versions"][version][name].virtual_sources():
        src = spaceid_to_slice(vs.src_space)
        vir = spaceid_to_slice(vs.vspace)
        slices[vir] = src

    return slices
