from __future__ import annotations

import datetime
import logging
import os
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from h5py import Dataset, VirtualLayout, h5s, h5z
from h5py._hl.filters import guess_chunk
from h5py._hl.selections import select
from h5py._selector import Selector
from ndindex import ChunkSize, Slice, Tuple, ndindex
from numpy.testing import assert_array_equal

from versioned_hdf5.h5py_compat import HAS_NPYSTRINGS
from versioned_hdf5.hashtable import Hashtable
from versioned_hdf5.typing_ import DEFAULT, Default

if TYPE_CHECKING:
    from versioned_hdf5.wrappers import FiltersMixin

DEFAULT_CHUNK_SIZE = 2**12
DATA_VERSION = 4
# data_version 2 has broken hashtables, always need to rebuild
# data_version 3 hash collisions for string arrays which, when concatenated,
# give the same string
CORRUPT_DATA_VERSIONS = frozenset([2, 3])


def is_vstring_dtype(dtype: np.dtype) -> bool:
    """Return True if the dtype is a variable length string dtype,
    either a NpyString (a.k.a. StringDType) or an h5py object string;
    False otherwise.
    """
    metadata = dtype.metadata or ()
    return (
        # NpyStrings
        HAS_NPYSTRINGS
        and dtype.kind == "T"
        # h5py object strings
        or "vlen" in metadata
        or "h5py_encoding" in metadata
    )


def are_compatible_dtypes(a: np.dtype, b: np.dtype) -> bool:
    """Return True if the dtypes are compatible.
    Compatible dtypes are those that are either equal or both variable length strings.
    """
    return a == b or is_vstring_dtype(a) and is_vstring_dtype(b)


def check_compatible_dtypes(a: np.dtype, b: np.dtype) -> None:
    """Raise if the dtypes are not compatible.
    Compatible dtypes are those that are either equal or both variable length strings.
    """
    if not are_compatible_dtypes(a, b):
        raise ValueError(f"dtypes are not compatible ({a} != {b})")


def initialize(f):
    from .versions import TIMESTAMP_FMT

    version_data = f.create_group("_version_data")
    versions = version_data.create_group("versions")
    versions.create_group("__first_version__")
    versions.attrs["current_version"] = "__first_version__"
    ts = datetime.datetime.now(datetime.timezone.utc)
    versions["__first_version__"].attrs["timestamp"] = ts.strftime(TIMESTAMP_FMT)
    versions.attrs["data_version"] = DATA_VERSION


@dataclass
class Filters:
    """Filters keyword arguments for create_dataset and modify_metadata.

    Not to be confused with h5py.Dataset._filters, which is a dict in the format

    {
        <compression name>: <compression_opts>,  # compression=None if key is missing
        scaleoffset: <mangled>,  # scaleoffset=None if key is missing
        shuffle: <mangled>,  # shuffle=True if present; False if key is missing
        fletcher32: None,  # fletcher32=True if present; False if key is missing
    }

    Note: you should not assume that the above is the complete content
    of _dataset._filters. There may be custom/unknown filters present.
    """

    # See matching class wrappers.FiltersMixin
    # compression will typically be str, int (for raw filter ID), or a hdf5plugin object
    compression: Any | None | Default = DEFAULT
    compression_opts: Any | None | Default = DEFAULT
    scaleoffset: int | None | Default = DEFAULT
    shuffle: bool | Default = DEFAULT
    fletcher32: bool | Default = DEFAULT

    def as_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for create_dataset."""
        return {k: v for k, v in self.__dict__.items() if v is not DEFAULT}

    @staticmethod
    def from_dataset(ds: Dataset | FiltersMixin) -> Filters:
        """Reverse engineer create_dataset kwargs from an h5py.Dataset."""
        compression = ds.compression
        compression_opts = ds.compression_opts

        # Hack for custom compression filters.
        # FIXME This should be fixable upstream. h5py would need to expose a hook for
        # hdf5plugin / pytables to let them declare that a filter ID is a
        # compression filter.
        if compression is None and isinstance(ds, Dataset):
            # From hdf5plugin._filters. Can't just try-import hdf5plugin because the
            # same filter IDs are also defined by pytables.
            CUSTOM_COMPRESSION_FILTERS = (
                32001,  # Blosc
                32026,  # Blosc2
                307,  # Bzip2
                32004,  # LZ4
                32008,  # Bitshuffle
                32013,  # ZFP
                32015,  # Zstandard
                32017,  # SZ
                32024,  # SZ3
                32018,  # FCIDECOMP
                32028,  # SPERR
            )

            for filter_id in CUSTOM_COMPRESSION_FILTERS:
                try:
                    compression_opts = ds._filters[str(filter_id)]
                    compression = filter_id
                    break
                except KeyError:
                    pass

            if compression is None:
                # If we're using a bespoke compression, there's no way of knowing
                # whether an unknown filter is a valid compression or some other
                # kind of filter, so we issue a warning about assuming that it is
                # the dataset's compression.
                for k, v in ds._filters.items():
                    try:
                        k = int(k)
                    except ValueError:
                        continue
                    compression, compression_opts = k, v
                    logging.warning(
                        "No default compression detected in this dataset. "
                        f"Guessed {compression=} {compression_opts=} "
                        f"from {ds._filters=}."
                    )
                    break

        return Filters(
            compression=compression,
            compression_opts=compression_opts,
            scaleoffset=ds.scaleoffset,
            shuffle=ds.shuffle,
            fletcher32=ds.fletcher32,
        )

    def overrides(self, other: Filters) -> bool:
        """Return True if there are any filters that are explicitly set and
        differ between self and other; False otherwise.
        """
        for k, v1 in self.__dict__.items():
            v2 = getattr(other, k)
            if v1 is not DEFAULT and v2 is not DEFAULT and v1 != v2:
                return True
        return False


def create_base_dataset(
    f,
    name,
    *,
    shape=None,
    data=None,
    dtype=None,
    chunks=None,
    fillvalue=None,
    filters: Filters | None = None,
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
    dtype = np.dtype(dtype)
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
    kwargs = filters.as_kwargs() if filters is not None else {}
    dataset = group.create_dataset(
        "raw_data",
        shape=(0,) + chunks[1:],
        chunks=tuple(chunks),
        maxshape=(None,) + chunks[1:],
        dtype=dtype,
        fillvalue=fillvalue,
        **kwargs,
    )
    dataset.attrs["chunks"] = chunks
    return write_dataset(f, name, data, chunks=chunks)


def write_dataset(
    f,
    name,
    data,
    chunks=None,
    dtype=None,
    fillvalue=None,
    filters: Filters | None = None,
):
    if name not in f["_version_data"]:
        return create_base_dataset(
            f,
            name,
            data=data,
            dtype=dtype,
            chunks=chunks,
            fillvalue=fillvalue,
            filters=filters,
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
        check_compatible_dtypes(dtype, ds.dtype)

    if filters is not None and filters.overrides(Filters.from_dataset(ds)):
        available_filters = textwrap.indent(
            "\n".join(str(filter) for filter in get_available_filters()), "  "
        )
        raise ValueError(
            "Compression options and other filters can only be specified for the first "
            "version of a dataset.\n"
            f"Dataset: {name}\n"
            f"Current filters: {ds._filters}\n"
            f"New filters: {filters}\n"
            f"Available hdf5 compression types:\n{available_filters}"
        )

    if (
        fillvalue is not None
        and fillvalue != ds.fillvalue
        # For variable length string dtypes, ds.fillvalue will be None in
        # this case (see create_virtual_dataset() below)
        and not is_vstring_dtype(ds.dtype)
    ):
        raise ValueError(f"fillvalues do not match ({fillvalue} != {ds.fillvalue})")

    check_compatible_dtypes(data.dtype, ds.dtype)
    # TODO: Handle more than one dimension
    old_shape = ds.shape
    slices: dict[Slice, Tuple] = {}
    slices_to_write: dict[Tuple, Slice] = {}
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
        "  %s: New chunks written: %d; Number of chunks reused: %d",
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
    slices_to_write: dict[Slice, Tuple] | None = None,
    data_to_write: dict[Slice, np.ndarray] | None = None,
) -> None:
    """Check that the data from the hashed slice matches the data to be written.

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
    slices_to_write : dict[slice, tuple] | None
        Dict of slices which will be written. Maps slices that will exist in the
        raw_data once the write is complete to slices of the dataset that is being
        written.
    data_to_write : dict[slice, tuple] | None
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
    #     raw_data[some_slice] = chunk_being_written[another_slice]  # noqa: ERA001
    # the data that gets written is bytes. So in certain cases, just calling
    # assert_array_equal doesn't work. Instead, we encode each element to bytes first.
    def normalize_chunk(chunk):
        # TODO object dtype and StringDType can be accelerated in C/Cython
        # See also hashtable.Hashtable.hash()
        if chunk.dtype.kind == "T":
            return _convert_to_bytes(chunk.astype("O"))
        if chunk.dtype.kind == "O":
            return _convert_to_bytes(chunk)
        return chunk

    to_be_written = normalize_chunk(chunk_being_written)
    to_be_reused = normalize_chunk(reused_chunk)

    try:
        assert_array_equal(to_be_reused, to_be_written, strict=True)
    except AssertionError as e:
        raise ValueError(
            f"Hash {data_hash!r} of existing data chunk {reused_chunk!r} "
            f"matches the hash of new data chunk {chunk_being_written!r}, "
            "but data does not."
        ) from e


@np.vectorize
def _convert_to_bytes(x: str | bytes) -> bytes:
    """Convert each element in the array to bytes.

    Each element in the array is assumed to be the same type, even if the input is an
    object dtype array.

    Parameters
    ----------
    arr : np.ndarray
        Array to be converted; no conversion is done if the elements are already bytes.

    Returns
    -------
    np.ndarray
        Object dtype array filled with elements of type bytes
    """
    return x.encode("utf-8") if isinstance(x, str) else x


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

        # Mapping from slices in the raw dataset after this write is complete to ndarray
        # chunks of the new data which will be written
        data_to_write = {}
        for chunk, data_s in data_dict.items():
            if isinstance(data_s, (slice, tuple, Tuple, Slice)):
                slices[chunk] = ndindex(data_s)
            else:
                check_compatible_dtypes(data_s.dtype, raw_data.dtype)

                data_hash = hashtable.hash(data_s)

                if data_hash in hashtable:
                    hashed_slice = hashtable[data_hash]
                    slices[chunk] = hashed_slice

                    if validate_reused_chunks:
                        _verify_new_chunk_reuse(
                            raw_dataset=raw_data,
                            new_data=data_s,
                            data_hash=data_hash,
                            hashed_slice=hashed_slice,
                            chunk_being_written=data_s,
                            data_to_write=data_to_write,
                        )

                    chunks_reused += 1

                else:
                    idx = hashtable.largest_index
                    raw_slice = Slice(
                        idx * chunk_size, idx * chunk_size + data_s.shape[0]
                    )
                    slices[chunk] = raw_slice
                    hashtable[data_hash] = raw_slice
                    data_to_write[raw_slice] = data_s

        new_chunks = hashtable.largest_index

    assert None not in slices.values()
    old_shape = raw_data.shape
    raw_data.resize((old_shape[0] + len(data_to_write) * chunk_size,) + chunks[1:])
    for raw_slice, data_s in data_to_write.items():
        c = (raw_slice.raw,) + tuple(slice(0, i) for i in data_s.shape[1:])
        raw_data[c] = data_s

    logging.debug(
        "  %s: New chunks written: %d; Number of chunks reused: %d",
        name,
        new_chunks - old_chunks,
        chunks_reused,
    )

    return slices


def create_virtual_dataset(
    f, version_name, name, shape, slices, attrs=None, fillvalue=None
):
    """Create a new virtual dataset by stitching the chunks of the
    raw dataset together, as indicated by the slices dict.

    See Also
    --------
    _recreate_virtual_dataset
    """
    raw_data = f["_version_data"][name]["raw_data"]
    raw_data_shape = raw_data.shape
    raw_data_name = raw_data.name.encode("utf-8")

    if len(raw_data) == 0:
        layout = VirtualLayout(shape=tuple([0 for _ in shape]), dtype=raw_data.dtype)
    else:
        layout = VirtualLayout(shape, dtype=raw_data.dtype)
        layout._src_filenames.add(b".")
        space = h5s.create_simple(shape)
        selector = Selector(space)

        # Chunks in the raw dataset are expanded along the first dimension only.
        # Since the chunks are pointed to by virtual datasets, it doesn't make
        # sense to expand the chunks in the raw dataset along multiple dimensions
        # (the true layout of the chunks in the raw dataset is irrelevant).
        for c, s0 in slices.items():
            if len(c.args[0]) != len(s0):
                raise ValueError(f"Inconsistent slices dictionary ({c.args[0]}, {s0})")
            if c.isempty():
                continue

            s = (s0.reduce().raw, *(slice(0, len(ci), 1) for ci in c.args[1:]))

            # This is equivalent to `layout[c] = vs[s]`,
            # but faster because vs[s] deep-copies vs, which is slow.
            vs_sel = select(raw_data_shape, s, dataset=None)
            sel = selector.make_selection(c.raw)
            layout.dcpl.set_virtual(sel.id, b".", raw_data_name, vs_sel.id)

    dtype_meta = raw_data.dtype.metadata
    if dtype_meta and ("vlen" in dtype_meta or "h5py_encoding" in dtype_meta):
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


def get_available_filters() -> Iterator[int]:
    """Retrieve all of the registered h5py filters.

    Returns
    -------
    Iterator[int]
        Filter ID numbers; each filter has a dedicated ID - see
        the docs for the particular filter being used for more information
        about these
    """
    for i in range(65536):
        if h5z.filter_avail(i):
            yield i
