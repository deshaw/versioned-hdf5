from __future__ import annotations

import datetime
import itertools
import logging
import math
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from h5py import Dataset, VirtualLayout, h5s, h5z
from h5py._hl.filters import guess_chunk
from h5py._hl.selections import select
from h5py._selector import Selector
from ndindex import ChunkSize, Slice, Tuple

from versioned_hdf5.cytools import ceil_a_over_b
from versioned_hdf5.h5py_compat import HAS_NPYSTRINGS, h5py_astype
from versioned_hdf5.hashtable import Hashtable
from versioned_hdf5.slicetools import RawDataView
from versioned_hdf5.staged_changes import StagedChangesArray
from versioned_hdf5.typing_ import DEFAULT, Default

if TYPE_CHECKING:
    from versioned_hdf5.wrappers import FiltersMixin

DEFAULT_CHUNK_SIZE = 2**12
# Amount of RAM that rewrite_dataset() can allocate as a scratch area
# This is a compromise between minimizing RAM usage and runtime.
REWRITE_BUFFER_BYTES = 2**26  # 64 MiB
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


def normalize_chunks(
    chunks: tuple[int, ...] | int | bool | None,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> tuple[int, ...]:
    """Normalize the ``chunks`` parameter of create_dataset(), guessing a sensible
    chunk size when it is not explicitly specified.
    """
    if isinstance(chunks, int) and not isinstance(chunks, bool):
        return (chunks,)
    if not isinstance(chunks, bool) and chunks is not None:
        return tuple(chunks)

    ndim = len(shape)
    assert ndim > 0  # 0d use case is caught upstream by wrappers.py
    if ndim == 1:
        return guess_chunk(shape, None, dtype.itemsize)
    raise NotImplementedError("chunks must be specified for multi-dimensional datasets")


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

    if dtype is None:
        # https://github.com/h5py/h5py/issues/1474
        dtype = data.dtype
    dtype = np.dtype(dtype)

    chunks = normalize_chunks(chunks, shape, dtype)
    group = f["_version_data"].create_group(name)

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

    with Hashtable(f, name) as hashtable:
        old_chunks = hashtable.largest_index
        chunks_reused = 0

        if data.ndim == 0 or data.size == 0:
            return {}

        for data_slice in ChunkSize(chunks).indices(data.shape):
            data_s = data[data_slice.raw]
            data_hash = hashtable.hash(data_s)

            if data_hash in hashtable:
                hashed_slice = hashtable[data_hash]
                slices[data_slice] = hashed_slice
                chunks_reused += 1

            else:
                idx = hashtable.largest_index
                raw_slice = Slice(idx * chunk_size, idx * chunk_size + data_s.shape[0])
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


def _data_v4_to_sc_hash_table(hash_table: Dataset, chunk_size0: int) -> np.ndarray:
    """Load the SHA256 digests from the on-disk hash table and convert it to the
    layout compatible with `StagedChangesArray.hash_tables`, which is indexed by the
    chunk index along axis 0 of raw_data.

    This on-the-fly conversion allows not increasing the DATA_VERSION and thus not
    having to migrate legacy datasets.
    """
    largest_index = int(hash_table.attrs["largest_index"])
    records = hash_table[:largest_index]  # Load into memory
    hashes = np.ascontiguousarray(records["hash"]).view(np.uint64)

    # The rows of the on-disk hash table are typically already sorted by raw_data
    # offset, but those of a hash table rebuilt by Hashtable.from_versions_traverse()
    # are in version order instead and may skip chunks altogether. Reorder them by
    # chunk index, leaving all-zeros (which means "no chunk here") in the gaps.
    rows = records["shape"][:, 0] // chunk_size0
    if not np.array_equal(rows, np.arange(largest_index)):
        reordered = np.zeros((int(rows.max()) + 1, 4), dtype=np.uint64)
        reordered[rows] = hashes
        hashes = reordered

    return hashes


def _sc_hash_table_to_data_v4(
    hashes: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    hash_table_dtype: np.dtype,
) -> np.ndarray:
    """Inverse conversion of `_data_v4_to_sc_hash_table`"""
    n = hashes.shape[0]
    out = np.empty(n, dtype=hash_table_dtype)
    out["hash"] = hashes.view(np.uint8).reshape(n, -1)
    out["shape"] = np.stack([starts, stops], axis=1)
    return out


def _raw_data_as_base_slab(raw_data: Dataset, dtype: np.dtype):
    """Return `raw_data`, to be used as a base slab of a StagedChangesArray of the
    given dtype.

    Variable-width strings are always stored as object dtype in raw_data, while the
    StagedChangesArray may be StringDType; wrap raw_data in a lazy view in that case.
    """
    return raw_data if dtype == raw_data.dtype else h5py_astype(raw_data, dtype)


def commit_staged_changes(
    f, name: str, staged_changes: StagedChangesArray
) -> dict[Tuple, Slice]:
    """Commit a StagedChangesArray into `raw_data` and its on-disk hash table.

    1. Load the on-disk hash table dataset that hashes all chunks of `raw_data`
       into memory
    2. Inject it as the hash table of `staged_changes.base_slabs[0]`, which is
       `raw_data`
    3. Define a callback function that mocks `numpy.empty`. The callback internally
       extends `raw_data` and returns a view to the new empty surface.
    4. Call staged_changes.commit, passing the callback above.
       This hashes all staged chunks vs. all present and past chunks in `raw_data`,
       saving the hashes of all unique staged chunks to a new np.ndarray, then
       writes to `raw_data` on disk. See docs/staged_changes.rst for details.
    5. Append the new chunks' hashes and slices to the on-disk hash table
    6. Shift staged_changes.slab_offsets for the new virtual base slab, so that
       offsets are correct for raw_data.
    7. Tail-call `staged_changes.changes`, which returns the
       `{chunk_index: raw_data slice}` dict to be passed to `create_virtual_dataset`.

    **TRANSITION NOTES**

    This function is the replacement for the legacy function `write_dataset` and the
    `Hashtable` class. At the moment of writing, the legacy path is still triggered by
    some use cases:

    New code path (commit_staged_changes + hash_slab)
        - commit on stage_version(...) context exit
        - recreate_dataset
        - modify_metadata

    Legacy code path
        - delete_versions (Hashtable, _recreate_raw_data)
        - VersionedHDF5.rebuild_hashtables (Hashtable)
    """
    sc = staged_changes
    group = f["_version_data"][name]
    raw_data = group["raw_data"]
    hash_table = group["hash_table"]
    assert sc.chunk_size == tuple(raw_data.chunks)
    chunk_size0 = sc.chunk_size[0]

    # Number of chunks that the previous versions committed to raw_data.
    # raw_data, and possibly hash_table too, can be longer than this if a previous
    # commit crashed halfway through; largest_index is updated last and is the only
    # trustworthy measure. Anything beyond it is garbage and will be overwritten.
    prev_n_chunks = int(hash_table.attrs["largest_index"])
    prev_len = prev_n_chunks * chunk_size0

    # A InMemoryDataset has exactly one base slab (raw_data); a
    # InMemoryArrayDataset or InMemorySparseDataset has none.
    assert sc.n_base_slabs in (0, 1)
    if sc.n_base_slabs == 0 and prev_n_chunks > 0:
        # DatasetWrapper hot-swapped a InMemoryDataset for an InMemorySparseDataset or
        # an InMemoryArrayDataset, or a dataset was deleted in an intermediate version
        # and then recreated, or it was created in two independent branches
        sc.slabs.insert(1, _raw_data_as_base_slab(raw_data, sc.dtype))
        sc.hash_tables.insert(1, None)
        sc.slab_indices[sc.slab_indices > 0] += 1
        sc.n_base_slabs = 1

    n_base_before = sc.n_base_slabs

    if n_base_before == 1:
        assert sc.hash_tables[1] is None
        sc.hash_tables[1] = _data_v4_to_sc_hash_table(hash_table, chunk_size0)

    def empty(shape: tuple[int, ...], dtype) -> RawDataView:
        """Mock API of np.empty. Extend raw_data and return view to the new area."""
        raw_data.resize((prev_len + shape[0], *raw_data.shape[1:]))
        return RawDataView(raw_data, prev_len, dtype)

    # Calculate hashes, deduplicate staged chunks, and write to raw_data
    sc.commit(empty=empty)

    n_appended_chunks = 0
    if sc.n_base_slabs > n_base_before:
        # At least one staged chunk is original (neither identical to a chunk
        # already on raw_data nor full of fill_value)
        # === Steps 6-10: a new base slab was appended; record its chunks on disk ===
        assert sc.n_base_slabs == n_base_before + 1
        new_slab_idx = sc.n_base_slabs
        new_hashes = sc.hash_tables[new_slab_idx]
        assert new_hashes is not None
        n_appended_chunks = new_hashes.shape[0]
        new_n_chunks = prev_n_chunks + n_appended_chunks

        # Calculate (start, stop) offsets of the chunks on raw_data
        starts = np.arange(
            prev_n_chunks * chunk_size0,
            new_n_chunks * chunk_size0,
            chunk_size0,
            dtype=np.int64,
        )
        stops = starts + chunk_size0
        if sc.shape[0] % chunk_size0:
            # Edge chunks along axis 0 are not full chunks; trim stops.
            # This matters when modify_metadata() recalculates the hashes.
            n_whole_chunks0 = sc.slab_indices.shape[0] - 1
            last_chunk_trim = chunk_size0 - sc.shape[0] + n_whole_chunks0 * chunk_size0
            new_edge_chunks_idx = (
                sc.slab_offsets[-1][sc.slab_indices[-1] == new_slab_idx] // chunk_size0
            )
            stops[new_edge_chunks_idx] -= last_chunk_trim

        # Append the new records to the on-disk hash table in a single write.
        disk = _sc_hash_table_to_data_v4(new_hashes, starts, stops, hash_table.dtype)
        hash_table.resize((new_n_chunks,))
        hash_table[prev_n_chunks:] = disk
        # Atomic update marking the successful commit (except the VDS creation).
        # In case of a crash halfway through commit, you will have the
        # raw_data or raw_data+hash_table larger than this.
        hash_table.attrs["largest_index"] = new_n_chunks

        # commit() wrote the new chunks through a RawDataView onto
        # raw_data[prev_len:], so their slab_offsets are relative to prev_len.
        # Shift them to absolute raw_data offsets and collapse the new base slab onto
        # slab 1 (raw_data).
        if n_base_before:
            new_slab_mask = sc.slab_indices == new_slab_idx
            sc.slab_indices[new_slab_mask] = 1
            sc.slab_offsets[new_slab_mask] += prev_len

        # Collapse back to a single raw_data base slab, so the committed
        # StagedChangesArray is structurally identical to a freshly-loaded
        # InMemoryDataset.
        sc.slabs = [sc.slabs[0], _raw_data_as_base_slab(raw_data, sc.dtype)]
        sc.hash_tables = [None, None]
        sc.n_base_slabs = 1

    logging.debug(
        "  %s: New chunks written: %d; Number of chunks reused: %d",
        name,
        n_appended_chunks,
        np.prod(sc.slab_indices.shape) - n_appended_chunks,
    )

    # Build the {virtual dataset index: raw_data slice} mapping
    # TODO Migrated to a Cythonized loop that reads sc.slab_offsets directly
    return {
        Tuple(*vds_slice): Slice(raw_data_slice[0])
        for vds_slice, _, raw_data_slice in sc.changes()
    }


def _chunk_blocks(
    shape: tuple[int, ...],
    chunk_size: tuple[int, ...],
    itemsize: int,
    max_bytes: int,
) -> Iterator[tuple[slice, ...]]:
    """Tile `shape` into blocks of whole chunks, each at most `max_bytes` in size.

    Blocks are grown one axis at a time, starting from the last one, so that each of
    them is as contiguous as possible in both the source dataset and raw_data. A block
    always contains at least one chunk, even if a single chunk is larger than
    `max_bytes`.
    """
    n_chunks = [ceil_a_over_b(s, c) for s, c in zip(shape, chunk_size, strict=True)]
    if not all(n_chunks):
        return

    # Number of chunks along each axis of a block
    block = [1] * len(shape)
    nbytes = itemsize * math.prod(chunk_size)
    for axis in reversed(range(len(shape))):
        block[axis] = max(1, min(n_chunks[axis], max_bytes // nbytes))
        nbytes *= block[axis]
        if block[axis] < n_chunks[axis]:
            break

    for start in itertools.product(
        *[range(0, n, b) for n, b in zip(n_chunks, block, strict=True)]
    ):
        yield tuple(
            slice(i * c, min((i + b) * c, s))
            for i, b, c, s in zip(start, block, chunk_size, shape, strict=True)
        )


def rewrite_dataset(
    f,
    name: str,
    data,
    *,
    chunks: tuple[int, ...],
    fillvalue: Any = None,
    max_bytes: int = REWRITE_BUFFER_BYTES,
) -> dict[Tuple, Slice]:
    """Copy every chunk of `data` into the `raw_data` of `f`, deduplicating it against
    the chunks already there, and return the `{chunk_index: raw_data slice}` dict to be
    passed to `create_virtual_dataset`.

    Unlike `commit_staged_changes`, which updates the `raw_data` that its
    StagedChangesArray is already built on top of, this rewrites `data` from scratch
    into an unrelated `raw_data`, which may live in another file. This is what
    `recreate_dataset` needs: it can't assume that the new hash table maps the chunks
    to the same locations as the old one, even where the data is unchanged.

    `data` is read one block of chunks at a time, so that peak memory usage is
    O(max_bytes) instead of O(data.size). Deduplication is unaffected: each block is
    deduplicated against the on-disk hash table, which by then already describes every
    chunk written by the previous blocks and by the previous versions.

    Parameters
    ----------
    f:
        Versioned hdf5 file or group, already initialized, which must contain
        ``_version_data/<name>/{raw_data,hash_table}``
    name:
        Name of the dataset
    data:
        Any NumPy-like object supporting ``__getitem__`` with a tuple of slices,
        e.g. a NumPy array, a h5py Dataset, or any of versioned-hdf5's dataset wrappers
    chunks:
        shape of a single chunk
    fillvalue:
        Fill value of the dataset. Chunks that are entirely full of it are not
        written to raw_data at all.
    max_bytes:
        Maximum amount of memory, in bytes, to use to buffer the chunks in transit.
        Rounded up to one chunk. This is only indicative for object string dtypes,
        where the size of the buffer doesn't account for the strings themselves.

    See Also
    --------
    commit_staged_changes
    """
    slices = {}

    for block in _chunk_blocks(data.shape, chunks, data.dtype.itemsize, max_bytes):
        staged_changes = StagedChangesArray.from_array(
            data[block],
            chunk_size=chunks,
            fill_value=fillvalue,
            as_base_slabs=False,
        )
        block_slices = commit_staged_changes(f, name, staged_changes)

        # The chunk indices are relative to the block; shift them back to `data`
        offsets = [s.start for s in block]
        if any(offsets):
            block_slices = {
                Tuple(
                    *[
                        Slice(c.args[0] + o, c.args[1] + o, c.args[2])
                        for c, o in zip(idx.args, offsets, strict=True)
                    ]
                ): raw_data_slice
                for idx, raw_data_slice in block_slices.items()
            }
        slices.update(block_slices)

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

    layout = VirtualLayout(shape=shape, dtype=raw_data.dtype)
    if len(raw_data) == 0:
        assert all(c.isempty() for c in slices)
    else:
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
