import itertools

import h5py
import numpy as np
import pytest
from h5py._hl.filters import guess_chunk
from ndindex import ChunkSize, Slice, Tuple
from numpy.testing import assert_equal

from versioned_hdf5 import backend, slicetools
from versioned_hdf5.backend import (
    DEFAULT_CHUNK_SIZE,
    Filters,
    _chunk_blocks,
    _data_v4_to_sc_hash_table,
    create_base_dataset,
    rewrite_dataset,
    write_dataset,
)
from versioned_hdf5.staged_changes import StagedChangesArray

CHUNK_SIZE_3D = 2**4  # = cbrt(DEFAULT_CHUNK_SIZE)


def test_initialize(setup_vfile):
    with setup_vfile() as f:
        pass
    f.close()


def test_create_base_dataset(h5file):
    create_base_dataset(h5file, "test_data", data=np.ones((DEFAULT_CHUNK_SIZE,)))
    assert h5file["_version_data/test_data/raw_data"].dtype == np.float64


def test_create_base_dataset_multidimension(h5file):
    create_base_dataset(
        h5file,
        "test_data",
        data=np.ones((CHUNK_SIZE_3D, CHUNK_SIZE_3D, 2)),
        chunks=(CHUNK_SIZE_3D, CHUNK_SIZE_3D, CHUNK_SIZE_3D),
    )
    assert h5file["_version_data/test_data/raw_data"].dtype == np.float64


def test_write_dataset(h5file):
    data1 = np.ones((2 * DEFAULT_CHUNK_SIZE,))
    data2 = np.concatenate(
        (
            2 * np.ones((DEFAULT_CHUNK_SIZE,)),
            2 * np.ones((DEFAULT_CHUNK_SIZE,)),
            3 * np.ones((DEFAULT_CHUNK_SIZE,)),
        )
    )
    slices1 = write_dataset(h5file, "test_data", data1)
    slices2 = write_dataset(h5file, "test_data", data2)

    # Chunk size is set by the size the first dataset
    chunksize = guess_chunk(data1.shape, None, data1.dtype.itemsize)[0]

    slices1_expected = {}
    for i in range(data1.size // chunksize):
        data_slice = (Slice(i * chunksize, (i + 1) * chunksize, 1),)
        slices1_expected[data_slice] = slice(0, chunksize)

    last_data1_idx = chunksize
    sorted_slices1 = sorted(slices1.items(), key=lambda x: x[0].raw[0].start)
    sorted_expected1 = sorted(slices1_expected.items(), key=lambda x: x[0][0].start)
    assert sorted_slices1 == sorted_expected1

    slices2_expected = {}

    for i in range(data2.size // chunksize):
        data_slice = (Slice(i * chunksize, (i + 1) * chunksize, 1),)

        if i * chunksize < 2 * DEFAULT_CHUNK_SIZE:
            # Handle first part of dataset
            slices2_expected[data_slice] = slice(
                last_data1_idx, last_data1_idx + chunksize
            )
        else:
            # Handle second part of dataset
            slices2_expected[data_slice] = slice(
                last_data1_idx + chunksize, last_data1_idx + 2 * chunksize
            )

    sorted_slices2 = sorted(slices2.items(), key=lambda x: x[0].raw[0].start)
    sorted_expected2 = sorted(slices2_expected.items(), key=lambda x: x[0][0].start)

    assert sorted_slices2 == sorted_expected2

    ds = h5file["/_version_data/test_data/raw_data"]

    # This will change depending on whether data1.size and data2.size evenly divide
    # chunksize.
    assert ds.shape == (3 * chunksize,)
    assert_equal(ds[0 : 1 * chunksize], 1.0)
    assert_equal(ds[1 * chunksize : 2 * chunksize], 2.0)
    assert_equal(ds[2 * chunksize : 3 * chunksize], 3.0)
    assert_equal(ds[3 * chunksize : 4 * chunksize], 0.0)
    assert ds.dtype == np.float64


def test_write_dataset_multidimension(h5file):
    chunks = 3 * (CHUNK_SIZE_3D,)
    data = np.zeros((2 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D))
    slices1 = write_dataset(h5file, "test_data", data, chunks=chunks)
    data2 = data.copy()
    for n, (i, j, k) in enumerate(itertools.product([0, 1], repeat=3)):
        data2[
            i * CHUNK_SIZE_3D : (i + 1) * CHUNK_SIZE_3D,
            j * CHUNK_SIZE_3D : (j + 1) * CHUNK_SIZE_3D,
            k * CHUNK_SIZE_3D : (k + 1) * CHUNK_SIZE_3D,
        ] = n

    slices2 = write_dataset(h5file, "test_data", data2, chunks=chunks)

    assert slices1 == {
        (
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
        ): slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D),
        (
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
        ): slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D),
        (
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
        ): slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D),
        (
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
        ): slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D),
        (
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
        ): slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D),
        (
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
        ): slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D),
        (
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
        ): slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D),
        (
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
        ): slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D),
    }
    assert slices2 == {
        (
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
        ): slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D),
        (
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
        ): slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D),
        (
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
        ): slice(2 * CHUNK_SIZE_3D, 3 * CHUNK_SIZE_3D),
        (
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
        ): slice(3 * CHUNK_SIZE_3D, 4 * CHUNK_SIZE_3D),
        (
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
        ): slice(4 * CHUNK_SIZE_3D, 5 * CHUNK_SIZE_3D),
        (
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
        ): slice(5 * CHUNK_SIZE_3D, 6 * CHUNK_SIZE_3D),
        (
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
        ): slice(6 * CHUNK_SIZE_3D, 7 * CHUNK_SIZE_3D),
        (
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
        ): slice(7 * CHUNK_SIZE_3D, 8 * CHUNK_SIZE_3D),
    }

    ds = h5file["/_version_data/test_data/raw_data"]
    assert ds.shape == (8 * CHUNK_SIZE_3D, CHUNK_SIZE_3D, CHUNK_SIZE_3D)
    for n in range(8):
        assert_equal(ds[n * CHUNK_SIZE_3D : (n + 1) * CHUNK_SIZE_3D], n)
    assert ds.dtype == np.float64


def test_write_dataset_offset(h5file):
    data1 = np.ones((2 * DEFAULT_CHUNK_SIZE,))
    data2 = np.concatenate(
        (
            2 * np.ones((DEFAULT_CHUNK_SIZE,)),
            2 * np.ones((DEFAULT_CHUNK_SIZE,)),
            3 * np.ones((DEFAULT_CHUNK_SIZE - 2,)),
        )
    )
    slices1 = write_dataset(h5file, "test_data", data1)
    slices2 = write_dataset(h5file, "test_data", data2)

    chunksize = guess_chunk(data1.shape, None, data1.dtype.itemsize)[0]

    slices1_expected = {}
    for i in range(data1.size // chunksize):
        data_slice = (Slice(i * chunksize, (i + 1) * chunksize, 1),)
        slices1_expected[data_slice] = slice(0, chunksize)

    last_data1_idx = chunksize
    slices2_expected = {}
    for i in range(data2.size // chunksize):
        data_slice = (Slice(i * chunksize, (i + 1) * chunksize, 1),)

        if i * chunksize < 2 * DEFAULT_CHUNK_SIZE:
            slices2_expected[data_slice] = slice(
                last_data1_idx, last_data1_idx + chunksize
            )
        else:
            slices2_expected[data_slice] = slice(
                last_data1_idx + chunksize, last_data1_idx + 2 * chunksize
            )

    n_remaining = data2.size % chunksize
    data_slice = (Slice((data2.size // chunksize) * chunksize, data2.size, 1),)
    slices2_expected[data_slice] = slice(
        last_data1_idx + 2 * chunksize,
        last_data1_idx + 2 * chunksize + n_remaining,
    )

    sorted_slices1 = sorted(slices1.items(), key=lambda x: x[0].raw[0].start)
    sorted_expected1 = sorted(slices1_expected.items(), key=lambda x: x[0][0].start)
    sorted_slices2 = sorted(slices2.items(), key=lambda x: x[0].raw[0].start)
    sorted_expected2 = sorted(slices2_expected.items(), key=lambda x: x[0][0].start)

    assert sorted_slices1 == sorted_expected1
    assert sorted_slices2 == sorted_expected2

    ds = h5file["/_version_data/test_data/raw_data"]
    assert ds.shape == (4 * chunksize,)
    assert_equal(ds[0 * chunksize : 1 * chunksize], 1.0)
    assert_equal(ds[1 * chunksize : 2 * chunksize], 2.0)
    assert_equal(ds[2 * chunksize : 3 * chunksize], 3.0)
    assert_equal(ds[2 * chunksize : 3 * chunksize], 3.0)
    assert_equal(ds[3 * chunksize : 4 * chunksize - 2], 3.0)
    assert_equal(ds[4 * chunksize - 2 : 4 * chunksize], 0.0)


def test_write_dataset_offset_multidimension(h5file):
    chunks = ChunkSize(3 * (CHUNK_SIZE_3D,))
    shape = (2 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D)
    data = np.zeros(shape)
    slices1 = write_dataset(h5file, "test_data", data, chunks=chunks)
    shape2 = (2 * CHUNK_SIZE_3D - 2, 2 * CHUNK_SIZE_3D - 2, 2 * CHUNK_SIZE_3D - 2)
    data2 = np.empty(shape2)
    for n, c in enumerate(chunks.indices(shape)):
        data2[c.raw] = n

    slices2 = write_dataset(h5file, "test_data", data2, chunks=chunks)

    assert slices1 == {
        (
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
        ): slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D),
        (
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
        ): slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D),
        (
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
        ): slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D),
        (
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
        ): slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D),
        (
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
        ): slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D),
        (
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
        ): slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D),
        (
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
        ): slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D),
        (
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 1),
        ): slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D),
    }

    assert slices2 == {
        (
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
        ): slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D),
        (
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D - 2, 1),
        ): slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D),
        (
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D - 2, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
        ): slice(2 * CHUNK_SIZE_3D, 3 * CHUNK_SIZE_3D),
        (
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D - 2, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D - 2, 1),
        ): slice(3 * CHUNK_SIZE_3D, 4 * CHUNK_SIZE_3D),
        (
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D - 2, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
        ): slice(4 * CHUNK_SIZE_3D, 5 * CHUNK_SIZE_3D - 2),
        (
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D - 2, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D - 2, 1),
        ): slice(5 * CHUNK_SIZE_3D, 6 * CHUNK_SIZE_3D - 2),
        (
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D - 2, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D - 2, 1),
            Slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D, 1),
        ): slice(6 * CHUNK_SIZE_3D, 7 * CHUNK_SIZE_3D - 2),
        (
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D - 2, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D - 2, 1),
            Slice(1 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D - 2, 1),
        ): slice(7 * CHUNK_SIZE_3D, 8 * CHUNK_SIZE_3D - 2),
    }

    ds = h5file["/_version_data/test_data/raw_data"]
    assert ds.shape == (8 * CHUNK_SIZE_3D, CHUNK_SIZE_3D, CHUNK_SIZE_3D)
    for n, c in enumerate(chunks.indices(shape2)):
        a = np.zeros(chunks)
        a[Tuple(*[slice(0, i) for i in shape2]).as_subindex(c).raw] = n
        assert_equal(ds[n * CHUNK_SIZE_3D : (n + 1) * CHUNK_SIZE_3D], a)
    assert ds.dtype == np.float64


def test_write_dataset_offset_chunk_size(h5file):
    chunk_size = 2**10
    chunks = (chunk_size,)
    slices1 = write_dataset(
        h5file, "test_data", 1 * np.ones((2 * chunk_size,)), chunks=chunks
    )
    slices2 = write_dataset(
        h5file,
        "test_data",
        np.concatenate(
            (2 * np.ones(chunks), 2 * np.ones(chunks), 3 * np.ones((chunk_size - 2,)))
        ),
    )

    assert slices1 == {
        Tuple(Slice(0 * chunk_size, 1 * chunk_size, 1)): slice(
            0 * chunk_size, 1 * chunk_size
        ),
        Tuple(Slice(1 * chunk_size, 2 * chunk_size, 1)): slice(
            0 * chunk_size, 1 * chunk_size
        ),
    }
    assert slices2 == {
        Tuple(Slice(0 * chunk_size, 1 * chunk_size, 1)): slice(
            1 * chunk_size, 2 * chunk_size
        ),
        Tuple(Slice(1 * chunk_size, 2 * chunk_size, 1)): slice(
            1 * chunk_size, 2 * chunk_size
        ),
        Tuple(Slice(2 * chunk_size, 3 * chunk_size - 2, 1)): slice(
            2 * chunk_size, 3 * chunk_size - 2
        ),
    }

    ds = h5file["/_version_data/test_data/raw_data"]
    assert ds.shape == (3 * chunk_size,)
    assert_equal(ds[0 * chunk_size : 1 * chunk_size], 1.0)
    assert_equal(ds[1 * chunk_size : 2 * chunk_size], 2.0)
    assert_equal(ds[2 * chunk_size : 3 * chunk_size - 2], 3.0)
    assert_equal(ds[3 * chunk_size - 2 : 4 * chunk_size], 0.0)


def test_write_dataset_compression(h5file):
    data = np.ones((2 * DEFAULT_CHUNK_SIZE,))

    # Chunk size is set by the size the first dataset
    chunksize = guess_chunk(data.shape, None, data.dtype.itemsize)[0]
    nchunks = int(np.ceil(data.size / chunksize))

    slices1 = write_dataset(
        h5file,
        "test_data",
        data,
        filters=Filters(compression="gzip", compression_opts=3),
    )

    with pytest.raises(ValueError):
        write_dataset(
            h5file,
            "test_data",
            np.ones((DEFAULT_CHUNK_SIZE,)),
            filters=Filters(compression="lzf"),
        )

    with pytest.raises(ValueError):
        write_dataset(
            h5file,
            "test_data",
            np.ones((DEFAULT_CHUNK_SIZE,)),
            filters=Filters(compression="gzip", compression_opts=4),
        )

    expected = {}
    for i in range(nchunks):
        expected[(Slice(i * chunksize, (i + 1) * chunksize, 1),)] = slice(0, chunksize)

    assert slices1 == expected
    ds = h5file["/_version_data/test_data/raw_data"]
    assert ds.shape == (chunksize,)
    assert_equal(ds[0:chunksize], 1.0)
    assert ds.dtype == np.float64
    assert ds.compression == "gzip"
    assert ds.compression_opts == 3


def _raw_data_hashtable(vfile, name):
    grp = vfile.f["_version_data"][name]
    return grp["raw_data"], grp["hash_table"]


def test_commit_staged_changes_modify(vfile):
    """Modifying a dataset carried over from a previous version goes through
    commit_staged_changes (an InMemoryDataset, i.e. one base slab on entry), not the
    legacy write_dataset path.
    """
    with vfile.stage_version("r0") as sv:
        sv.create_dataset("x", data=np.arange(30), chunks=(10,))
    with vfile.stage_version("r1") as sv:
        sv["x"][5] = 999  # partial write -> InMemoryDataset -> commit_staged_changes

    assert_equal(vfile["r0"]["x"][:], np.arange(30))
    expected = np.arange(30)
    expected[5] = 999
    assert_equal(vfile["r1"]["x"][:], expected)

    # Invariant: the on-disk hash table records exactly one chunk per raw_data chunk,
    # and holds no rows beyond them.
    raw_data, hash_table = _raw_data_hashtable(vfile, "x")
    assert raw_data.shape[0] // 10 == hash_table.attrs["largest_index"] == 4
    assert hash_table.shape == (4,)


def test_commit_staged_changes_edge_chunk_hashtable(vfile):
    """The on-disk hash table records the *trimmed* (start, stop) of a rewritten edge
    chunk - start still lands on a full-chunk boundary, but stop is the chunk's logical
    length - so replay keeps matching it against the virtual dataset's slices.
    """
    with vfile.stage_version("r0") as sv:
        sv.create_dataset("x", data=np.arange(25), chunks=(10,))  # chunk 2 is edge (5)
    with vfile.stage_version("r1") as sv:
        sv["x"][22] = 999  # rewrite the trailing edge chunk (indices 20..24)

    _, hash_table = _raw_data_hashtable(vfile, "x")
    # r0 committed 3 chunks; r1 appends one new (edge) chunk in row 3.
    assert hash_table.attrs["largest_index"] == 4
    assert tuple(int(v) for v in hash_table[3]["shape"]) == (30, 35)

    expected = np.arange(25)
    expected[22] = 999
    assert_equal(vfile["r1"]["x"][:], expected)


@pytest.mark.parametrize("n_new", [4, 6], ids=["shrink", "grow"])
@pytest.mark.parametrize("ht_grown", [False, True], ids=["raw_only", "raw+ht"])
def test_commit_staged_changes_recovers_from_failed_commit(vfile, ht_grown, n_new):
    """A commit that crashes halfway through leaves raw_data - and possibly the hash
    table too - enlarged, with garbage past the last recorded chunk. Neither shape is
    trustworthy; hash_table.attrs["largest_index"], written last, is the packed length.
    commit_staged_changes reads it and resizes both datasets to fit exactly the
    surviving chunks plus the new ones, so a failed commit is fully overwritten
    regardless of whether the next version is smaller or larger than it.
    """
    with vfile.stage_version("r0") as sv:
        sv.create_dataset("x", data=np.arange(100), chunks=(10,))

    raw_data, hash_table = _raw_data_hashtable(vfile, "x")
    assert raw_data.shape == (100,)
    assert hash_table.shape == (10,)
    assert hash_table.attrs["largest_index"] == 10

    # Simulate r1 crashing while committing 5 extra chunks. raw_data grows first;
    # the hash table may or may not have been enlarged (with garbage rows) yet.
    # Either way largest_index remains 10, which makes the crash detectable.
    raw_data.resize((150,))
    raw_data[100:] = -12345
    if ht_grown:
        garbage = np.zeros(5, dtype=hash_table.dtype)
        garbage["hash"] = 123
        garbage["shape"] = [(s, s + 10) for s in range(100, 150, 10)]
        hash_table.resize((15,))
        hash_table[10:] = garbage

    # r2 commits n_new original chunks; fewer or more than r1 was going to
    with vfile.stage_version("r2") as sv:
        sv["x"][: n_new * 10 : 10] = -1

    n = 10 + n_new
    raw_data, hash_table = _raw_data_hashtable(vfile, "x")
    assert raw_data.shape == (n * 10,)
    assert not np.any(raw_data[:] == -12345)
    assert hash_table.attrs["largest_index"] == n
    assert hash_table.shape == (n,)

    assert_equal(vfile["r0"]["x"][:], np.arange(100))
    expected = np.arange(100)
    expected[: n_new * 10 : 10] = -1
    assert_equal(vfile["r2"]["x"][:], expected)

    # The rows r2 appended are sane: r3 dedups against them and writes nothing new
    with vfile.stage_version("r3") as sv:
        sv["x"][0] = -1
    raw_data, hash_table = _raw_data_hashtable(vfile, "x")
    assert raw_data.shape == (n * 10,)
    assert hash_table.attrs["largest_index"] == n


def test_commit_staged_changes_dedup_no_new_chunks(vfile):
    """A version that only re-stages chunks identical to ones already in raw_data
    appends no new base slab and leaves the on-disk hash table untouched.
    """
    with vfile.stage_version("r0") as sv:
        sv.create_dataset("x", data=np.arange(30), chunks=(10,))
    raw_data, hash_table = _raw_data_hashtable(vfile, "x")
    raw_data_before = raw_data[:]
    hash_table_before = hash_table[:]
    assert int(hash_table.attrs["largest_index"]) == 3

    with vfile.stage_version("r1") as sv:
        sv["x"][5] = sv["x"][5]

    raw_data, hash_table = _raw_data_hashtable(vfile, "x")
    assert_equal(raw_data[:], raw_data_before)
    assert_equal(hash_table[:], hash_table_before)
    assert int(hash_table.attrs["largest_index"]) == 3


def test_commit_staged_changes_sparse_edge_chunk(vfile):
    """A brand new (sparse) dataset has no base slab on entry (n_base_slabs == 0).
    Partially filling it, edge chunk included, commits the right chunks.
    """
    with vfile.stage_version("r0") as sv:
        sv.create_dataset("x", shape=(25,), chunks=(10,))
        sv["x"][:22] = np.arange(22)  # leaves the edge chunk's tail as fill_value

    expected = np.zeros(25)
    expected[:22] = np.arange(22)
    assert_equal(vfile["r0"]["x"][:], expected)

    raw_data, hash_table = _raw_data_hashtable(vfile, "x")
    assert raw_data.shape[0] // 10 == hash_table.attrs["largest_index"] == 3


@pytest.mark.parametrize("delete", [True, False])
def test_commit_staged_changes_recreated_sparse_dataset(vfile, delete):
    """An InMemorySparseDataset (no base slabs) can nonetheless have raw_data, either
    because it was deleted and then created anew or because it was created independently
    on two branches of the version DAG. Its chunks must be appended to raw_data and
    deduplicated against it, not overwrite it.
    """
    with vfile.stage_version("r0") as sv:
        sv.create_dataset("x", data=[1, 2, 3, 4, 5, 6], chunks=(2,), dtype=np.int64)
    raw_data, hash_table = _raw_data_hashtable(vfile, "x")
    assert hash_table.attrs["largest_index"] == 3

    if delete:
        with vfile.stage_version("r1") as sv:
            del sv["x"]

    with vfile.stage_version("r2", prev_version="r1" if delete else "") as sv:
        sv.create_dataset("x", shape=(6,), dtype=np.int64, chunks=(2,))
        sv["x"][:2] = [1, 2]  # identical to chunk 0 of r0; deduplicated
        sv["x"][2:4] = 7, 8  # original; appended to raw_data
        # chunk 2 is left full of fill_value

    # r0 was not overwritten
    assert_equal(vfile["r0"]["x"][:], [1, 2, 3, 4, 5, 6])
    assert_equal(vfile["r2"]["x"][:], [1, 2, 7, 8, 0, 0])

    # Exactly one new chunk was appended
    raw_data, hash_table = _raw_data_hashtable(vfile, "x")
    assert_equal(raw_data[:8], [1, 2, 3, 4, 5, 6, 7, 8])
    assert hash_table.attrs["largest_index"] == 4
    assert hash_table.shape == (4,)


def test_commit_staged_changes_hotswapped_sparse_dataset(vfile):
    """An InMemorySparseDataset (no base slabs) can nonetheless have raw_data
    after DatasetWrapper hot-swapped it from a InMemoryDataset.
    """
    with vfile.stage_version("r0") as sv:
        sv.create_dataset("x", data=[1, 2, 3, 4, 5, 6], chunks=(2,), dtype=np.int64)
    raw_data, hash_table = _raw_data_hashtable(vfile, "x")
    assert hash_table.attrs["largest_index"] == 3

    with vfile.stage_version("r1") as sv:
        sv["x"][:] = 0  # hot-swap InMemoryDataset -> InMemoryArrayDataset
        sv["x"].resize((7,))  # hot-swap InMemoryArrayDataset -> InMemorySparseDataset
        sv["x"].resize((6,))
        sv["x"][:2] = [1, 2]  # identical to chunk 0 of r0; deduplicated
        sv["x"][2:4] = 7, 8  # original; appended to raw_data

    # r0 was not overwritten
    assert_equal(vfile["r0"]["x"][:], [1, 2, 3, 4, 5, 6])
    assert_equal(vfile["r1"]["x"][:], [1, 2, 7, 8, 0, 0])

    # Exactly one new chunk was appended
    raw_data, hash_table = _raw_data_hashtable(vfile, "x")
    assert_equal(raw_data[:8], [1, 2, 3, 4, 5, 6, 7, 8])
    assert hash_table.attrs["largest_index"] == 4
    assert hash_table.shape == (4,)


def test_data_v4_to_sc_hash_table_out_of_order(vfile):
    """_data_v4_to_sc_hash_table() indexes the digests by chunk index, whatever order
    the records happen to be in on disk. VersionedHDF5File.rebuild_hashtables() writes
    them in the order the versions reference them, which is not the order they lie in
    along axis 0 of raw_data.
    """
    # rebuild_hashtables() traverses versions in alphabetical order,
    # not in creation order
    with vfile.stage_version("z") as sv:
        sv.create_dataset("x", data=np.array([1, 2, 3, 4]), chunks=(2,))
    with vfile.stage_version("a") as sv:
        # Swap the two chunks around. Both are deduplicated against version z, so
        # raw_data is unchanged; version a just references its chunks in reverse order.
        sv["x"][:2] = [3, 4]
        sv["x"][2:] = [1, 2]

    raw_data = vfile.f["_version_data/x/raw_data"]
    assert_equal(raw_data[:], [1, 2, 3, 4])

    vfile.rebuild_hashtables()

    hash_table = vfile.f["_version_data/x/hash_table"]
    records = hash_table[: int(hash_table.attrs["largest_index"])]
    assert records["shape"][:, 0].tolist() == [2, 0]  # Not in chunk order

    on_disk = np.ascontiguousarray(records["hash"]).view(np.uint64)
    actual = _data_v4_to_sc_hash_table(hash_table, 2)
    assert_equal(actual, on_disk[::-1])


def test_commit_staged_changes_out_of_order_hashtable(vfile):
    """Staged chunks are deduplicated onto the correct raw_data offset even when the
    records of the on-disk hash table are not in chunk order, which is the case after
    VersionedHDF5File.rebuild_object_dtype_hashtables().
    """
    with vfile.stage_version("r0") as sv:
        sv.create_dataset("x", data=np.array([1, 2, 3, 4]), chunks=(2,))

    raw_data, hash_table = _raw_data_hashtable(vfile, "x")
    assert raw_data.shape == (4,)
    # Reverse the records, preserving each (hash, (start, stop)) pairing
    nrows = int(hash_table.attrs["largest_index"])
    hash_table[:nrows] = hash_table[:nrows][::-1]

    with vfile.stage_version("r1") as sv:
        # Rewrite chunk 0 with the contents of chunk 1. It must be deduplicated onto
        # raw_data[2:4] and not onto raw_data[0:2], which still holds [1, 2].
        sv["x"][:2] = [3, 4]

    assert_equal(vfile["r1"]["x"][:], [3, 4, 3, 4])
    raw_data, hash_table = _raw_data_hashtable(vfile, "x")
    assert raw_data.shape == (4,)  # No new chunk was written
    assert hash_table.attrs["largest_index"] == 2


@pytest.mark.parametrize(
    ("shape", "chunk_size", "max_bytes", "expect"),
    [
        # A generous budget yields a single block covering everything
        ((5,), (2,), 1000, [(slice(0, 5),)]),
        # One chunk per block; the last block is trimmed to the edge of the array
        ((5,), (2,), 16, [(slice(0, 2),), (slice(2, 4),), (slice(4, 5),)]),
        # Two chunks per block
        ((7,), (2,), 32, [(slice(0, 4),), (slice(4, 7),)]),
        # max_bytes is rounded up to one whole chunk
        ((4,), (2,), 0, [(slice(0, 2),), (slice(2, 4),)]),
        # Blocks grow from the last axis, so that they are as contiguous as possible
        (
            (4, 6),
            (2, 2),
            64,
            [
                (slice(0, 2), slice(0, 4)),
                (slice(0, 2), slice(4, 6)),
                (slice(2, 4), slice(0, 4)),
                (slice(2, 4), slice(4, 6)),
            ],
        ),
        # Axis 1 fits entirely, so axis 0 starts growing too
        ((4, 6), (2, 2), 96, [(slice(0, 2), slice(0, 6)), (slice(2, 4), slice(0, 6))]),
        ((4, 6), (2, 2), 192, [(slice(0, 4), slice(0, 6))]),
        # Size-0 arrays have no chunks at all
        ((0,), (2,), 1000, []),
        ((4, 0), (2, 2), 1000, []),
    ],
)
def test_chunk_blocks(shape, chunk_size, max_bytes, expect):
    assert list(_chunk_blocks(shape, chunk_size, 8, max_bytes)) == expect


# One whole chunk, one chunk row, and the whole array at a time
@pytest.mark.parametrize("max_bytes", [0, 64, 1000])
def test_rewrite_dataset(vfile, max_bytes):
    """rewrite_dataset() copies every chunk of an array into a brand new raw_data,
    deduplicating them, and returns the same committed StagedChangesArray
    regardless of how many chunks it buffers at a time.
    """
    # Chunk (1, 0) is a duplicate of chunk (0, 1) and chunk (1, 1) of chunk (0, 0);
    # the last row is a pair of edge chunks.
    data = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [3, 4, 1, 2],
            [7, 8, 5, 6],
            [9, 9, 0, 0],
        ]
    )
    create_base_dataset(vfile.f, "x", data=data[:0], chunks=(2, 2), fillvalue=0)
    staged_changes = rewrite_dataset(
        vfile.f, "x", data, chunks=(2, 2), fillvalue=0, max_bytes=max_bytes
    )

    # Chunks (1, 0) and (1, 1) are deduplicated against (0, 1) and (0, 0).
    # Chunk (2, 1) is written despite being entirely fillvalue: edge chunks are
    # hashed over their visible cells only, so they can never match the full slab
    assert_equal(
        staged_changes.slab_indices,
        [[1, 1], [1, 1], [1, 1]],
    )
    assert_equal(
        staged_changes.slab_offsets,
        [[0, 2], [2, 0], [4, 6]],
    )
    # The chunk map stitches the original array back together
    assert_equal(staged_changes[()], data)

    # Only 4 chunks were written; the two duplicates were deduplicated away.
    # raw_data always grows by whole chunks, so the two edge chunks are padded.
    raw_data, hash_table = _raw_data_hashtable(vfile, "x")
    assert hash_table.attrs["largest_index"] == 4
    assert_equal(
        raw_data[:],
        [[1, 2], [5, 6], [3, 4], [7, 8], [9, 9], [0, 0], [0, 0], [0, 0]],
    )

    # ...and so does the virtual dataset it feeds
    vfile.f["_version_data/versions"].create_group("r0")
    slicetools.create_virtual_dataset(vfile.f, "r0", "x", staged_changes, fillvalue=0)
    assert_equal(vfile.f["_version_data/versions/r0/x"][:], data)


def _rewritten(vfile, name, data, chunks, fillvalue, max_bytes):
    """rewrite_dataset() into a brand new dataset; return the committed
    StagedChangesArray and the raw_data/hash_table it produced"""
    create_base_dataset(
        vfile.f, name, data=data[:0], chunks=chunks, fillvalue=fillvalue
    )
    sc = rewrite_dataset(
        vfile.f,
        name,
        data,
        chunks=chunks,
        fillvalue=fillvalue,
        max_bytes=max_bytes,
    )
    raw_data, hash_table = _raw_data_hashtable(vfile, name)
    return sc, raw_data, hash_table


# One whole chunk, one chunk row, and the whole array at a time
@pytest.mark.parametrize("max_bytes", [0, 128, 100000])
def test_rewrite_dataset_multidimension(vfile, max_bytes):
    """A 3-D array with duplicate chunks, edge chunks along two axes, and a full-sized
    chunk which is entirely fillvalue: however it is diced into blocks, the same chunks
    end up on raw_data. They are deduplicated within a block and across blocks, and a
    full-sized chunk entirely fillvalue is dropped to the full slab.
    """
    chunks = (2, 2, 2)
    data = np.broadcast_to(np.array(1.5), (5, 4, 3)).copy()
    data[0:2, 0:2, 0:2] = 11.0
    data[2:4, 2:4, 0:2] = 11.0  # Duplicate of (0, 0, 0)
    data[2:4, 0:2, 0:2] = 33.0
    data[0:2, 2:4, 0:2] = 1.5  # Full-sized chunk entirely fillvalue: never written
    # The chunks at z=2 are edge chunks (visible shape (*, *, 1)). Edge chunks hash
    # over their visible cells only, so they are always written, even when entirely
    # fillvalue
    sc, _, hash_table = _rewritten(vfile, f"x{max_bytes}", data, chunks, 1.5, max_bytes)

    # 11.0 and 33.0 once each, plus one chunk each for the three distinct visible
    # edge shapes: 1x2x2, 2x2x1 and 1x2x1
    assert hash_table.attrs["largest_index"] == 5
    assert_equal(sc[()], data)


@pytest.mark.parametrize("max_bytes", [0, 100000])
def test_rewrite_dataset_preexisting_raw_data(vfile, max_bytes):
    """rewrite_dataset() deduplicates against chunks that were already on raw_data
    (e.g. written by a previous version or an earlier rewrite) and appends new chunks
    after them; offsets of reused chunks are absolute into raw_data."""
    chunks = (2, 2)
    fillvalue = 0.0
    data = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
        ],
        dtype=float,
    )
    sc1, _, hash_table = _rewritten(vfile, "x", data, chunks, fillvalue, 1000)
    # Chunks (1, 0) and (1, 1) are identical, so only 3 of the 4 were written
    n_chunks_1 = int(hash_table.attrs["largest_index"])
    assert n_chunks_1 == 3

    # Rewrite a modified copy: three of its chunks are duplicates of chunks already on
    # raw_data and one is brand new
    data2 = data.copy()
    data2[0:2, 0:2] = data[0:2, 2:4]
    data2[2:4, 2:4] = [[8.0, 8.0], [8.0, 8.0]]
    sc2 = rewrite_dataset(
        vfile.f, "x", data2, chunks=chunks, fillvalue=fillvalue, max_bytes=max_bytes
    )

    # Only the one new chunk was written; everything else was reused
    assert hash_table.attrs["largest_index"] == n_chunks_1 + 1

    # Chunks point at the right places: the new duplicate of chunk (0, 1) points at
    # its old absolute offset, and the brand new chunk is appended after the old data
    assert_equal(sc2[()], data2)
    assert sc2.slab_indices[0, 0] == 1
    assert sc2.slab_offsets[0, 0] == sc1.slab_offsets[0, 1]
    assert sc2.slab_offsets[1, 1] == n_chunks_1 * chunks[0]


def test_rewrite_dataset_reuses_hash_state_across_blocks(vfile, monkeypatch):
    """Each rewrite block shares one explicit hash state, but reads disk table once."""
    create_base_dataset(vfile.f, "x", data=np.arange(4), chunks=(2,))
    reads = 0
    original = backend._data_v4_to_sc_hash_table

    def counted_read(*args, **kwargs):
        nonlocal reads
        reads += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(backend, "_data_v4_to_sc_hash_table", counted_read)
    data = np.array([10, 11, 0, 1, 2, 3, 30, 31])
    sc = rewrite_dataset(vfile.f, "x", data, chunks=(2,), max_bytes=16)

    assert reads == 1
    raw_data, hash_table = _raw_data_hashtable(vfile, "x")
    assert raw_data.shape == (8,)
    assert hash_table.attrs["largest_index"] == 4
    assert_equal(raw_data[:], np.array([0, 1, 2, 3, 10, 11, 30, 31]))
    assert_equal(sc[()], data)


def test_commit_state_rejects_different_target(h5file):
    create_base_dataset(h5file, "x", data=np.empty(0, dtype=np.int64), chunks=(2,))
    create_base_dataset(h5file, "y", data=np.empty(0, dtype=np.int64), chunks=(4,))
    state = backend.CommitState()
    first = StagedChangesArray.from_array(
        np.array([1, 2]), chunk_size=(2,), as_base_slabs=False
    )
    backend.commit_staged_changes(h5file, "x", first, state)

    second = StagedChangesArray.from_array(
        np.array([1, 2, 3, 4]), chunk_size=(4,), as_base_slabs=False
    )
    with pytest.raises(ValueError, match="different target or chunk size"):
        backend.commit_staged_changes(h5file, "y", second, state)


def test_commit_state_rejects_different_file(h5file, tmp_path):
    create_base_dataset(h5file, "x", data=np.empty(0, dtype=np.int64), chunks=(2,))
    state = backend.CommitState()
    backend.commit_staged_changes(
        h5file,
        "x",
        StagedChangesArray.from_array(
            np.array([1, 2]), chunk_size=(2,), as_base_slabs=False
        ),
        state,
    )

    with h5py.File(tmp_path / "other.h5", "w") as other:
        backend.initialize(other)
        create_base_dataset(other, "x", data=np.empty(0, dtype=np.int64), chunks=(2,))
        with pytest.raises(ValueError, match="different target or chunk size"):
            backend.commit_staged_changes(
                other,
                "x",
                StagedChangesArray.from_array(
                    np.array([3, 4]), chunk_size=(2,), as_base_slabs=False
                ),
                state,
            )


def test_commit_state_resets_after_failed_commit(h5file, monkeypatch):
    create_base_dataset(h5file, "x", data=np.empty(0, dtype=np.int64), chunks=(2,))
    state = backend.CommitState()
    data = np.array([10, 11])
    original = backend._sc_hash_table_to_data_v4
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated hash-table failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(backend, "_sc_hash_table_to_data_v4", fail_once)
    with pytest.raises(RuntimeError, match="simulated hash-table failure"):
        backend.commit_staged_changes(
            h5file,
            "x",
            StagedChangesArray.from_array(data, chunk_size=(2,), as_base_slabs=False),
            state,
        )
    assert not state.is_initialized()

    raw_data = h5file["_version_data/x/raw_data"]
    hash_table = h5file["_version_data/x/hash_table"]
    assert raw_data.shape == (0,)
    assert hash_table.shape == (0,)
    assert hash_table.attrs["largest_index"] == 0

    retried = StagedChangesArray.from_array(data, chunk_size=(2,), as_base_slabs=False)
    backend.commit_staged_changes(h5file, "x", retried, state)
    # The failed append is deliberately interrupted between raw-data and hash-table
    # writes. State and durable extents are the contract under test here; normal
    # commit/rewrite tests cover payload values without that artificial interruption.
    assert hash_table.attrs["largest_index"] == 1
    assert state.is_initialized()


@pytest.mark.parametrize("max_bytes", [0, 1000])
def test_rewrite_dataset_all_fillvalue(vfile, max_bytes):
    """A dataset whose chunks are all entirely fillvalue writes nothing to raw_data,
    and is returned with no base slab at all rather than an empty raw_data one."""
    chunks = (2, 2)
    data = np.full((4, 6), 2.5)
    sc, _, hash_table = _rewritten(vfile, "x", data, chunks, 2.5, max_bytes)
    assert hash_table.attrs["largest_index"] == 0
    assert sc.n_base_slabs == 0
    assert_equal(sc[()], data)


@pytest.mark.parametrize("max_bytes", [0, 1000])
def test_rewrite_dataset_empty(vfile, max_bytes):
    """A size-0 dataset has no chunks to rewrite."""
    data = np.empty((0, 3))
    create_base_dataset(vfile.f, "x", data=data, chunks=(2, 2))
    staged_changes = rewrite_dataset(
        vfile.f, "x", data, chunks=(2, 2), max_bytes=max_bytes
    )
    assert staged_changes.shape == (0, 3)
    assert not staged_changes.has_base_chunks
