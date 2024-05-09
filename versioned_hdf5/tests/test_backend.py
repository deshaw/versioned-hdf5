import itertools
from unittest import mock

import h5py
import numpy as np
from h5py._hl.filters import guess_chunk
from ndindex import ChunkSize, Slice, Tuple
from numpy.testing import assert_equal
from pytest import mark, raises

import versioned_hdf5 as vhdf

from ..backend import (
    DEFAULT_CHUNK_SIZE,
    SplitResult,
    create_base_dataset,
    create_virtual_dataset,
    get_space_remaining,
    last_raw_used_chunk,
    write_dataset,
    write_dataset_chunks,
)

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


def test_write_dataset_chunks(h5file):
    data = np.ones((2 * DEFAULT_CHUNK_SIZE,))
    slices1 = write_dataset(h5file, "test_data", data)
    chunksize = guess_chunk(data.shape, None, data.dtype.itemsize)[0]

    raw_slice1 = Tuple(Slice(0 * chunksize, 1 * chunksize, 1))
    slices2 = write_dataset_chunks(
        h5file,
        "test_data",
        {
            Tuple(Slice(0 * chunksize, 1 * chunksize, 1)): slices1[raw_slice1],
            Tuple(Slice(1 * chunksize, 2 * chunksize, 1)): 2 * np.ones((chunksize,)),
            Tuple(Slice(2 * chunksize, 3 * chunksize, 1)): 2 * np.ones((chunksize,)),
            Tuple(Slice(3 * chunksize, 4 * chunksize, 1)): 3 * np.ones((chunksize,)),
        },
    )

    slices1_expected = {}
    for i in range(data.size // chunksize):
        data_slice = Tuple(Slice(i * chunksize, (i + 1) * chunksize, 1))
        slices1_expected[data_slice] = slice(0, chunksize)

    assert slices1 == slices1_expected
    assert slices2 == {
        Tuple(Slice(0 * chunksize, 1 * chunksize, 1)): slice(
            0 * chunksize, 1 * chunksize
        ),
        Tuple(Slice(1 * chunksize, 2 * chunksize, 1)): slice(
            1 * chunksize, 2 * chunksize
        ),
        Tuple(Slice(2 * chunksize, 3 * chunksize, 1)): slice(
            1 * chunksize, 2 * chunksize
        ),
        Tuple(Slice(3 * chunksize, 4 * chunksize, 1)): slice(
            2 * chunksize, 3 * chunksize
        ),
    }

    ds = h5file["/_version_data/test_data/raw_data"]
    assert ds.shape == (3 * chunksize,)
    assert_equal(ds[0 * chunksize : 1 * chunksize], 1.0)
    assert_equal(ds[1 * chunksize : 2 * chunksize], 2.0)
    assert_equal(ds[2 * chunksize : 3 * chunksize], 3.0)
    assert_equal(ds[3 * chunksize : 4 * chunksize], 0.0)
    assert ds.dtype == np.float64


def test_write_dataset_chunks_multidimension(h5file):
    chunks = ChunkSize(3 * (CHUNK_SIZE_3D,))
    shape = (2 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D)
    data = np.zeros(shape)
    slices1 = write_dataset(h5file, "test_data", data, chunks=chunks)
    data_dict = {}
    for n, c in enumerate(chunks.indices(shape)):
        if n == 0:
            data_dict[c] = slices1[c]
        else:
            data_dict[c] = n * np.ones(chunks)

    slices1 = write_dataset(h5file, "test_data", data, chunks=chunks)
    slices2 = write_dataset_chunks(h5file, "test_data", data_dict)

    assert slices1 == {
        c: slice(0 * CHUNK_SIZE_3D, 1 * CHUNK_SIZE_3D) for c in chunks.indices(shape)
    }
    assert slices2 == {
        c: slice(i * CHUNK_SIZE_3D, (i + 1) * CHUNK_SIZE_3D)
        for i, c in enumerate(chunks.indices(shape))
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


@mark.setup_args(version_name="test_version")
def test_create_virtual_dataset(h5file):
    """Check that creating a virtual dataset from chunks of real datasets works."""
    data1 = np.ones((2 * DEFAULT_CHUNK_SIZE,))
    data2 = np.concatenate(
        (
            2 * np.ones((DEFAULT_CHUNK_SIZE,)),
            3 * np.ones((DEFAULT_CHUNK_SIZE,)),
        )
    )

    # Chunk size is set by the size the first dataset
    chunksize = guess_chunk(data1.shape, None, data1.dtype.itemsize)[0]

    with h5file as f:
        slices1 = write_dataset(f, "test_data", data1)
        slices2 = write_dataset(f, "test_data", data2)

        nchunks1 = int(np.ceil(2 * DEFAULT_CHUNK_SIZE / chunksize))

        # The virtual dataset contains all the data from slices1, and one chunk of data
        # from slices2
        virtual_data = create_virtual_dataset(
            f,
            "test_version",
            "test_data",
            ((nchunks1 + 1) * chunksize,),
            {
                **slices1,
                Tuple(
                    Slice(nchunks1 * chunksize, (nchunks1 + 1) * chunksize, 1),
                ): slices2[(Slice(1 * chunksize, 2 * chunksize, 1),)],
            },
        )

        assert virtual_data.shape == ((nchunks1 + 1) * chunksize,)
        assert_equal(virtual_data[0 : 2 * DEFAULT_CHUNK_SIZE], 1.0)
        assert_equal(virtual_data[2 * DEFAULT_CHUNK_SIZE : 3 * DEFAULT_CHUNK_SIZE], 2.0)
        assert virtual_data.dtype == np.float64


@mark.setup_args(version_name="test_version")
def test_create_virtual_dataset_attrs(h5file):
    data1 = np.ones((2 * DEFAULT_CHUNK_SIZE,))
    data2 = np.concatenate(
        (
            2 * np.ones((DEFAULT_CHUNK_SIZE,)),
            3 * np.ones((DEFAULT_CHUNK_SIZE,)),
        )
    )

    # Chunk size is set by the size the first dataset
    chunksize = guess_chunk(data1.shape, None, data1.dtype.itemsize)[0]

    with h5file as f:
        slices1 = write_dataset(f, "test_data", data1)
        slices2 = write_dataset(f, "test_data", data2)

        nchunks1 = int(np.ceil(2 * DEFAULT_CHUNK_SIZE / chunksize))

        attrs = {"attribute": "value"}
        # The virtual dataset contains all the data from slices1, and one chunk of data
        # from slices2
        virtual_data = create_virtual_dataset(
            f,
            "test_version",
            "test_data",
            ((nchunks1 + 1) * chunksize,),
            {
                **slices1,
                Tuple(
                    Slice(nchunks1 * chunksize, (nchunks1 + 1) * chunksize, 1),
                ): slices2[(Slice(1 * chunksize, 2 * chunksize, 1),)],
            },
            attrs=attrs,
        )

        assert dict(virtual_data.attrs) == {
            **attrs,
            "raw_data": "/_version_data/test_data/raw_data",
            "chunks": np.array([chunksize]),
        }


@mark.setup_args(version_name=["test_version1", "test_version2"])
def test_create_virtual_dataset_multidimension(h5file):
    chunks = 3 * (CHUNK_SIZE_3D,)
    shape = (2 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D)
    data = np.ones(shape)
    slices1 = write_dataset(h5file, "test_data", data, chunks=chunks)

    virtual_data = create_virtual_dataset(
        h5file, "test_version1", "test_data", shape, slices1
    )

    assert virtual_data.shape == shape
    assert_equal(virtual_data[:], 1)
    assert virtual_data.dtype == np.float64

    data2 = data.copy()
    for n, (i, j, k) in enumerate(itertools.product([0, 1], repeat=3)):
        data2[
            i * CHUNK_SIZE_3D : (i + 1) * CHUNK_SIZE_3D,
            j * CHUNK_SIZE_3D : (j + 1) * CHUNK_SIZE_3D,
            k * CHUNK_SIZE_3D : (k + 1) * CHUNK_SIZE_3D,
        ] = n

    slices2 = write_dataset(h5file, "test_data", data2, chunks=chunks)

    virtual_data2 = create_virtual_dataset(
        h5file, "test_version2", "test_data", shape, slices2
    )

    assert virtual_data2.shape == shape
    for n, (i, j, k) in enumerate(itertools.product([0, 1], repeat=3)):
        assert_equal(
            virtual_data2[
                i * CHUNK_SIZE_3D : (i + 1) * CHUNK_SIZE_3D,
                j * CHUNK_SIZE_3D : (j + 1) * CHUNK_SIZE_3D,
                k * CHUNK_SIZE_3D : (k + 1) * CHUNK_SIZE_3D,
            ],
            n,
        )
    assert virtual_data2.dtype == np.float64


@mark.setup_args(version_name="test_version")
def test_create_virtual_dataset_offset(h5file):
    data1 = np.ones((2 * DEFAULT_CHUNK_SIZE,))
    data2 = np.concatenate(
        (
            2 * np.ones((DEFAULT_CHUNK_SIZE,)),
            3 * np.ones((DEFAULT_CHUNK_SIZE - 2,)),
        )
    )

    slices1 = write_dataset(h5file, "test_data", data1)
    slices2 = write_dataset(h5file, "test_data", data2)

    # Chunk size is set by the size the first dataset
    chunksize = guess_chunk(data1.shape, None, data1.dtype.itemsize)[0]
    nchunks1 = int(np.ceil(data1.size / chunksize))
    nchunks2 = int(np.ceil(data2.size / chunksize))

    # After writing the data above, there is now 4 chunks in the raw dataset:
    #   raw_data[0*chunksize:1*chunksize] == 1.0
    #   raw_data[1*chunksize:2*chunksize] == 2.0
    #   raw_data[2*chunksize:3*chunksize] == 3.0
    #   raw_data[3*chunksize:4*chunksize-2] == 3.0
    # Create a virtual dataset including all data from the first dataset
    # and the last chunk of data from the second dataset.
    virtual_data = create_virtual_dataset(
        h5file,
        "test_version",
        "test_data",
        ((nchunks1 + 1) * chunksize - 2,),
        {
            **slices1,
            Tuple(
                Slice(nchunks1 * chunksize, (nchunks1 + 1) * chunksize - 2, 1),
            ): slices2[
                (Slice((nchunks2 - 1) * chunksize, nchunks2 * chunksize - 2, 1),)
            ],
        },
    )

    assert virtual_data.shape == ((nchunks1 + 1) * chunksize - 2,)
    assert_equal(virtual_data[0 : nchunks1 * chunksize], 1.0)
    assert_equal(
        virtual_data[nchunks1 * chunksize : (nchunks1 + 1) * chunksize - 2], 3.0
    )


@mark.setup_args(version_name="test_version")
def test_create_virtual_dataset_offset_multidimension(h5file):
    chunks = ChunkSize(3 * (CHUNK_SIZE_3D,))
    shape = (2 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D, 2 * CHUNK_SIZE_3D)
    data = np.zeros(shape)
    write_dataset(h5file, "test_data", data, chunks=chunks)
    shape2 = (2 * CHUNK_SIZE_3D - 2, 2 * CHUNK_SIZE_3D - 2, 2 * CHUNK_SIZE_3D - 2)
    data2 = np.empty(shape2)
    for n, c in enumerate(chunks.indices(shape)):
        data2[c.raw] = n

    slices2 = write_dataset(h5file, "test_data", data2, chunks=chunks)

    virtual_data = create_virtual_dataset(
        h5file, "test_version", "test_data", shape2, slices2
    )

    assert virtual_data.shape == shape2
    assert_equal(virtual_data[()], data2)
    assert virtual_data.dtype == np.float64


def test_write_dataset_chunk_size(h5file):
    chunk_size = 2**10
    chunks = (chunk_size,)
    slices1 = write_dataset(
        h5file, "test_data", np.ones((2 * chunk_size,)), chunks=chunks
    )
    raises(
        ValueError,
        lambda: write_dataset(h5file, "test_data", np.ones(chunks), chunks=(2**9,)),
    )
    slices2 = write_dataset_chunks(
        h5file,
        "test_data",
        {
            Tuple(Slice(0 * chunk_size, 1 * chunk_size, 1)): slices1[
                Tuple(Slice(0 * chunk_size, 1 * chunk_size, 1))
            ],
            Tuple(Slice(1 * chunk_size, 2 * chunk_size, 1)): 2 * np.ones((chunk_size,)),
            Tuple(Slice(2 * chunk_size, 3 * chunk_size, 1)): 2 * np.ones((chunk_size,)),
            Tuple(Slice(3 * chunk_size, 4 * chunk_size, 1)): 3 * np.ones((chunk_size,)),
        },
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
            0 * chunk_size, 1 * chunk_size
        ),
        Tuple(Slice(1 * chunk_size, 2 * chunk_size, 1)): slice(
            1 * chunk_size, 2 * chunk_size
        ),
        Tuple(Slice(2 * chunk_size, 3 * chunk_size, 1)): slice(
            1 * chunk_size, 2 * chunk_size
        ),
        Tuple(Slice(3 * chunk_size, 4 * chunk_size, 1)): slice(
            2 * chunk_size, 3 * chunk_size
        ),
    }

    ds = h5file["/_version_data/test_data/raw_data"]
    assert ds.shape == (3 * chunk_size,)
    assert_equal(ds[0 : 1 * chunk_size], 1.0)
    assert_equal(ds[1 * chunk_size : 2 * chunk_size], 2.0)
    assert_equal(ds[2 * chunk_size : 3 * chunk_size], 3.0)
    assert_equal(ds[3 * chunk_size : 4 * chunk_size], 0.0)
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
        h5file, "test_data", data, compression="gzip", compression_opts=3
    )

    with raises(ValueError):
        write_dataset(
            h5file, "test_data", np.ones((DEFAULT_CHUNK_SIZE,)), compression="lzf"
        )

    with raises(ValueError):
        write_dataset(
            h5file,
            "test_data",
            np.ones((DEFAULT_CHUNK_SIZE,)),
            compression="gzip",
            compression_opts=4,
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


def test_create_empty_virtual_dataset(setup_vfile):
    """Check that creating an empty virtual dataset writes no raw data.

    Also check that the empty virtual dataset is formed correctly.
    See https://github.com/deshaw/versioned-hdf5/issues/314 for context.
    """
    name = "empty_dataset"

    with setup_vfile(version_name="r0") as f:
        write_dataset(f, "empty_dataset", np.array([]))
        create_virtual_dataset(
            f,
            "r0",
            name,
            (0,),
            {},
        )

        # Check that the raw data has only fill_value in it
        assert_equal(f["_version_data"][name]["raw_data"][:], 0.0)

        # Check that the virtual data is empty
        ds = f["_version_data"]["versions"]["r0"][name][:]
        assert_equal(ds, np.array([]))
        assert ds.shape == (0,)
        assert ds.size == 0


@mark.parametrize(("chunk_size"), [2, 5, 10])
def test_last_raw_used_chunk(tmp_path, chunk_size):
    """Test that last_raw_used_chunk returns the slice containing only used elements in the last chunk."""
    filename = tmp_path / "data.h5"
    updates = 30
    chunks = (chunk_size,)

    with h5py.File(filename, "w") as f:
        vf = vhdf.VersionedHDF5File(f)
        with vf.stage_version("r00") as sv:
            sv.create_dataset("values", data=np.array([0]), chunks=chunks)

        assert last_raw_used_chunk(f, "values") == Tuple(Slice(0, 1))

        for i in range(1, updates + 1):
            with vf.stage_version(f"r{i:02d}") as sv:
                sv["values"].append(np.array([i]))

            # Calculate number of chunks; use that to figure out start/end of elements
            # in the last chunk
            nchunks = (i // chunk_size) + 1
            last_chunk_start = (nchunks - 1) * chunk_size
            used_last_chunk_elements = i + 1 - last_chunk_start

            assert last_raw_used_chunk(f, "values") == Tuple(
                Slice(last_chunk_start, last_chunk_start + used_last_chunk_elements)
            )


@mark.parametrize(("chunk_size"), [2, 5, 10])
def test_get_space_remaining(tmp_path, chunk_size):
    """Test that get_space_remaining returns the number of unused elements in the last chunk."""
    filename = tmp_path / "data.h5"
    updates = 30
    chunks = (chunk_size,)

    with h5py.File(filename, "w") as f:
        vf = vhdf.VersionedHDF5File(f)
        with vf.stage_version("r00") as sv:
            sv.create_dataset("values", data=np.array([0]), chunks=chunks)

        assert get_space_remaining(f, "values") == chunk_size - 1

        for i in range(1, updates + 1):
            with vf.stage_version(f"r{i:02d}") as sv:
                sv["values"].append(np.array([i]))

            # Calculate number of chunks; use that to figure out start/end of elements
            # in the last chunk
            nchunks = (i // chunk_size) + 1
            last_chunk_start = (nchunks - 1) * chunk_size
            used_last_chunk_elements = i + 1 - last_chunk_start

            assert (
                get_space_remaining(f, "values")
                == chunk_size - used_last_chunk_elements
            )


def test_split_result():
    """Test basic indexing calculations provided by the SplitResult class."""
    chunk_size = 3

    # Assume chunk_size = 3, existing data = [0]
    split = SplitResult(
        arr_to_append=np.array([1, 2]),
        arr_to_write=np.array([11, 12]),
        new_raw_last_chunk=Tuple(Slice(3, 5)),
        new_raw_last_chunk_data=np.array([11, 12]),
    )

    assert split.has_append_data()
    assert split.has_write_data()

    # Need one more chunk of size 3 to hold arr_to_write
    assert split.get_additional_rchunks_needed(chunk_size) == 1

    # Adding 4 new elements, so new shape is 5
    assert split.get_new_vshape((1,)) == (5,)

    chunks = {"chunks": (3,)}
    mock_raw_data = mock.MagicMock()
    mock_raw_data.shape = (9,)
    mock_raw_data.attrs.__getitem__.side_effect = chunks.__getitem__

    # If the raw data starts with 9 elements, we are adding
    # one new chunk of size 3 for a total new raw shape of (12,)
    assert split.get_new_raw_shape(mock_raw_data) == (12,)

    previous_version_chunks = {Tuple(Slice(0, 1)): Tuple(Slice(6, 7))}

    # virtual_data              raw_data
    # Before append
    # [0] --------------------> [6]
    #
    # After append
    # [0 1 2] ----------------> [6  7  8]
    # [3 4] ------------------> [9  10]
    assert split.get_new_last_vchunk(previous_version_chunks) == Tuple(Slice(0, 3))
    assert split.get_new_last_rchunk(mock_raw_data, previous_version_chunks) == Tuple(
        Slice(6, 9)
    )

    # The target indices in the raw dataset where append data is to be written
    assert split.get_append_rchunk_slice(previous_version_chunks) == Tuple(Slice(7, 9))
