import itertools

import numpy as np
import pytest
from h5py._hl.filters import guess_chunk
from ndindex import ChunkSize, Slice, Tuple
from numpy.testing import assert_equal

from versioned_hdf5.backend import (
    DEFAULT_CHUNK_SIZE,
    Filters,
    _data_v4_to_sc_hash_table,
    create_base_dataset,
    create_virtual_dataset,
    write_dataset,
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


@pytest.mark.setup_args(version_name="test_version")
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


@pytest.mark.setup_args(version_name="test_version")
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


@pytest.mark.setup_args(version_name=["test_version1", "test_version2"])
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


@pytest.mark.setup_args(version_name="test_version")
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
    #   raw_data[0*chunksize:1*chunksize] == 1.0  # noqa: ERA001
    #   raw_data[1*chunksize:2*chunksize] == 2.0  # noqa: ERA001
    #   raw_data[2*chunksize:3*chunksize] == 3.0  # noqa: ERA001
    #   raw_data[3*chunksize:4*chunksize-2] == 3.0  # noqa: ERA001
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


@pytest.mark.setup_args(version_name="test_version")
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


def test_create_empty_multidimensional_virtual_dataset(setup_vfile):
    """Check that creating an empty multidimensional virtual dataset writes no raw data.

    See https://github.com/deshaw/versioned-hdf5/issues/430 for context.
    """
    name = "empty_dataset"

    with setup_vfile(version_name="r0") as f:
        write_dataset(f, name, np.array([[]]), chunks=(100, 100))
        create_virtual_dataset(
            f,
            "r0",
            name,
            (0, 0),
            {},
        )

        # Check that the raw data has only fill_value in it
        assert_equal(f["_version_data"][name]["raw_data"][:], 0.0)

        # Check that the virtual data is empty
        ds = f["_version_data"]["versions"]["r0"][name][:]
        assert_equal(ds, np.zeros((0, 0)))
        assert ds.shape == (0, 0)
        assert ds.size == 0


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
