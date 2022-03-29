import numpy as np
from numpy.testing import assert_equal

from ndindex import Slice, Tuple, ChunkSize

from pytest import mark, raises

import itertools

from .helpers import setup_vfile

from ..backend import (create_base_dataset, write_dataset,
                       create_virtual_dataset, DEFAULT_CHUNK_SIZE,
                       write_dataset_chunks)

CHUNK_SIZE_3D = 2**4  # = cbrt(DEFAULT_CHUNK_SIZE)


def test_initialize():
    with setup_vfile() as f:
        pass
    f.close()


def test_create_base_dataset(h5file):
    create_base_dataset(h5file, 'test_data', data=np.ones((DEFAULT_CHUNK_SIZE,)))
    assert h5file['_version_data/test_data/raw_data'].dtype == np.float64


def test_create_base_dataset_multidimension(h5file):
    create_base_dataset(h5file, 'test_data', data=np.ones((CHUNK_SIZE_3D, CHUNK_SIZE_3D, 2)),
                        chunks=(CHUNK_SIZE_3D, CHUNK_SIZE_3D, CHUNK_SIZE_3D))
    assert h5file['_version_data/test_data/raw_data'].dtype == np.float64


def test_write_dataset(h5file):
    slices1 = write_dataset(h5file, 'test_data', np.ones((2*DEFAULT_CHUNK_SIZE,)))
    slices2 = write_dataset(h5file, 'test_data',
                            np.concatenate((2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                            2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                            3*np.ones((DEFAULT_CHUNK_SIZE,)),
                            )))

    assert slices1 == {
        (Slice(0*DEFAULT_CHUNK_SIZE, 1*DEFAULT_CHUNK_SIZE, 1),):
            slice(0*DEFAULT_CHUNK_SIZE, 1*DEFAULT_CHUNK_SIZE),
        (Slice(1*DEFAULT_CHUNK_SIZE, 2*DEFAULT_CHUNK_SIZE, 1),):
            slice(0*DEFAULT_CHUNK_SIZE, 1*DEFAULT_CHUNK_SIZE)}
    assert slices2 == {
        (Slice(0*DEFAULT_CHUNK_SIZE, 1*DEFAULT_CHUNK_SIZE, 1),):
            slice(1*DEFAULT_CHUNK_SIZE, 2*DEFAULT_CHUNK_SIZE),
        (Slice(1*DEFAULT_CHUNK_SIZE, 2*DEFAULT_CHUNK_SIZE, 1),):
            slice(1*DEFAULT_CHUNK_SIZE, 2*DEFAULT_CHUNK_SIZE),
        (Slice(2*DEFAULT_CHUNK_SIZE, 3*DEFAULT_CHUNK_SIZE, 1),):
            slice(2*DEFAULT_CHUNK_SIZE, 3*DEFAULT_CHUNK_SIZE)}

    ds = h5file['/_version_data/test_data/raw_data']
    assert ds.shape == (3*DEFAULT_CHUNK_SIZE,)
    assert_equal(ds[0:1*DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(ds[1*DEFAULT_CHUNK_SIZE:2*DEFAULT_CHUNK_SIZE], 2.0)
    assert_equal(ds[2*DEFAULT_CHUNK_SIZE:3*DEFAULT_CHUNK_SIZE], 3.0)
    assert_equal(ds[3*DEFAULT_CHUNK_SIZE:4*DEFAULT_CHUNK_SIZE], 0.0)
    assert ds.dtype == np.float64


def test_write_dataset_multidimension(h5file):
    chunks = 3*(CHUNK_SIZE_3D,)
    data = np.zeros((2*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D))
    slices1 = write_dataset(h5file, 'test_data', data, chunks=chunks)
    data2 = data.copy()
    for n, (i, j, k) in enumerate(itertools.product([0, 1], repeat=3)):
        data2[i*CHUNK_SIZE_3D:(i+1)*CHUNK_SIZE_3D,
              j*CHUNK_SIZE_3D:(j+1)*CHUNK_SIZE_3D,
              k*CHUNK_SIZE_3D:(k+1)*CHUNK_SIZE_3D] = n

    slices2 = write_dataset(h5file, 'test_data', data2, chunks=chunks)

    assert slices1 == {
        (Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         ): slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D),
        (Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         ): slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D),
        (Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         ): slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D),
        (Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         ): slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D),
        (Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         ): slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D),
        (Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         ): slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D),
        (Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         ): slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D),
        (Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         ): slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D),
    }
    assert slices2 == {
        (Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         ): slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D),
        (Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         ): slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D),
        (Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         ): slice(2*CHUNK_SIZE_3D, 3*CHUNK_SIZE_3D),
        (Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         ): slice(3*CHUNK_SIZE_3D, 4*CHUNK_SIZE_3D),
        (Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         ): slice(4*CHUNK_SIZE_3D, 5*CHUNK_SIZE_3D),
        (Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         ): slice(5*CHUNK_SIZE_3D, 6*CHUNK_SIZE_3D),
        (Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         ): slice(6*CHUNK_SIZE_3D, 7*CHUNK_SIZE_3D),
        (Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 1),
         ): slice(7*CHUNK_SIZE_3D, 8*CHUNK_SIZE_3D),
    }

    ds = h5file['/_version_data/test_data/raw_data']
    assert ds.shape == (8*CHUNK_SIZE_3D, CHUNK_SIZE_3D, CHUNK_SIZE_3D)
    for n in range(8):
        assert_equal(ds[n*CHUNK_SIZE_3D:(n+1)*CHUNK_SIZE_3D], n)
    assert ds.dtype == np.float64


def test_write_dataset_chunks(h5file):
    slices1 = write_dataset(h5file, 'test_data', np.ones((2*DEFAULT_CHUNK_SIZE,)))
    slices2 = write_dataset_chunks(h5file, 'test_data', {
        Tuple(Slice(0*DEFAULT_CHUNK_SIZE, 1*DEFAULT_CHUNK_SIZE, 1)):
            slices1[Tuple(Slice(0*DEFAULT_CHUNK_SIZE, 1*DEFAULT_CHUNK_SIZE, 1))],
        Tuple(Slice(1*DEFAULT_CHUNK_SIZE, 2*DEFAULT_CHUNK_SIZE, 1)): 2*np.ones((DEFAULT_CHUNK_SIZE,)),
        Tuple(Slice(2*DEFAULT_CHUNK_SIZE, 3*DEFAULT_CHUNK_SIZE, 1)): 2*np.ones((DEFAULT_CHUNK_SIZE,)),
        Tuple(Slice(3*DEFAULT_CHUNK_SIZE, 4*DEFAULT_CHUNK_SIZE, 1)): 3*np.ones((DEFAULT_CHUNK_SIZE,)),
    })

    assert slices1 == {
        Tuple(Slice(0*DEFAULT_CHUNK_SIZE, 1*DEFAULT_CHUNK_SIZE, 1)):
            slice(0*DEFAULT_CHUNK_SIZE, 1*DEFAULT_CHUNK_SIZE),
        Tuple(Slice(1*DEFAULT_CHUNK_SIZE, 2*DEFAULT_CHUNK_SIZE, 1)):
            slice(0*DEFAULT_CHUNK_SIZE, 1*DEFAULT_CHUNK_SIZE),
        }
    assert slices2 == {
        Tuple(Slice(0*DEFAULT_CHUNK_SIZE, 1*DEFAULT_CHUNK_SIZE, 1)):
            slice(0*DEFAULT_CHUNK_SIZE, 1*DEFAULT_CHUNK_SIZE),
        Tuple(Slice(1*DEFAULT_CHUNK_SIZE, 2*DEFAULT_CHUNK_SIZE, 1)):
            slice(1*DEFAULT_CHUNK_SIZE, 2*DEFAULT_CHUNK_SIZE),
        Tuple(Slice(2*DEFAULT_CHUNK_SIZE, 3*DEFAULT_CHUNK_SIZE, 1)):
            slice(1*DEFAULT_CHUNK_SIZE, 2*DEFAULT_CHUNK_SIZE),
        Tuple(Slice(3*DEFAULT_CHUNK_SIZE, 4*DEFAULT_CHUNK_SIZE, 1)):
            slice(2*DEFAULT_CHUNK_SIZE, 3*DEFAULT_CHUNK_SIZE),
        }

    ds = h5file['/_version_data/test_data/raw_data']
    assert ds.shape == (3*DEFAULT_CHUNK_SIZE,)
    assert_equal(ds[0:1*DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(ds[1*DEFAULT_CHUNK_SIZE:2*DEFAULT_CHUNK_SIZE], 2.0)
    assert_equal(ds[2*DEFAULT_CHUNK_SIZE:3*DEFAULT_CHUNK_SIZE], 3.0)
    assert_equal(ds[3*DEFAULT_CHUNK_SIZE:4*DEFAULT_CHUNK_SIZE], 0.0)
    assert ds.dtype == np.float64


def test_write_dataset_chunks_multidimension(h5file):
    chunks = ChunkSize(3*(CHUNK_SIZE_3D,))
    shape = (2*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D)
    data = np.zeros(shape)
    slices1 = write_dataset(h5file, 'test_data', data, chunks=chunks)
    data_dict = {}
    for n, c in enumerate(chunks.indices(shape)):
        if n == 0:
            data_dict[c] = slices1[c]
        else:
            data_dict[c] = n*np.ones(chunks)

    slices1 = write_dataset(h5file, 'test_data', data, chunks=chunks)
    slices2 = write_dataset_chunks(h5file, 'test_data', data_dict)

    assert slices1 == {c: slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D) for c in
                       chunks.indices(shape)}
    assert slices2 == {c: slice(i*CHUNK_SIZE_3D, (i+1)*CHUNK_SIZE_3D) for
                       i, c in enumerate(chunks.indices(shape))}

    ds = h5file['/_version_data/test_data/raw_data']
    assert ds.shape == (8*CHUNK_SIZE_3D, CHUNK_SIZE_3D, CHUNK_SIZE_3D)
    for n in range(8):
        assert_equal(ds[n*CHUNK_SIZE_3D:(n+1)*CHUNK_SIZE_3D], n)
    assert ds.dtype == np.float64


def test_write_dataset_offset(h5file):
    slices1 = write_dataset(h5file, 'test_data', np.ones((2*DEFAULT_CHUNK_SIZE,)))
    slices2 = write_dataset(h5file, 'test_data',
                            np.concatenate((2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                            2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                            3*np.ones((DEFAULT_CHUNK_SIZE - 2,)))))

    assert slices1 == {
        (Slice(0*DEFAULT_CHUNK_SIZE, 1*DEFAULT_CHUNK_SIZE, 1),):
            slice(0*DEFAULT_CHUNK_SIZE, 1*DEFAULT_CHUNK_SIZE),
        (Slice(1*DEFAULT_CHUNK_SIZE, 2*DEFAULT_CHUNK_SIZE, 1),):
            slice(0*DEFAULT_CHUNK_SIZE, 1*DEFAULT_CHUNK_SIZE)}
    assert slices2 == {
        (Slice(0*DEFAULT_CHUNK_SIZE, 1*DEFAULT_CHUNK_SIZE, 1),):
            slice(1*DEFAULT_CHUNK_SIZE, 2*DEFAULT_CHUNK_SIZE),
        (Slice(1*DEFAULT_CHUNK_SIZE, 2*DEFAULT_CHUNK_SIZE, 1),):
            slice(1*DEFAULT_CHUNK_SIZE, 2*DEFAULT_CHUNK_SIZE),
        (Slice(2*DEFAULT_CHUNK_SIZE, 3*DEFAULT_CHUNK_SIZE - 2, 1),):
            slice(2*DEFAULT_CHUNK_SIZE, 3*DEFAULT_CHUNK_SIZE - 2)}

    ds = h5file['/_version_data/test_data/raw_data']
    assert ds.shape == (3*DEFAULT_CHUNK_SIZE,)
    assert_equal(ds[0*DEFAULT_CHUNK_SIZE:1*DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(ds[1*DEFAULT_CHUNK_SIZE:2*DEFAULT_CHUNK_SIZE], 2.0)
    assert_equal(ds[2*DEFAULT_CHUNK_SIZE:3*DEFAULT_CHUNK_SIZE - 2], 3.0)
    assert_equal(ds[3*DEFAULT_CHUNK_SIZE-2:4*DEFAULT_CHUNK_SIZE], 0.0)


def test_write_dataset_offset_multidimension(h5file):
    chunks = ChunkSize(3*(CHUNK_SIZE_3D,))
    shape = (2*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D)
    data = np.zeros(shape)
    slices1 = write_dataset(h5file, 'test_data', data, chunks=chunks)
    shape2 = (2*CHUNK_SIZE_3D - 2, 2*CHUNK_SIZE_3D - 2,
              2*CHUNK_SIZE_3D - 2)
    data2 = np.empty(shape2)
    for n, c in enumerate(chunks.indices(shape)):
        data2[c.raw] = n

    slices2 = write_dataset(h5file, 'test_data', data2, chunks=chunks)

    assert slices1 == {
        (Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
        ): slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D),
        (Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D , 1),
        ): slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D),
        (Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D , 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
        ): slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D),
        (Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D , 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D , 1),
        ): slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D),
        (Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D , 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
        ): slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D),
        (Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D , 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D , 1),
        ): slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D),
        (Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D , 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D , 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
        ): slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D),
        (Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D , 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D , 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D , 1),
        ): slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D),
    }

    assert slices2 == {
        (Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
        ): slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D),
        (Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D - 2, 1),
        ): slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D),
        (Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D - 2, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
        ): slice(2*CHUNK_SIZE_3D, 3*CHUNK_SIZE_3D),
        (Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D - 2, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D - 2, 1),
        ): slice(3*CHUNK_SIZE_3D, 4*CHUNK_SIZE_3D),
        (Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D - 2, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
        ): slice(4*CHUNK_SIZE_3D, 5*CHUNK_SIZE_3D - 2),
        (Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D - 2, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D - 2, 1),
        ): slice(5*CHUNK_SIZE_3D, 6*CHUNK_SIZE_3D - 2),
        (Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D - 2, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D - 2, 1),
         Slice(0*CHUNK_SIZE_3D, 1*CHUNK_SIZE_3D, 1),
        ): slice(6*CHUNK_SIZE_3D, 7*CHUNK_SIZE_3D - 2),
        (Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D - 2, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D - 2, 1),
         Slice(1*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D - 2, 1),
        ): slice(7*CHUNK_SIZE_3D, 8*CHUNK_SIZE_3D - 2),
    }

    ds = h5file['/_version_data/test_data/raw_data']
    assert ds.shape == (8*CHUNK_SIZE_3D, CHUNK_SIZE_3D, CHUNK_SIZE_3D)
    for n, c in enumerate(chunks.indices(shape2)):
        a = np.zeros(chunks)
        a[Tuple(*[slice(0, i) for i in shape2]).as_subindex(c).raw] = n
        assert_equal(ds[n*CHUNK_SIZE_3D:(n+1)*CHUNK_SIZE_3D], a)
    assert ds.dtype == np.float64


@mark.setup_args(version_name='test_version')
def test_create_virtual_dataset(h5file):
    with h5file as f:
        slices1 = write_dataset(f, 'test_data', np.ones((2*DEFAULT_CHUNK_SIZE,)))
        slices2 = write_dataset(f, 'test_data',
                                np.concatenate((2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                                3*np.ones((DEFAULT_CHUNK_SIZE,)))))

        virtual_data = create_virtual_dataset(f, 'test_version', 'test_data', (3*DEFAULT_CHUNK_SIZE,),
            {**slices1,
             Tuple(Slice(2*DEFAULT_CHUNK_SIZE, 3*DEFAULT_CHUNK_SIZE, 1),):
             slices2[(Slice(1*DEFAULT_CHUNK_SIZE, 2*DEFAULT_CHUNK_SIZE, 1),)]})

        assert virtual_data.shape == (3*DEFAULT_CHUNK_SIZE,)
        assert_equal(virtual_data[0:2*DEFAULT_CHUNK_SIZE], 1.0)
        assert_equal(virtual_data[2*DEFAULT_CHUNK_SIZE:3*DEFAULT_CHUNK_SIZE], 3.0)
        assert virtual_data.dtype == np.float64

@mark.setup_args(version_name='test_version')
def test_create_virtual_dataset_attrs(h5file):
    with h5file as f:
        slices1 = write_dataset(f, 'test_data', np.ones((2*DEFAULT_CHUNK_SIZE,)))
        slices2 = write_dataset(f, 'test_data',
                                np.concatenate((2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                                3*np.ones((DEFAULT_CHUNK_SIZE,)))))

        attrs = {"attribute": "value"}
        virtual_data = create_virtual_dataset(f, 'test_version', 'test_data', (3*DEFAULT_CHUNK_SIZE,),
            {**slices1,
             Tuple(Slice(2*DEFAULT_CHUNK_SIZE, 3*DEFAULT_CHUNK_SIZE, 1),):
             slices2[(Slice(1*DEFAULT_CHUNK_SIZE, 2*DEFAULT_CHUNK_SIZE,
                            1),)]}, attrs=attrs)

        assert dict(virtual_data.attrs) == {**attrs, "raw_data": '/_version_data/test_data/raw_data', "chunks": np.array([DEFAULT_CHUNK_SIZE])}

@mark.setup_args(version_name=['test_version1', 'test_version2'])
def test_create_virtual_dataset_multidimension(h5file):
    chunks = 3*(CHUNK_SIZE_3D,)
    shape = (2*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D)
    data = np.ones(shape)
    slices1 = write_dataset(h5file, 'test_data', data, chunks=chunks)

    virtual_data = create_virtual_dataset(h5file, 'test_version1',
                                          'test_data', shape, slices1)

    assert virtual_data.shape == shape
    assert_equal(virtual_data[:], 1)
    assert virtual_data.dtype == np.float64

    data2 = data.copy()
    for n, (i, j, k) in enumerate(itertools.product([0, 1], repeat=3)):
        data2[i*CHUNK_SIZE_3D:(i+1)*CHUNK_SIZE_3D,
              j*CHUNK_SIZE_3D:(j+1)*CHUNK_SIZE_3D,
              k*CHUNK_SIZE_3D:(k+1)*CHUNK_SIZE_3D] = n

    slices2 = write_dataset(h5file, 'test_data', data2, chunks=chunks)

    virtual_data2 = create_virtual_dataset(h5file, 'test_version2',
                                           'test_data', shape, slices2)

    assert virtual_data2.shape == shape
    for n, (i, j, k) in enumerate(itertools.product([0, 1], repeat=3)):
        assert_equal(virtual_data2[i*CHUNK_SIZE_3D:(i+1)*CHUNK_SIZE_3D,
                                   j*CHUNK_SIZE_3D:(j+1)*CHUNK_SIZE_3D,
                                   k*CHUNK_SIZE_3D:(k+1)*CHUNK_SIZE_3D], n)
    assert virtual_data2.dtype == np.float64


@mark.setup_args(version_name='test_version')
def test_create_virtual_dataset_offset(h5file):
    slices1 = write_dataset(h5file, 'test_data', np.ones((2*DEFAULT_CHUNK_SIZE,)))
    slices2 = write_dataset(h5file, 'test_data',
                            np.concatenate((2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                            3*np.ones((DEFAULT_CHUNK_SIZE - 2,)))))

    virtual_data = create_virtual_dataset(h5file, 'test_version', 'test_data',
                                          (3*DEFAULT_CHUNK_SIZE - 2,),
        {**slices1,
         Tuple(Slice(2*DEFAULT_CHUNK_SIZE, 3*DEFAULT_CHUNK_SIZE - 2, 1),):
         slices2[(Slice(1*DEFAULT_CHUNK_SIZE, 2*DEFAULT_CHUNK_SIZE - 2, 1),)]})

    assert virtual_data.shape == (3*DEFAULT_CHUNK_SIZE - 2,)
    assert_equal(virtual_data[0:2*DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(virtual_data[2*DEFAULT_CHUNK_SIZE:3*DEFAULT_CHUNK_SIZE - 2], 3.0)


@mark.setup_args(version_name='test_version')
def test_create_virtual_dataset_offset_multidimension(h5file):
    chunks = ChunkSize(3*(CHUNK_SIZE_3D,))
    shape = (2*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D, 2*CHUNK_SIZE_3D)
    data = np.zeros(shape)
    write_dataset(h5file, 'test_data', data, chunks=chunks)
    shape2 = (2*CHUNK_SIZE_3D - 2, 2*CHUNK_SIZE_3D - 2,
              2*CHUNK_SIZE_3D - 2)
    data2 = np.empty(shape2)
    for n, c in enumerate(chunks.indices(shape)):
        data2[c.raw] = n

    slices2 = write_dataset(h5file, 'test_data', data2, chunks=chunks)

    virtual_data = create_virtual_dataset(h5file, 'test_version', 'test_data',
                                          shape2, slices2)

    assert virtual_data.shape == shape2
    assert_equal(virtual_data[()], data2)
    assert virtual_data.dtype == np.float64


def test_write_dataset_chunk_size(h5file):
    chunk_size = 2**10
    chunks = (chunk_size,)
    slices1 = write_dataset(h5file, 'test_data', np.ones((2*chunk_size,)),
                            chunks=chunks)
    raises(ValueError, lambda: write_dataset(h5file, 'test_data',
                                             np.ones(chunks), chunks=(2**9,)))
    slices2 = write_dataset_chunks(h5file, 'test_data', {
        Tuple(Slice(0*chunk_size, 1*chunk_size, 1)):
            slices1[Tuple(Slice(0*chunk_size, 1*chunk_size, 1))],
        Tuple(Slice(1*chunk_size, 2*chunk_size, 1)): 2*np.ones((chunk_size,)),
        Tuple(Slice(2*chunk_size, 3*chunk_size, 1)): 2*np.ones((chunk_size,)),
        Tuple(Slice(3*chunk_size, 4*chunk_size, 1)): 3*np.ones((chunk_size,)),
    })

    assert slices1 == {
        Tuple(Slice(0*chunk_size, 1*chunk_size, 1)):
            slice(0*chunk_size, 1*chunk_size),
        Tuple(Slice(1*chunk_size, 2*chunk_size, 1)):
            slice(0*chunk_size, 1*chunk_size),
        }
    assert slices2 == {
        Tuple(Slice(0*chunk_size, 1*chunk_size, 1)):
            slice(0*chunk_size, 1*chunk_size),
        Tuple(Slice(1*chunk_size, 2*chunk_size, 1)):
            slice(1*chunk_size, 2*chunk_size),
        Tuple(Slice(2*chunk_size, 3*chunk_size, 1)):
            slice(1*chunk_size, 2*chunk_size),
        Tuple(Slice(3*chunk_size, 4*chunk_size, 1)):
            slice(2*chunk_size, 3*chunk_size),
        }

    ds = h5file['/_version_data/test_data/raw_data']
    assert ds.shape == (3*chunk_size,)
    assert_equal(ds[0:1*chunk_size], 1.0)
    assert_equal(ds[1*chunk_size:2*chunk_size], 2.0)
    assert_equal(ds[2*chunk_size:3*chunk_size], 3.0)
    assert_equal(ds[3*chunk_size:4*chunk_size], 0.0)
    assert ds.dtype == np.float64


def test_write_dataset_offset_chunk_size(h5file):
    chunk_size = 2**10
    chunks = (chunk_size,)
    slices1 = write_dataset(h5file, 'test_data', 1*np.ones((2*chunk_size,)), chunks=chunks)
    slices2 = write_dataset(h5file, 'test_data',
                            np.concatenate((2*np.ones(chunks),
                                            2*np.ones(chunks),
                                            3*np.ones((chunk_size - 2,)))))

    assert slices1 == {
        Tuple(Slice(0*chunk_size, 1*chunk_size, 1)):
            slice(0*chunk_size, 1*chunk_size),
        Tuple(Slice(1*chunk_size, 2*chunk_size, 1)):
            slice(0*chunk_size, 1*chunk_size),
        }
    assert slices2 == {
        Tuple(Slice(0*chunk_size, 1*chunk_size, 1)):
            slice(1*chunk_size, 2*chunk_size),
        Tuple(Slice(1*chunk_size, 2*chunk_size, 1)):
            slice(1*chunk_size, 2*chunk_size),
        Tuple(Slice(2*chunk_size, 3*chunk_size - 2, 1)):
            slice(2*chunk_size, 3*chunk_size - 2),
        }

    ds = h5file['/_version_data/test_data/raw_data']
    assert ds.shape == (3*chunk_size,)
    assert_equal(ds[0*chunk_size:1*chunk_size], 1.0)
    assert_equal(ds[1*chunk_size:2*chunk_size], 2.0)
    assert_equal(ds[2*chunk_size:3*chunk_size - 2], 3.0)
    assert_equal(ds[3*chunk_size-2:4*chunk_size], 0.0)


def test_write_dataset_compression(h5file):
    slices1 = write_dataset(h5file, 'test_data',
                            np.ones((2*DEFAULT_CHUNK_SIZE,)),
                            compression='gzip', compression_opts=3)
    raises(ValueError, lambda: write_dataset(h5file, 'test_data',
        np.ones((DEFAULT_CHUNK_SIZE,)), compression='lzf'))
    raises(ValueError, lambda: write_dataset(h5file, 'test_data',
        np.ones((DEFAULT_CHUNK_SIZE,)), compression='gzip', compression_opts=4))

    assert slices1 == {
        (Slice(0*DEFAULT_CHUNK_SIZE, 1*DEFAULT_CHUNK_SIZE, 1),):
            slice(0*DEFAULT_CHUNK_SIZE, 1*DEFAULT_CHUNK_SIZE),
        (Slice(1*DEFAULT_CHUNK_SIZE, 2*DEFAULT_CHUNK_SIZE, 1),):
            slice(0*DEFAULT_CHUNK_SIZE, 1*DEFAULT_CHUNK_SIZE)}

    ds = h5file['/_version_data/test_data/raw_data']
    assert ds.shape == (1*DEFAULT_CHUNK_SIZE,)
    assert_equal(ds[0:1*DEFAULT_CHUNK_SIZE], 1.0)
    assert ds.dtype == np.float64
    assert ds.compression == 'gzip'
    assert ds.compression_opts == 3
