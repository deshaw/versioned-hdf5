# TODO: Use a fixture for the test file
import h5py
import numpy as np
from numpy.testing import assert_equal

from ..versions import (create_base_dataset, initialize, write_dataset,
                        create_virtual_dataset, CHUNK_SIZE, hashtable)

def setup(name=None):
    f = h5py.File('test.hdf5', 'w')
    initialize(f)
    if name:
        f['_version_data'].create_group(name)
    return f

def test_initialize():
    with setup():
        pass

def test_create_base_dataset():
    with setup() as f:
        create_base_dataset(f, 'test_data', data=np.ones((CHUNK_SIZE,)))

def test_write_dataset():
    with setup() as f:
        slices1 = write_dataset(f, 'test_data', np.ones((2*CHUNK_SIZE,)))
        slices2 = write_dataset(f, 'test_data',
                                np.concatenate((2*np.ones((CHUNK_SIZE,)),
                                                3*np.ones((CHUNK_SIZE,)))))

        assert slices1 == [slice(0*CHUNK_SIZE, 1*CHUNK_SIZE),
                           slice(1*CHUNK_SIZE, 2*CHUNK_SIZE)]
        assert slices2 == [slice(2*CHUNK_SIZE, 3*CHUNK_SIZE),
                           slice(3*CHUNK_SIZE, 4*CHUNK_SIZE)]

        ds = f['/_version_data/test_data/raw_data']
        assert ds.shape == (4*CHUNK_SIZE,)
        assert_equal(ds[0:2*CHUNK_SIZE], 1.0)
        assert_equal(ds[2*CHUNK_SIZE:3*CHUNK_SIZE], 2.0)
        assert_equal(ds[3*CHUNK_SIZE:4*CHUNK_SIZE], 3.0)

def test_write_dataset_offset():
    with setup() as f:
        slices1 = write_dataset(f, 'test_data', np.ones((2*CHUNK_SIZE,)))
        slices2 = write_dataset(f, 'test_data',
                                np.concatenate((2*np.ones((CHUNK_SIZE,)),
                                                3*np.ones((CHUNK_SIZE - 2,)))))

        assert slices1 == [slice(0*CHUNK_SIZE, 1*CHUNK_SIZE),
                           slice(1*CHUNK_SIZE, 2*CHUNK_SIZE)]
        assert slices2 == [slice(2*CHUNK_SIZE, 3*CHUNK_SIZE),
                           slice(3*CHUNK_SIZE, 4*CHUNK_SIZE - 2)]

        ds = f['/_version_data/test_data/raw_data']
        assert ds.shape == (4*CHUNK_SIZE,)
        assert_equal(ds[0*CHUNK_SIZE:2*CHUNK_SIZE], 1.0)
        assert_equal(ds[2*CHUNK_SIZE:3*CHUNK_SIZE], 2.0)
        assert_equal(ds[3*CHUNK_SIZE:4*CHUNK_SIZE - 2], 3.0)


def test_create_virtual_dataset():
    with setup() as f:
        slices1 = write_dataset(f, 'test_data', np.ones((CHUNK_SIZE,)))
        slices2 = write_dataset(f, 'test_data',
                                np.concatenate((2*np.ones((CHUNK_SIZE,)),
                                                3*np.ones((CHUNK_SIZE,)))))

        virtual_data = create_virtual_dataset(f, 'test_data',
                                              slices1 + [slices2[1]])

        assert virtual_data.shape == (2*CHUNK_SIZE,)
        assert_equal(virtual_data[0:CHUNK_SIZE], 1.0)
        assert_equal(virtual_data[CHUNK_SIZE:2*CHUNK_SIZE], 3.0)


def test_create_virtual_dataset_offset():
    with setup() as f:
        slices1 = write_dataset(f, 'test_data', np.ones((CHUNK_SIZE,)))
        slices2 = write_dataset(f, 'test_data',
                                np.concatenate((2*np.ones((CHUNK_SIZE,)),
                                                3*np.ones((CHUNK_SIZE - 2,)))))

        virtual_data = create_virtual_dataset(f, 'test_data',
                                              slices1 + [slices2[1]])

        assert virtual_data.shape == (2*CHUNK_SIZE - 2,)
        assert_equal(virtual_data[0:CHUNK_SIZE], 1.0)
        assert_equal(virtual_data[CHUNK_SIZE:2*CHUNK_SIZE - 2], 3.0)

def test_hashtable():
    with setup('test_data') as f:
        h = hashtable(f, 'test_data')
        assert len(h) == 0
        h[b'\xff'*32] = (0, 1)
        assert len(h) == 1
        assert h[b'\xff'*32] == (0, 1)
        assert h.largest_index == 1
        assert bytes(h.hash_table[0][0]) == b'\xff'*32
        assert tuple(h.hash_table[0][1]) == (0, 1)
        assert h == {b'\xff'*32: (0, 1)}
