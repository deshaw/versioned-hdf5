# TODO: Use a fixture for the test file
import h5py
import numpy as np
from numpy.testing import assert_equal

from ..versions import (create_base_dataset, initialize, write_dataset, create_virtual_dataset, CHUNK_SIZE)

def setup():
    f = h5py.File('test.hdf5', 'w')
    initialize(f)
    return f

def test_initialize():
    with setup():
        pass

def test_create_base_dataset():
    with setup() as f:
        create_base_dataset(f, 'test_data', data=np.ones((CHUNK_SIZE,)))

def test_write_dataset():
    with setup() as f:
        write_dataset(f, 'test_data', np.ones((CHUNK_SIZE,)))
        write_dataset(f, 'test_data', 2*np.ones((CHUNK_SIZE,)))
        write_dataset(f, 'test_data', 3*np.ones((CHUNK_SIZE,)))

        ds = f['/_version_data/raw_data/test_data']
        assert ds.shape == (3*CHUNK_SIZE,)
        assert_equal(ds[0:CHUNK_SIZE], 1.0)
        assert_equal(ds[CHUNK_SIZE], 2.0)
        assert_equal(ds[2*CHUNK_SIZE], 3.0)

def test_create_virtual_dataset():
    with setup() as f:
        write_dataset(f, 'test_data', np.ones((CHUNK_SIZE,)))
        write_dataset(f, 'test_data', 2*np.ones((CHUNK_SIZE,)))
        write_dataset(f, 'test_data', 3*np.ones((CHUNK_SIZE,)))

        virtual_data = create_virtual_dataset(f, 'test_data', (2*CHUNK_SIZE,), [0, 2])

        assert virtual_data.shape == (2*CHUNK_SIZE,)
        assert_equal(virtual_data[0:CHUNK_SIZE], 1.0)
        assert_equal(virtual_data[CHUNK_SIZE:2*CHUNK_SIZE], 3.0)
