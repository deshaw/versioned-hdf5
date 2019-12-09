# TODO: Use a fixture for the test file
import h5py
import numpy as np

from ..versions import create_base_dataset, initialize

def test_create_base_dataset():
    f = h5py.File('test.hdf5', 'w')
    initialize(f)
    create_base_dataset(f, 'test_data', data=np.ones((1024,)))
