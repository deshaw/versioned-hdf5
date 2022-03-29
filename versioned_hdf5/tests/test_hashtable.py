from pytest import raises

import numpy as np
import h5py

from ..backend import create_base_dataset
from ..hashtable import Hashtable
from .helpers import setup_vfile
from .. import VersionedHDF5File

def test_hashtable(h5file):
    create_base_dataset(h5file, 'test_data', data=np.empty((0,)))
    with Hashtable(h5file, 'test_data') as h:
        assert len(h) == 0
        h[b'\xff'*32] = slice(0, 1)
        assert len(h) == 1
        assert h[b'\xff'*32] == slice(0, 1)
        assert h.largest_index == 1
        assert bytes(h.hash_table[0][0]) == b'\xff'*32
        assert tuple(h.hash_table[0][1]) == (0, 1)
        assert h == {b'\xff'*32: slice(0, 1)}

        with raises(TypeError):
            h['\x01'*32] = slice(0, 1)
        with raises(ValueError):
            h[b'\x01'] = slice(0, 1)
        with raises(TypeError):
            h[b'\x01'*32] = (0, 1)
        with raises(ValueError):
            h[b'\x01'*32] = slice(0, 4, 2)

def test_from_raw_data():
    with setup_vfile('test.h5') as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version('0') as sv:
            sv.create_dataset('test_data', data=np.arange(100), chunks=(10,))

        h = Hashtable(f, 'test_data')
        h_dataset = h.hash_table_dataset
        h2 = Hashtable.from_raw_data(f, 'test_data',
                                     hash_table_name='test_hash_table')
        h2_dataset = h2.hash_table_dataset
        assert h2_dataset.name == '/_version_data/test_data/test_hash_table'
        np.testing.assert_equal(h_dataset[:], h2_dataset[:])

def test_hashtable_multidimension(h5file):
    # Ensure that the same data with different shape hashes differently
    create_base_dataset(h5file, 'test_data', data=np.empty((0,)))
    h = Hashtable(h5file, 'test_data')
    assert h.hash(np.ones((1, 2, 3,))) != h.hash(np.ones((3, 2, 1)))

def test_issue_208():
    with setup_vfile('test.h5') as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version('0') as sv:
            sv.create_dataset('bar', data=np.arange(10))

    with h5py.File('test.h5', 'r+') as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version('1') as sv:
            sv['bar'].resize((12,))
            sv['bar'][8:12] = sv['bar'][6:10]
            sv['bar'][6:8] = [0, 0]
