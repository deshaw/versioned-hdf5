from pytest import raises

import numpy as np
import h5py

from ..backend import create_base_dataset
from ..hashtable import Hashtable
from .helpers import setup
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


def test_hashtable_multidimension(h5file):
    # Ensure that the same data with different shape hashes differently
    create_base_dataset(h5file, 'test_data', data=np.empty((0,)))
    h = Hashtable(h5file, 'test_data')
    assert h.hash(np.ones((1, 2, 3,))) != h.hash(np.ones((3, 2, 1)))

def test_issue_208():
    with setup('test.h5') as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version('0') as sv:
            sv.create_dataset('bar', data=np.arange(10))

    with h5py.File('test.h5', 'r+') as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version('1') as sv:
            sv['bar'].resize((12,))
            sv['bar'][8:12] = sv['bar'][6:10]
            sv['bar'][6:8] = [0, 0]
