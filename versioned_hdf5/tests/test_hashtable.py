from pytest import raises

import numpy as np

from ..backend import create_base_dataset
from ..hashtable import Hashtable

from .helpers import setup

def test_hashtable():
    with setup() as f:
        create_base_dataset(f, 'test_data', data=np.empty((0,)))
        h = Hashtable(f, 'test_data')
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

def test_hashtable_multidimension():
    # Ensure that the same data with different shape hashes differently
    with setup() as f:
        create_base_dataset(f, 'test_data', data=np.empty((0,)))
        h = Hashtable(f, 'test_data')
        assert h.hash(np.ones((1, 2, 3,))) != h.hash(np.ones((3, 2, 1)))
