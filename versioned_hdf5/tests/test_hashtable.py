from __future__ import print_function, division

import sys

from pytest import raises

from ..hashtable import Hashtable

from .test_backend import setup

def test_hashtable():
    with setup('test_data') as f:
        h = Hashtable(f, 'test_data')
        assert len(h) == 0
        h[b'\xff'*32] = slice(0, 1)
        assert len(h) == 1
        assert h[b'\xff'*32] == slice(0, 1)
        assert h.largest_index == 1
        assert h.hash_table[0][0].tostring() == b'\xff'*32
        assert tuple(h.hash_table[0][1]) == (0, 1)
        assert h == {b'\xff'*32: slice(0, 1)}

        if sys.version_info[0] == 3:
            with raises(TypeError):
                h['\x01'*32] = slice(0, 1)
        with raises(ValueError):
            h[b'\x01'] = slice(0, 1)
        with raises(TypeError):
            h[b'\x01'*32] = (0, 1)
        with raises(ValueError):
            h[b'\x01'*32] = slice(0, 4, 2)
