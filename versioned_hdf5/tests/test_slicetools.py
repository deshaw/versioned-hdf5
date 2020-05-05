import numpy as np
from numpy.testing import assert_equal

from h5py._hl.selections import Selection

from .helpers import setup
from ..slicetools import split_slice, slice_size, spaceid_to_slice

def test_split_slice():
    chunk = 10
    for start in range(20):
        for stop in range(30):
            for step in range(1, 10):
                s = slice(start, stop, step)
                slices = list(split_slice(s, chunk))
                base = list(range(100)[s])
                assert sum([slice_size(s_) for i, s_ in slices]) ==\
                    slice_size(s), (s, slices)
                pieces = [list(range(i*chunk, (i+1)*chunk)[s_.raw]) for i, s_ in
                          slices]
                extended = []
                for p in pieces:
                    extended.extend(p)
                assert base == extended, (s, slices)

def test_slice_size():
    r = range(1000)
    for start in range(20):
        for stop in range(30):
            for step in range(1, 10):
                s = slice(start, stop, step)
                assert len(r[s]) == slice_size(s)

    s = slice(10)
    assert len(r[s]) == slice_size(s)
    s = slice(1, 10)
    assert len(r[s]) == slice_size(s)


def test_spaceid_to_slice():
    with setup() as f:
        shape = 10
        a = f.create_dataset('a', data=np.arange(shape))

        for start in range(0, shape):
            for count in range(0, shape):
                for stride in range(1, shape):
                    for block in range(0, shape):
                        if count != 1 and block != 1:
                            # Not yet supported. Doesn't seem to be supported
                            # by HDF5 either (?)
                            continue

                        spaceid = a.id.get_space()
                        spaceid.select_hyperslab((start,), (count,),
                                                 (stride,), (block,))
                        sel = Selection((shape,), spaceid)
                        try:
                            a[sel]
                        except ValueError:
                            # HDF5 doesn't allow stride/count combinations
                            # that are impossible (the count must be the exact
                            # number of elements in the selected block).
                            # Rather than trying to enumerate those here, we
                            # just check what doesn't give an error.
                            continue
                        try:
                            s = spaceid_to_slice(spaceid)
                        except:
                            print(start, count, stride, block)
                            raise
                        assert_equal(a[s.raw], a[sel], f"{(start, count, stride, block)}")
