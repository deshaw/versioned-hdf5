import numpy as np
from numpy.testing import assert_equal

from ndindex import Slice

from h5py._hl.selections import Selection

from ..slicetools import split_slice, spaceid_to_slice, split_chunks


def test_split_slice():
    chunk = 10
    for start in range(20):
        for stop in range(30):
            for step in range(1, 10):
                s = Slice(start, stop, step)
                slices = list(split_slice(s, chunk))
                base = list(range(100)[s.raw])
                assert sum([len(s_) for i, s_ in slices]) ==\
                    len(s), (s, slices)
                pieces = [list(range(i*chunk, (i+1)*chunk)[s_.raw]) for i, s_ in
                          slices]
                extended = []
                for p in pieces:
                    extended.extend(p)
                assert base == extended, (s, slices)


def test_spaceid_to_slice(h5file):
    shape = 10
    a = h5file.create_dataset('a', data=np.arange(shape))

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


def test_split_chunks():
    shape = (100, 101, 102)
    chunks = (50, 50, 20)
    # split_chunks() actually returns ndindex types, but we use raw types here
    # since they are more terse.
    assert list(split_chunks(shape=shape, chunks=chunks)) ==\
        [(slice(0, 50, 1), slice(0, 50, 1), slice(0, 20, 1)),
         (slice(0, 50, 1), slice(0, 50, 1), slice(20, 40, 1)),
         (slice(0, 50, 1), slice(0, 50, 1), slice(40, 60, 1)),
         (slice(0, 50, 1), slice(0, 50, 1), slice(60, 80, 1)),
         (slice(0, 50, 1), slice(0, 50, 1), slice(80, 100, 1)),
         (slice(0, 50, 1), slice(0, 50, 1), slice(100, 102, 1)),
         (slice(0, 50, 1), slice(50, 100, 1), slice(0, 20, 1)),
         (slice(0, 50, 1), slice(50, 100, 1), slice(20, 40, 1)),
         (slice(0, 50, 1), slice(50, 100, 1), slice(40, 60, 1)),
         (slice(0, 50, 1), slice(50, 100, 1), slice(60, 80, 1)),
         (slice(0, 50, 1), slice(50, 100, 1), slice(80, 100, 1)),
         (slice(0, 50, 1), slice(50, 100, 1), slice(100, 102, 1)),
         (slice(0, 50, 1), slice(100, 101, 1), slice(0, 20, 1)),
         (slice(0, 50, 1), slice(100, 101, 1), slice(20, 40, 1)),
         (slice(0, 50, 1), slice(100, 101, 1), slice(40, 60, 1)),
         (slice(0, 50, 1), slice(100, 101, 1), slice(60, 80, 1)),
         (slice(0, 50, 1), slice(100, 101, 1), slice(80, 100, 1)),
         (slice(0, 50, 1), slice(100, 101, 1), slice(100, 102, 1)),
         (slice(50, 100, 1), slice(0, 50, 1), slice(0, 20, 1)),
         (slice(50, 100, 1), slice(0, 50, 1), slice(20, 40, 1)),
         (slice(50, 100, 1), slice(0, 50, 1), slice(40, 60, 1)),
         (slice(50, 100, 1), slice(0, 50, 1), slice(60, 80, 1)),
         (slice(50, 100, 1), slice(0, 50, 1), slice(80, 100, 1)),
         (slice(50, 100, 1), slice(0, 50, 1), slice(100, 102, 1)),
         (slice(50, 100, 1), slice(50, 100, 1), slice(0, 20, 1)),
         (slice(50, 100, 1), slice(50, 100, 1), slice(20, 40, 1)),
         (slice(50, 100, 1), slice(50, 100, 1), slice(40, 60, 1)),
         (slice(50, 100, 1), slice(50, 100, 1), slice(60, 80, 1)),
         (slice(50, 100, 1), slice(50, 100, 1), slice(80, 100, 1)),
         (slice(50, 100, 1), slice(50, 100, 1), slice(100, 102, 1)),
         (slice(50, 100, 1), slice(100, 101, 1), slice(0, 20, 1)),
         (slice(50, 100, 1), slice(100, 101, 1), slice(20, 40, 1)),
         (slice(50, 100, 1), slice(100, 101, 1), slice(40, 60, 1)),
         (slice(50, 100, 1), slice(100, 101, 1), slice(60, 80, 1)),
         (slice(50, 100, 1), slice(100, 101, 1), slice(80, 100, 1)),
         (slice(50, 100, 1), slice(100, 101, 1), slice(100, 102, 1))]
