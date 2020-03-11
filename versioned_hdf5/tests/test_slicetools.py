from __future__ import print_function, division

from ..slicetools import split_slice, slice_size

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
                pieces = [list(range(i*chunk, (i+1)*chunk)[s_]) for i, s_ in
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
