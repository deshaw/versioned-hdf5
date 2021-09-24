import numpy as np
from numpy.testing import assert_equal

from h5py._hl.selections import Selection

from ..slicetools import spaceid_to_slice

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
                    except (ValueError, OSError):
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
