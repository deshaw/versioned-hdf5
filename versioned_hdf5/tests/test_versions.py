import numpy as np
from numpy.testing import assert_equal

from .test_backend import setup

from ..backend import CHUNK_SIZE
from ..versions import create_version

def test_create_version():
    with setup() as f:
        data = np.concatenate((np.ones((2*CHUNK_SIZE,)),
                               2*np.ones((CHUNK_SIZE,)),
                               3*np.ones((CHUNK_SIZE,))))

        version1 = create_version(f, 'version1', '', {'test_data': data})
        assert version1.attrs['prev_version'] == ''
        assert_equal(version1['test_data'], data)

        ds = f['/_version_data/test_data/raw_data']

        assert ds.shape == (3*CHUNK_SIZE,)
        assert_equal(ds[0:1*CHUNK_SIZE], 1.0)
        assert_equal(ds[1*CHUNK_SIZE:2*CHUNK_SIZE], 2.0)
        assert_equal(ds[2*CHUNK_SIZE:3*CHUNK_SIZE], 3.0)


        data[0] = 0.0
        version2 = create_version(f, 'version2', 'version1', {'test_data': data})
        assert version2.attrs['prev_version'] == 'version1'
        assert_equal(version2['test_data'], data)

        assert ds.shape == (4*CHUNK_SIZE,)
        assert_equal(ds[0:1*CHUNK_SIZE], 1.0)
        assert_equal(ds[1*CHUNK_SIZE:2*CHUNK_SIZE], 2.0)
        assert_equal(ds[2*CHUNK_SIZE:3*CHUNK_SIZE], 3.0)
        assert_equal(ds[3*CHUNK_SIZE], 0.0)
        assert_equal(ds[3*CHUNK_SIZE+1:4*CHUNK_SIZE], 1.0)
