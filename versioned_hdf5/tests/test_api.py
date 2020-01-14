import numpy as np
from numpy.testing import assert_equal

from ..backend import CHUNK_SIZE
from ..api import Data

from .test_backend import setup

def test_api():
    with setup() as f:
        data = Data(f)

        test_data = np.concatenate((np.ones((2*CHUNK_SIZE,)),
                                    2*np.ones((CHUNK_SIZE,)),
                                    3*np.ones((CHUNK_SIZE,))))


        with data.version.stage_version('version1', '') as group:
            group['test_data'] = test_data

        version1 = data.version['version1']
        assert version1.attrs['prev_version'] == '__first_version__'
        assert_equal(version1['test_data'], test_data)

        ds = f['/_version_data/test_data/raw_data']

        assert ds.shape == (3*CHUNK_SIZE,)
        assert_equal(ds[0:1*CHUNK_SIZE], 1.0)
        assert_equal(ds[1*CHUNK_SIZE:2*CHUNK_SIZE], 2.0)
        assert_equal(ds[2*CHUNK_SIZE:3*CHUNK_SIZE], 3.0)

        with data.version.stage_version('version2', 'version1') as group:
            group['test_data'][0] = 0.0

        version2 = data.version['version2']
        assert version2.attrs['prev_version'] == 'version1'
        test_data[0] = 0.0
        assert_equal(version2['test_data'], test_data)

        assert ds.shape == (4*CHUNK_SIZE,)
        assert_equal(ds[0:1*CHUNK_SIZE], 1.0)
        assert_equal(ds[1*CHUNK_SIZE:2*CHUNK_SIZE], 2.0)
        assert_equal(ds[2*CHUNK_SIZE:3*CHUNK_SIZE], 3.0)
        assert_equal(ds[3*CHUNK_SIZE], 0.0)
        assert_equal(ds[3*CHUNK_SIZE+1:4*CHUNK_SIZE], 1.0)
