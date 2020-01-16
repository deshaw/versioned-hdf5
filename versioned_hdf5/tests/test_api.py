from pytest import raises

import numpy as np
from numpy.testing import assert_equal

from ..backend import CHUNK_SIZE
from ..api import VersionedHDF5File

from .test_backend import setup

def test_stage_version():
    with setup() as f:
        file = VersionedHDF5File(f)

        test_data = np.concatenate((np.ones((2*CHUNK_SIZE,)),
                                    2*np.ones((CHUNK_SIZE,)),
                                    3*np.ones((CHUNK_SIZE,))))


        with file.stage_version('version1', '') as group:
            group['test_data'] = test_data

        version1 = file['version1']
        assert version1.attrs['prev_version'] == '__first_version__'
        assert_equal(version1['test_data'], test_data)

        ds = f['/_version_data/test_data/raw_data']

        assert ds.shape == (3*CHUNK_SIZE,)
        assert_equal(ds[0:1*CHUNK_SIZE], 1.0)
        assert_equal(ds[1*CHUNK_SIZE:2*CHUNK_SIZE], 2.0)
        assert_equal(ds[2*CHUNK_SIZE:3*CHUNK_SIZE], 3.0)

        with file.stage_version('version2', 'version1') as group:
            group['test_data'][0] = 0.0

        version2 = file['version2']
        assert version2.attrs['prev_version'] == 'version1'
        test_data[0] = 0.0
        assert_equal(version2['test_data'], test_data)

        assert ds.shape == (4*CHUNK_SIZE,)
        assert_equal(ds[0:1*CHUNK_SIZE], 1.0)
        assert_equal(ds[1*CHUNK_SIZE:2*CHUNK_SIZE], 2.0)
        assert_equal(ds[2*CHUNK_SIZE:3*CHUNK_SIZE], 3.0)
        assert_equal(ds[3*CHUNK_SIZE], 0.0)
        assert_equal(ds[3*CHUNK_SIZE+1:4*CHUNK_SIZE], 1.0)

def test_version_int_slicing():
    with setup() as f:
        file = VersionedHDF5File(f)
        test_data = np.concatenate((np.ones((2*CHUNK_SIZE,)),
                               2*np.ones((CHUNK_SIZE,)),
                               3*np.ones((CHUNK_SIZE,))))

        with file.stage_version('version1', '') as group:
            group['test_data'] = test_data

        with file.stage_version('version2', 'version1') as group:
            group['test_data'][0] = 2.0

        with file.stage_version('version3', 'version2') as group:
            group['test_data'][0] = 3.0

        with file.stage_version('version2_1', 'version1', make_current=False) as group:
            group['test_data'][0] = 2.0


        assert file[0]['test_data'][0] == 3.0

        with raises(ValueError):
            file[1]

        assert file[-1]['test_data'][0] == 2.0
        assert file[-2]['test_data'][0] == 1.0, file[-2]
        with raises(ValueError):
            file[-3]

        file.current_version = 'version2'

        assert file[0]['test_data'][0] == 2.0
        assert file[-1]['test_data'][0] == 1.0
        with raises(ValueError):
            file[-2]

        file.current_version = 'version2_1'

        assert file[0]['test_data'][0] == 2.0
        assert file[-1]['test_data'][0] == 1.0
        with raises(ValueError):
            file[-2]

        file.current_version = 'version1'

        assert file[0]['test_data'][0] == 1.0
        with raises(ValueError):
            file[-1]

def test_version_name_slicing():
    with setup() as f:
        file = VersionedHDF5File(f)
        test_data = np.concatenate((np.ones((2*CHUNK_SIZE,)),
                               2*np.ones((CHUNK_SIZE,)),
                               3*np.ones((CHUNK_SIZE,))))

        with file.stage_version('version1', '') as group:
            group['test_data'] = test_data

        with file.stage_version('version2', 'version1') as group:
            group['test_data'][0] = 2.0

        with file.stage_version('version3', 'version2') as group:
            group['test_data'][0] = 3.0

        with file.stage_version('version2_1', 'version1', make_current=False) as group:
            group['test_data'][0] = 2.0


        assert file[0]['test_data'][0] == 3.0

        with raises(ValueError):
            file[1]

        assert file[-1]['test_data'][0] == 2.0
        assert file[-2]['test_data'][0] == 1.0, file[-2]
