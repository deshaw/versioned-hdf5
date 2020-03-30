from pytest import raises

import numpy as np
from numpy.testing import assert_equal

from .test_backend import setup

from ..backend import DEFAULT_CHUNK_SIZE
from ..versions import (create_version, get_nth_previous_version,
                        set_current_version, all_versions)

def test_create_version():
    with setup() as f:
        chunk_size = 2**10
        data = np.concatenate((np.ones((2*chunk_size,)),
                               2*np.ones((chunk_size,)),
                               3*np.ones((chunk_size,))))

        version1 = create_version(f, 'version1', {'test_data': data}, '',
                                  chunk_size={'test_data': chunk_size},
                                  compression={'test_data': 'gzip'},
                                  compression_opts={'test_data': 3})
        raises(ValueError, lambda: create_version(f, 'version_bad',
                                  {'test_data': data},
                                  chunk_size={'test_data': 2**9}))
        raises(ValueError, lambda: create_version(f, 'version_bad',
                                  {'test_data': data},
                                  compression={'test_data': 'lzf'}))
        raises(ValueError, lambda: create_version(f, 'version_bad',
                                  {'test_data': data},
                                  compression_opts={'test_data': 4}))
        assert version1.attrs['prev_version'] == '__first_version__'
        assert version1.parent.attrs['current_version'] == 'version1'
        assert_equal(version1['test_data'], data)

        ds = f['/_version_data/test_data/raw_data']

        assert ds.shape == (3*chunk_size,)
        assert_equal(ds[0:1*chunk_size], 1.0)
        assert_equal(ds[1*chunk_size:2*chunk_size], 2.0)
        assert_equal(ds[2*chunk_size:3*chunk_size], 3.0)
        assert ds.compression == 'gzip'
        assert ds.compression_opts == 3

        data[0] = 0.0
        version2 = create_version(f, 'version2', {'test_data':
        data}, 'version1', make_current=False)
        assert version2.attrs['prev_version'] == 'version1'
        assert_equal(version2['test_data'], data)
        assert version2.parent.attrs['current_version'] == 'version1'

        assert ds.shape == (4*chunk_size,)
        assert_equal(ds[0:1*chunk_size], 1.0)
        assert_equal(ds[1*chunk_size:2*chunk_size], 2.0)
        assert_equal(ds[2*chunk_size:3*chunk_size], 3.0)
        assert_equal(ds[3*chunk_size], 0.0)
        assert_equal(ds[3*chunk_size+1:4*chunk_size], 1.0)
        assert ds.compression == 'gzip'
        assert ds.compression_opts == 3

        assert set(all_versions(f)) == {'version1', 'version2'}
        assert set(all_versions(f, include_first=True)) == {'version1',
                                                            'version2',
                                                            '__first_version__'}

def test_create_version_chunks():
    with setup() as f:
        chunk_size = 2**10
        data = np.concatenate((np.ones((2*chunk_size,)),
                               2*np.ones((chunk_size,)),
                               3*np.ones((chunk_size,))))
        # TODO: Support creating the initial version with chunks
        version1 = create_version(f, 'version1', {'test_data': data},
                                  chunk_size={'test_data': chunk_size},
                                  compression={'test_data': 'gzip'},
                                  compression_opts={'test_data': 3})
        raises(ValueError, lambda: create_version(f, 'version_bad',
                                                  {'test_data': data},
                                                  chunk_size={'test_data':2**9}))
        raises(ValueError, lambda: create_version(f, 'version_bad',
                                                  {'test_data': data},
                                                  compression={'test_data':'lzf'}))
        raises(ValueError, lambda: create_version(f, 'version_bad',
                                                  {'test_data': data},
                                                  compression_opts={'test_data':4}))
        assert_equal(version1['test_data'], data)

        ds = f['/_version_data/test_data/raw_data']

        assert ds.shape == (3*chunk_size,)
        assert_equal(ds[0:1*chunk_size], 1.0)
        assert_equal(ds[1*chunk_size:2*chunk_size], 2.0)
        assert_equal(ds[2*chunk_size:3*chunk_size], 3.0)
        assert ds.compression == 'gzip'
        assert ds.compression_opts == 3

        data2_chunks = {0: np.ones((chunk_size,)),
                        1: np.ones((chunk_size,)),
                        2: 2*np.ones((chunk_size,)),
                        3: 3*np.ones((chunk_size,)),
        }
        data2_chunks[0][0] = 0.0
        data[0] = 0.0

        version2 = create_version(f, 'version2', {'test_data':  data2_chunks})
        assert_equal(version2['test_data'], data)

        assert ds.shape == (4*chunk_size,)
        assert_equal(ds[0:1*chunk_size], 1.0)
        assert_equal(ds[1*chunk_size:2*chunk_size], 2.0)
        assert_equal(ds[2*chunk_size:3*chunk_size], 3.0)
        assert_equal(ds[3*chunk_size], 0.0)
        assert_equal(ds[3*chunk_size+1:4*chunk_size], 1.0)
        assert ds.compression == 'gzip'
        assert ds.compression_opts == 3


        data3_chunks = {0: np.ones((chunk_size,)),
                        1: slice(0*chunk_size, 1*chunk_size),
                        2: slice(1*chunk_size, 2*chunk_size),
                        3: slice(2*chunk_size, 3*chunk_size),
        }
        data3_chunks[0][0] = 2.0
        data[0] = 2.0

        version3 = create_version(f, 'version3', {'test_data':  data3_chunks})
        assert_equal(version3['test_data'], data)

        assert ds.shape == (5*chunk_size,)
        assert_equal(ds[0:1*chunk_size], 1.0)
        assert_equal(ds[1*chunk_size:2*chunk_size], 2.0)
        assert_equal(ds[2*chunk_size:3*chunk_size], 3.0)
        assert_equal(ds[3*chunk_size], 0.0)
        assert_equal(ds[3*chunk_size+1:4*chunk_size], 1.0)
        assert_equal(ds[4*chunk_size], 2.0)
        assert_equal(ds[4*chunk_size+1:5*chunk_size], 1.0)

        assert set(all_versions(f)) == {'version1', 'version2', 'version3'}

def test_get_nth_prev_version():
    with setup() as f:
        data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                               2*np.ones((DEFAULT_CHUNK_SIZE,)),
                               3*np.ones((DEFAULT_CHUNK_SIZE,))))

        create_version(f, 'version1', {'test_data': data})

        data[0] = 2.0
        create_version(f, 'version2', {'test_data': data})

        data[0] = 3.0
        create_version(f, 'version3', {'test_data': data})

        data[1] = 2.0
        create_version(f, 'version2_1', {'test_data': data}, 'version1')

        assert get_nth_previous_version(f, 'version1', 0) == 'version1'

        with raises(IndexError):
            get_nth_previous_version(f, 'version1', 1)

        assert get_nth_previous_version(f, 'version2', 0) == 'version2'
        assert get_nth_previous_version(f, 'version2', 1) == 'version1'
        with raises(IndexError):
            get_nth_previous_version(f, 'version2', 2)

        assert get_nth_previous_version(f, 'version3', 0) == 'version3'
        assert get_nth_previous_version(f, 'version3', 1) == 'version2'
        assert get_nth_previous_version(f, 'version3', 2) == 'version1'
        with raises(IndexError):
            get_nth_previous_version(f, 'version3', 3)

        assert get_nth_previous_version(f, 'version2_1', 0) == 'version2_1'
        assert get_nth_previous_version(f, 'version2_1', 1) == 'version1'
        with raises(IndexError):
            get_nth_previous_version(f, 'version2_1', 2)

def test_set_current_version():
    with setup() as f:
        data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                               2*np.ones((DEFAULT_CHUNK_SIZE,)),
                               3*np.ones((DEFAULT_CHUNK_SIZE,))))

        create_version(f, 'version1', {'test_data': data})
        versions = f['_version_data/versions']
        assert versions.attrs['current_version'] == 'version1'

        data[0] = 2.0
        create_version(f, 'version2', {'test_data': data})
        assert versions.attrs['current_version'] == 'version2'

        set_current_version(f, 'version1')
        assert versions.attrs['current_version'] == 'version1'

        with raises(ValueError):
            set_current_version(f, 'version3')
