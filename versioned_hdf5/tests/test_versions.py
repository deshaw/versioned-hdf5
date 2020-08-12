from pytest import raises

import numpy as np
from numpy.testing import assert_equal

from ndindex import Tuple, Slice

from ..backend import DEFAULT_CHUNK_SIZE
from ..versions import (create_version_group, commit_version,
                        get_nth_previous_version, set_current_version,
                        all_versions, delete_version)


def test_create_version(h5file):
    chunk_size = 2**10
    chunks = (chunk_size,)
    data = np.concatenate((np.ones((2*chunk_size,)),
                           2*np.ones(chunks),
                           3*np.ones(chunks)))

    version1 = create_version_group(h5file, 'version1', '')
    commit_version(version1, {'test_data': data},
                   chunks={'test_data': chunks},
                   compression={'test_data': 'gzip'},
                   compression_opts={'test_data': 3})

    version_bad = create_version_group(h5file, 'version_bad', '')
    raises(ValueError, lambda: commit_version(version_bad, {'test_data': data},
                                              chunks={'test_data': (2**9,)}))
    delete_version(h5file, 'version_bad', 'version1')

    version_bad = create_version_group(h5file, 'version_bad', '')
    raises(ValueError, lambda: commit_version(version_bad, {'test_data': data},
                                              compression={'test_data': 'lzf'}))
    delete_version(h5file, 'version_bad', 'version1')

    version_bad = create_version_group(h5file, 'version_bad', '')
    raises(ValueError, lambda: commit_version(version_bad,
                                              {'test_data': data},
                                              compression_opts={'test_data': 4}))
    delete_version(h5file, 'version_bad', 'version1')

    assert version1.attrs['prev_version'] == '__first_version__'
    assert version1.parent.attrs['current_version'] == 'version1'
    # Test against the file here, not version1, since version1 is the
    # InMemoryGroup returned from create_version_group, but we did not add
    # the datasets to it directly.
    assert_equal(h5file['_version_data/versions/version1/test_data'], data)

    ds = h5file['/_version_data/test_data/raw_data']

    assert ds.shape == (3*chunk_size,)
    assert_equal(ds[0:1*chunk_size], 1.0)
    assert_equal(ds[1*chunk_size:2*chunk_size], 2.0)
    assert_equal(ds[2*chunk_size:3*chunk_size], 3.0)
    assert ds.compression == 'gzip'
    assert ds.compression_opts == 3

    data[0] = 0.0
    version2 = create_version_group(h5file, 'version2', 'version1')

    raises(ValueError,
           lambda: commit_version(version1, {'test_data': data},
                                  make_current=False))
    commit_version(version2, {'test_data': data},
                   make_current=False)
    assert version2.attrs['prev_version'] == 'version1'
    assert_equal(h5file['_version_data/versions/version2/test_data'], data)
    assert version2.parent.attrs['current_version'] == 'version1'

    assert ds.shape == (4*chunk_size,)
    assert_equal(ds[0:1*chunk_size], 1.0)
    assert_equal(ds[1*chunk_size:2*chunk_size], 2.0)
    assert_equal(ds[2*chunk_size:3*chunk_size], 3.0)
    assert_equal(ds[3*chunk_size], 0.0)
    assert_equal(ds[3*chunk_size+1:4*chunk_size], 1.0)
    assert ds.compression == 'gzip'
    assert ds.compression_opts == 3

    assert set(all_versions(h5file)) == {'version1', 'version2'}
    assert set(all_versions(h5file, include_first=True)) == {'version1',
                                                        'version2',
                                                        '__first_version__'}


def test_create_version_chunks(h5file):
    chunk_size = 2**10
    chunks = (chunk_size,)
    data = np.concatenate((np.ones((2*chunk_size,)),
                           2*np.ones(chunks),
                           3*np.ones(chunks)))
    # TODO: Support creating the initial version with chunks
    version1 = create_version_group(h5file, 'version1')
    commit_version(version1, {'test_data': data},
                   chunks={'test_data': chunks},
                   compression={'test_data': 'gzip'},
                   compression_opts={'test_data': 3})
    version_bad = create_version_group(h5file, 'version_bad', '')
    raises(ValueError, lambda: commit_version(version_bad,
                                              {'test_data': data},
                                              chunks={'test_data': (2**9,)}))
    delete_version(h5file, 'version_bad', 'version1')

    version_bad = create_version_group(h5file, 'version_bad', '')
    raises(ValueError, lambda: commit_version(version_bad,
                                              {'test_data': data},
                                              compression={'test_data':'lzf'}))
    delete_version(h5file, 'version_bad', 'version1')

    version_bad = create_version_group(h5file, 'version_bad', '')
    raises(ValueError, lambda: commit_version(version_bad,
                                              {'test_data': data},
                                              compression_opts={'test_data':4}))
    delete_version(h5file, 'version_bad', 'version1')

    assert_equal(h5file['_version_data/versions/version1/test_data'], data)

    ds = h5file['/_version_data/test_data/raw_data']

    assert ds.shape == (3*chunk_size,)
    assert_equal(ds[0:1*chunk_size], 1.0)
    assert_equal(ds[1*chunk_size:2*chunk_size], 2.0)
    assert_equal(ds[2*chunk_size:3*chunk_size], 3.0)
    assert ds.compression == 'gzip'
    assert ds.compression_opts == 3

    data2_chunks = {
        Tuple(Slice(0*chunk_size, 1*chunk_size, 1)): np.ones(chunks),
        Tuple(Slice(1*chunk_size, 2*chunk_size, 1)): np.ones(chunks),
        Tuple(Slice(2*chunk_size, 3*chunk_size, 1)): 2*np.ones(chunks),
        Tuple(Slice(3*chunk_size, 4*chunk_size, 1)): 3*np.ones(chunks),
    }
    data2_chunks[Tuple(Slice(0*chunk_size, 1*chunk_size, 1))][0] = 0.0
    data[0] = 0.0

    version2 = create_version_group(h5file, 'version2')
    commit_version(version2, {'test_data':  data2_chunks})
    assert_equal(h5file['_version_data/versions/version2/test_data'], data)

    assert ds.shape == (4*chunk_size,)
    assert_equal(ds[0:1*chunk_size], 1.0)
    assert_equal(ds[1*chunk_size:2*chunk_size], 2.0)
    assert_equal(ds[2*chunk_size:3*chunk_size], 3.0)
    assert_equal(ds[3*chunk_size], 0.0)
    assert_equal(ds[3*chunk_size+1:4*chunk_size], 1.0)
    assert ds.compression == 'gzip'
    assert ds.compression_opts == 3

    data3_chunks = {
        Tuple(Slice(0*chunk_size, 1*chunk_size, 1)): np.ones(chunks),
        Tuple(Slice(1*chunk_size, 2*chunk_size, 1)): Slice(0*chunk_size, 1*chunk_size),
        Tuple(Slice(2*chunk_size, 3*chunk_size, 1)): Slice(1*chunk_size, 2*chunk_size),
        Tuple(Slice(3*chunk_size, 4*chunk_size, 1)): Slice(2*chunk_size, 3*chunk_size),
    }
    data3_chunks[Tuple(Slice(0*chunk_size, 1*chunk_size, 1))][0] = 2.0
    data[0] = 2.0

    version3 = create_version_group(h5file, 'version3')
    commit_version(version3, {'test_data':  data3_chunks})
    assert_equal(h5file['_version_data/versions/version3/test_data'], data)

    assert ds.shape == (5*chunk_size,)
    assert_equal(ds[0:1*chunk_size], 1.0)
    assert_equal(ds[1*chunk_size:2*chunk_size], 2.0)
    assert_equal(ds[2*chunk_size:3*chunk_size], 3.0)
    assert_equal(ds[3*chunk_size], 0.0)
    assert_equal(ds[3*chunk_size+1:4*chunk_size], 1.0)
    assert_equal(ds[4*chunk_size], 2.0)
    assert_equal(ds[4*chunk_size+1:5*chunk_size], 1.0)

    assert set(all_versions(h5file)) == {'version1', 'version2', 'version3'}


def test_get_nth_prev_version(h5file):
    data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                           2*np.ones((DEFAULT_CHUNK_SIZE,)),
                           3*np.ones((DEFAULT_CHUNK_SIZE,))))

    version1 = create_version_group(h5file, 'version1')
    commit_version(version1, {'test_data': data})

    data[0] = 2.0
    version2 = create_version_group(h5file, 'version2')
    commit_version(version2, {'test_data': data})

    data[0] = 3.0
    version3 = create_version_group(h5file, 'version3')
    commit_version(version3, {'test_data': data})

    data[1] = 2.0
    version2_1 = create_version_group(h5file, 'version2_1', 'version1')
    commit_version(version2_1, {'test_data': data})

    assert get_nth_previous_version(h5file, 'version1', 0) == 'version1'

    with raises(IndexError):
        get_nth_previous_version(h5file, 'version1', 1)

    assert get_nth_previous_version(h5file, 'version2', 0) == 'version2'
    assert get_nth_previous_version(h5file, 'version2', 1) == 'version1'
    with raises(IndexError):
        get_nth_previous_version(h5file, 'version2', 2)

    assert get_nth_previous_version(h5file, 'version3', 0) == 'version3'
    assert get_nth_previous_version(h5file, 'version3', 1) == 'version2'
    assert get_nth_previous_version(h5file, 'version3', 2) == 'version1'
    with raises(IndexError):
        get_nth_previous_version(h5file, 'version3', 3)

    assert get_nth_previous_version(h5file, 'version2_1', 0) == 'version2_1'
    assert get_nth_previous_version(h5file, 'version2_1', 1) == 'version1'
    with raises(IndexError):
        get_nth_previous_version(h5file, 'version2_1', 2)


def test_set_current_version(h5file):
    data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                           2*np.ones((DEFAULT_CHUNK_SIZE,)),
                           3*np.ones((DEFAULT_CHUNK_SIZE,))))

    version1 = create_version_group(h5file, 'version1')
    commit_version(version1, {'test_data': data})
    versions = h5file['_version_data/versions']
    assert versions.attrs['current_version'] == 'version1'

    data[0] = 2.0
    version2 = create_version_group(h5file, 'version2')
    commit_version(version2, {'test_data': data})
    assert versions.attrs['current_version'] == 'version2'

    set_current_version(h5file, 'version1')
    assert versions.attrs['current_version'] == 'version1'

    with raises(ValueError):
        set_current_version(h5file, 'version3')


def test_delete_version(h5file):
    raises(ValueError, lambda: delete_version(h5file, 'version1'))

    data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                           2*np.ones((DEFAULT_CHUNK_SIZE,)),
                           3*np.ones((DEFAULT_CHUNK_SIZE,))))

    version1 = create_version_group(h5file, 'version1')
    commit_version(version1, {'test_data': data})
    versions = h5file['_version_data/versions']
    assert versions.attrs['current_version'] == 'version1'

    raises(ValueError, lambda: delete_version(h5file, 'version1', 'doesntexist'))

    delete_version(h5file, 'version1')
    versions = h5file['_version_data/versions']

    assert versions.attrs['current_version'] == '__first_version__'
    assert list(versions) == ['__first_version__']
