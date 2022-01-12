import numpy as np

from versioned_hdf5.replay import (modify_metadata, delete_version,
                                   delete_versions, _recreate_raw_data,
                                   _recreate_hashtable,
                                   _recreate_virtual_dataset)
from versioned_hdf5.hashtable import Hashtable

def setup_vfile(file):
    with file.stage_version('version1') as g:
        data = g.create_dataset('test_data', data=None, fillvalue=1., shape=(10000,), chunks=(1000,))
        data[0] = 0.
        g.create_dataset('test_data2', data=[1, 2, 3], chunks=(1000,))

    with file.stage_version('version2') as g:
        g['test_data'][2000] = 2.
        g.create_dataset('test_data3', data=[1, 2, 3, 4], chunks=(1000,))

def check_data(file, test_data_fillvalue=1., version2=True):
    assert file['version1']['test_data'].shape == (10000,)
    assert file['version1']['test_data'][0] == 0.
    assert np.all(file['version1']['test_data'][1:] == test_data_fillvalue)

    if version2:
        assert file['version2']['test_data'].shape == (10000,)
        assert file['version2']['test_data'][0] == 0.
        assert np.all(file['version2']['test_data'][1:2000] == test_data_fillvalue)
        assert file['version2']['test_data'][2000] == 2.
        assert np.all(file['version2']['test_data'][2001:] == test_data_fillvalue)

    assert file['version1']['test_data2'].shape == (3,)
    assert np.all(file['version1']['test_data2'][:] == [1, 2, 3])

    if version2:
        assert file['version2']['test_data2'].shape == (3,)
        assert np.all(file['version2']['test_data2'][:] == [1, 2, 3])

    assert 'test_data3' not in file['version1']

    if version2:
        assert file['version2']['test_data3'].shape == (4,)
        assert np.all(file['version2']['test_data3'][:] == [1, 2, 3, 4])

def test_modify_metadata_compression(vfile):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile['version1']['test_data'].compression == None
    assert vfile['version2']['test_data'].compression == None
    assert vfile['version1']['test_data'].compression_opts == None
    assert vfile['version2']['test_data'].compression_opts == None

    assert vfile['version1']['test_data2'].compression == None
    assert vfile['version2']['test_data2'].compression == None
    assert vfile['version1']['test_data2'].compression_opts == None
    assert vfile['version2']['test_data2'].compression_opts == None

    assert vfile['version2']['test_data3'].compression == None
    assert vfile['version2']['test_data3'].compression_opts == None

    assert f['_version_data']['test_data']['raw_data'].compression == None
    assert f['_version_data']['test_data2']['raw_data'].compression == None
    assert f['_version_data']['test_data3']['raw_data'].compression == None

    assert f['_version_data']['test_data']['raw_data'].compression_opts == None
    assert f['_version_data']['test_data2']['raw_data'].compression_opts == None
    assert f['_version_data']['test_data3']['raw_data'].compression_opts == None

    modify_metadata(f, 'test_data2', compression='gzip', compression_opts=3)
    check_data(vfile)

    assert vfile['version1']['test_data'].compression == None
    assert vfile['version2']['test_data'].compression == None
    assert vfile['version1']['test_data'].compression_opts == None
    assert vfile['version2']['test_data'].compression_opts == None

    assert vfile['version1']['test_data2'].compression == 'gzip'
    assert vfile['version2']['test_data2'].compression == 'gzip'
    assert vfile['version1']['test_data2'].compression_opts == 3
    assert vfile['version2']['test_data2'].compression_opts == 3

    assert vfile['version2']['test_data3'].compression == None
    assert vfile['version2']['test_data3'].compression_opts == None

    assert f['_version_data']['test_data']['raw_data'].compression == None
    assert f['_version_data']['test_data2']['raw_data'].compression == 'gzip'
    assert f['_version_data']['test_data3']['raw_data'].compression == None

    assert f['_version_data']['test_data']['raw_data'].compression_opts == None
    assert f['_version_data']['test_data2']['raw_data'].compression_opts == 3
    assert f['_version_data']['test_data3']['raw_data'].compression_opts == None

    # Make sure the tmp group group has been destroyed.
    assert list(f['_version_data']) == ['test_data', 'test_data2', 'test_data3', 'versions']

def test_modify_metadata_chunks(vfile):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile['version1']['test_data'].chunks == (1000,)
    assert vfile['version2']['test_data'].chunks == (1000,)

    assert vfile['version1']['test_data2'].chunks == (1000,)
    assert vfile['version2']['test_data2'].chunks == (1000,)

    assert vfile['version2']['test_data3'].chunks == (1000,)

    assert f['_version_data']['test_data']['raw_data'].chunks == (1000,)
    assert f['_version_data']['test_data2']['raw_data'].chunks == (1000,)
    assert f['_version_data']['test_data3']['raw_data'].chunks == (1000,)

    modify_metadata(f, 'test_data2', chunks=(500,))
    check_data(vfile)

    assert vfile['version1']['test_data'].chunks == (1000,)
    assert vfile['version2']['test_data'].chunks == (1000,)

    assert vfile['version1']['test_data2'].chunks == (500,)
    assert vfile['version2']['test_data2'].chunks == (500,)

    assert vfile['version2']['test_data3'].chunks == (1000,)

    assert f['_version_data']['test_data']['raw_data'].chunks == (1000,)
    assert f['_version_data']['test_data2']['raw_data'].chunks == (500,)
    assert f['_version_data']['test_data3']['raw_data'].chunks == (1000,)

    # Make sure the tmp group group has been destroyed.
    assert list(f['_version_data']) == ['test_data', 'test_data2', 'test_data3', 'versions']

def test_modify_metadata_dtype(vfile):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile['version1']['test_data'].dtype == np.float64
    assert vfile['version2']['test_data'].dtype == np.float64

    assert vfile['version1']['test_data2'].dtype == np.int64
    assert vfile['version2']['test_data2'].dtype == np.int64

    assert vfile['version2']['test_data3'].dtype == np.int64

    assert f['_version_data']['test_data']['raw_data'].dtype == np.float64
    assert f['_version_data']['test_data2']['raw_data'].dtype == np.int64
    assert f['_version_data']['test_data3']['raw_data'].dtype == np.int64

    modify_metadata(f, 'test_data2', dtype=np.float64)
    check_data(vfile)


    assert vfile['version1']['test_data'].dtype == np.float64
    assert vfile['version2']['test_data'].dtype == np.float64

    assert vfile['version1']['test_data2'].dtype == np.float64
    assert vfile['version2']['test_data2'].dtype == np.float64

    assert vfile['version2']['test_data3'].dtype == np.int64

    assert f['_version_data']['test_data']['raw_data'].dtype == np.float64
    assert f['_version_data']['test_data2']['raw_data'].dtype == np.float64
    assert f['_version_data']['test_data3']['raw_data'].dtype == np.int64

    # Make sure the tmp group group has been destroyed.
    assert list(f['_version_data']) == ['test_data', 'test_data2', 'test_data3', 'versions']

def test_modify_metadata_fillvalue1(vfile):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile['version1']['test_data'].fillvalue == 1.
    assert vfile['version2']['test_data'].fillvalue == 1.

    assert vfile['version1']['test_data2'].fillvalue == 0
    assert vfile['version2']['test_data2'].fillvalue == 0

    assert vfile['version2']['test_data3'].fillvalue == 0

    assert f['_version_data']['test_data']['raw_data'].fillvalue == 1.
    assert f['_version_data']['test_data2']['raw_data'].fillvalue == 0
    assert f['_version_data']['test_data3']['raw_data'].fillvalue == 0

    modify_metadata(f, 'test_data', fillvalue=3.)
    check_data(vfile, test_data_fillvalue=3.)

    assert vfile['version1']['test_data'].fillvalue == 3.
    assert vfile['version2']['test_data'].fillvalue == 3.

    assert vfile['version1']['test_data2'].fillvalue == 0
    assert vfile['version2']['test_data2'].fillvalue == 0

    assert vfile['version2']['test_data3'].fillvalue == 0

    assert f['_version_data']['test_data']['raw_data'].fillvalue == 3.
    assert f['_version_data']['test_data2']['raw_data'].fillvalue == 0
    assert f['_version_data']['test_data3']['raw_data'].fillvalue == 0

    # Make sure the tmp group group has been destroyed.
    assert list(f['_version_data']) == ['test_data', 'test_data2', 'test_data3', 'versions']

def test_modify_metadata_fillvalue2(vfile):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile['version1']['test_data'].fillvalue == 1.
    assert vfile['version2']['test_data'].fillvalue == 1.

    assert vfile['version1']['test_data2'].fillvalue == 0
    assert vfile['version2']['test_data2'].fillvalue == 0

    assert vfile['version2']['test_data3'].fillvalue == 0

    assert f['_version_data']['test_data']['raw_data'].fillvalue == 1.
    assert f['_version_data']['test_data2']['raw_data'].fillvalue == 0
    assert f['_version_data']['test_data3']['raw_data'].fillvalue == 0

    modify_metadata(f, 'test_data2', fillvalue=3)
    check_data(vfile)

    assert vfile['version1']['test_data'].fillvalue == 1.
    assert vfile['version2']['test_data'].fillvalue == 1.

    assert vfile['version1']['test_data2'].fillvalue == 3
    assert vfile['version2']['test_data2'].fillvalue == 3

    assert vfile['version2']['test_data3'].fillvalue == 0

    assert f['_version_data']['test_data']['raw_data'].fillvalue == 1.
    assert f['_version_data']['test_data2']['raw_data'].fillvalue == 3
    assert f['_version_data']['test_data3']['raw_data'].fillvalue == 0

    # Make sure the tmp group group has been destroyed.
    assert list(f['_version_data']) == ['test_data', 'test_data2', 'test_data3', 'versions']

def test_delete_version(vfile):
    setup_vfile(vfile)
    f = vfile.f

    delete_version(f, 'version2')
    check_data(vfile, version2=False)
    assert list(vfile) == ['version1']
    assert list(f['_version_data']) == ['test_data', 'test_data2', 'versions']
    assert not np.isin(2., f['_version_data']['test_data']['raw_data'][:])

def test_delete_versions(vfile):
    setup_vfile(vfile)
    with vfile.stage_version('version3') as g:
        g['test_data'][2000] = 3.
        g.create_dataset('test_data4', data=[1, 2, 3, 4], chunks=(1000,))
    f = vfile.f

    delete_versions(f, ['version2', 'version3'])
    check_data(vfile, version2=False)
    assert list(vfile) == ['version1']
    assert list(f['_version_data']) == ['test_data', 'test_data2', 'versions']
    assert not np.isin(2., f['_version_data']['test_data']['raw_data'][:])

def setup2(vfile):
    with vfile.stage_version('version1') as g:
        g.create_dataset('test_data',
                         data=np.arange(20000).reshape((1000, 20)),
                         chunks=(101,11))

    with vfile.stage_version('version2') as g:
        g['test_data'][::200] = -g['test_data'][::200]

def test_recreate_raw_data(vfile):
    setup2(vfile)

    chunks_map = _recreate_raw_data(vfile.f, 'test_data', ['version1'], tmp=True)

    assert len(chunks_map) == 20

    raw_data = vfile.f['_version_data/test_data/raw_data']
    tmp_raw_data = vfile.f['_version_data/test_data/_tmp_raw_data']

    assert raw_data.shape == (3030, 11)
    assert tmp_raw_data.shape == (2020, 11)
    for old, new in chunks_map.items():
        a = raw_data[old.raw]
        b = tmp_raw_data[new.raw]
        assert a.shape == b.shape
        np.testing.assert_equal(a, b)

def test_recreate_hashtable(vfile):
    setup2(vfile)
    chunks_map = _recreate_raw_data(vfile.f, 'test_data', ['version1'], tmp=False)

    # Recreate a separate, independent version, with the dataset as it would
    # be with version1 deleted.
    with vfile.stage_version('version2_2', prev_version='') as g:
        g.create_dataset('test_data2',
                         data=np.arange(20000).reshape((1000, 20)),
                         chunks=(101,11))
        g['test_data2'][::200] = -g['test_data2'][::200]

    # orig_hashtable = Hashtable(vfile.f, 'test_data')

    _recreate_hashtable(vfile.f, 'test_data', chunks_map, tmp=True)

    new_hash_table = Hashtable(vfile.f, 'test_data',
                               hash_table_name='_tmp_hash_table')

    new_hash_table2 = Hashtable(vfile.f, 'test_data2')
    d1 = dict(new_hash_table)
    d2 = dict(new_hash_table2)
    assert d1.keys() == d2.keys()

    # The exact slices won't be the same because raw data won't be in the same
    # order
    for h in d1:
        np.testing.assert_equal(
            vfile.f['_version_data/test_data/raw_data'][d1[h].raw],
            vfile.f['_version_data/test_data2/raw_data'][d2[h].raw],
        )

def test_recreate_virtual_dataset(vfile):
    setup2(vfile)
    orig_virtual_dataset = vfile.f['_version_data/versions/version2/test_data'][:]

    chunks_map = _recreate_raw_data(vfile.f, 'test_data', ['version1'], tmp=False)

    _recreate_hashtable(vfile.f, 'test_data', chunks_map, tmp=False)

    _recreate_virtual_dataset(vfile.f, 'test_data', ['version2'], chunks_map, tmp=True)

    new_virtual_dataset = vfile.f['_version_data/versions/version2/_tmp_test_data'][:]

    np.testing.assert_equal(orig_virtual_dataset, new_virtual_dataset)

def test_delete_versions2(vfile):
    setup2(vfile)
    data = np.arange(20000).reshape((1000, 20))
    data[::200] = -data[::200]

    assert vfile['version2']['test_data'].shape == data.shape

    delete_versions(vfile, ['version1'])

    assert list(vfile) == ['version2']

    assert list(vfile['version2']) == ['test_data']


    assert vfile['version2']['test_data'].shape == data.shape
    np.testing.assert_equal(vfile['version2']['test_data'][:], data)

    assert set(vfile.f['_version_data/test_data/raw_data'][:].flat) == set(data.flat)
