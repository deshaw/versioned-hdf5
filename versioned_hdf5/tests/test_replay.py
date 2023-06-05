import h5py
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
        group = g.create_group('group')
        group.create_dataset('test_data4', data=[1, 2, 3, 4], chunks=(1000,))

    with file.stage_version('version2') as g:
        g['test_data'][2000] = 2.
        g.create_dataset('test_data3', data=[1, 2, 3, 4], chunks=(1000,))
        g['group']['test_data4'][0] = 5

def check_data(file, test_data_fillvalue=1., version2=True, test_data4_fillvalue=0):
    assert set(file['version1']) == {'test_data', 'test_data2', 'group'}
    assert file['version1']['test_data'].shape == (10000,)
    assert file['version1']['test_data'][0] == 0.
    assert np.all(file['version1']['test_data'][1:] == test_data_fillvalue)

    if version2:
        assert set(file['version2']) == {'test_data', 'test_data2',
                                         'test_data3', 'group'}
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

    assert set(file['version1']['group']) == {'test_data4'}
    assert file['version1']['group']['test_data4'].shape == (4,)
    np.testing.assert_equal(file['version1']['group']['test_data4'][:4],
                            [1, 2, 3, 4])
    assert np.all(file['version1']['group']['test_data4'][4:] == test_data4_fillvalue)

    if version2:
        assert set(file['version2']['group']) == {'test_data4'}
        assert file['version2']['group']['test_data4'].shape == (4,)
        np.testing.assert_equal(file['version2']['group']['test_data4'][:4],
                                [5, 2, 3, 4])
        assert np.all(file['version2']['group']['test_data4'][4:] == test_data4_fillvalue)

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

    assert vfile['version1']['group']['test_data4'].compression == None
    assert vfile['version2']['group']['test_data4'].compression == None
    assert vfile['version1']['group']['test_data4'].compression_opts == None
    assert vfile['version2']['group']['test_data4'].compression_opts == None

    assert f['_version_data']['test_data']['raw_data'].compression == None
    assert f['_version_data']['test_data2']['raw_data'].compression == None
    assert f['_version_data']['test_data3']['raw_data'].compression == None
    assert f['_version_data']['group']['test_data4']['raw_data'].compression == None

    assert f['_version_data']['test_data']['raw_data'].compression_opts == None
    assert f['_version_data']['test_data2']['raw_data'].compression_opts == None
    assert f['_version_data']['test_data3']['raw_data'].compression_opts == None
    assert f['_version_data']['group']['test_data4']['raw_data'].compression_opts == None

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


    assert vfile['version1']['group']['test_data4'].compression == None
    assert vfile['version2']['group']['test_data4'].compression == None
    assert vfile['version1']['group']['test_data4'].compression_opts == None
    assert vfile['version2']['group']['test_data4'].compression_opts == None

    assert f['_version_data']['test_data']['raw_data'].compression == None
    assert f['_version_data']['test_data2']['raw_data'].compression == 'gzip'
    assert f['_version_data']['test_data3']['raw_data'].compression == None
    assert f['_version_data']['group']['test_data4']['raw_data'].compression == None

    assert f['_version_data']['test_data']['raw_data'].compression_opts == None
    assert f['_version_data']['test_data2']['raw_data'].compression_opts == 3
    assert f['_version_data']['test_data3']['raw_data'].compression_opts == None
    assert f['_version_data']['group']['test_data4']['raw_data'].compression_opts == None

    # Make sure the tmp group group has been destroyed.
    assert set(f['_version_data']) == {'test_data', 'test_data2',
                                        'test_data3', 'group', 'versions'}
    assert set(f['_version_data']['group']) == {'test_data4'}

def test_modify_metadata_compressio2(vfile):
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

    assert vfile['version1']['group']['test_data4'].compression == None
    assert vfile['version2']['group']['test_data4'].compression == None
    assert vfile['version1']['group']['test_data4'].compression_opts == None
    assert vfile['version2']['group']['test_data4'].compression_opts == None

    assert f['_version_data']['test_data']['raw_data'].compression == None
    assert f['_version_data']['test_data2']['raw_data'].compression == None
    assert f['_version_data']['test_data3']['raw_data'].compression == None
    assert f['_version_data']['group']['test_data4']['raw_data'].compression == None

    assert f['_version_data']['test_data']['raw_data'].compression_opts == None
    assert f['_version_data']['test_data2']['raw_data'].compression_opts == None
    assert f['_version_data']['test_data3']['raw_data'].compression_opts == None
    assert f['_version_data']['group']['test_data4']['raw_data'].compression_opts == None

    modify_metadata(f, 'group/test_data4', compression='gzip', compression_opts=3)
    check_data(vfile)

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


    assert vfile['version1']['group']['test_data4'].compression == 'gzip'
    assert vfile['version2']['group']['test_data4'].compression == 'gzip'
    assert vfile['version1']['group']['test_data4'].compression_opts == 3
    assert vfile['version2']['group']['test_data4'].compression_opts == 3

    assert f['_version_data']['test_data']['raw_data'].compression == None
    assert f['_version_data']['test_data2']['raw_data'].compression == None
    assert f['_version_data']['test_data3']['raw_data'].compression == None
    assert f['_version_data']['group']['test_data4']['raw_data'].compression == 'gzip'

    assert f['_version_data']['test_data']['raw_data'].compression_opts == None
    assert f['_version_data']['test_data2']['raw_data'].compression_opts == None
    assert f['_version_data']['test_data3']['raw_data'].compression_opts == None
    assert f['_version_data']['group']['test_data4']['raw_data'].compression_opts == 3

    # Make sure the tmp group group has been destroyed.
    assert set(f['_version_data']) == {'test_data', 'test_data2',
                                        'test_data3', 'group', 'versions'}
    assert set(f['_version_data']['group']) == {'test_data4'}

def test_modify_metadata_chunks(vfile):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile['version1']['test_data'].chunks == (1000,)
    assert vfile['version2']['test_data'].chunks == (1000,)

    assert vfile['version1']['test_data2'].chunks == (1000,)
    assert vfile['version2']['test_data2'].chunks == (1000,)

    assert vfile['version2']['test_data3'].chunks == (1000,)

    assert vfile['version1']['group']['test_data4'].chunks == (1000,)
    assert vfile['version2']['group']['test_data4'].chunks == (1000,)

    assert f['_version_data']['test_data']['raw_data'].chunks == (1000,)
    assert f['_version_data']['test_data2']['raw_data'].chunks == (1000,)
    assert f['_version_data']['test_data3']['raw_data'].chunks == (1000,)
    assert f['_version_data']['group']['test_data4']['raw_data'].chunks == (1000,)

    modify_metadata(f, 'test_data2', chunks=(500,))
    check_data(vfile)

    assert vfile['version1']['test_data'].chunks == (1000,)
    assert vfile['version2']['test_data'].chunks == (1000,)

    assert vfile['version1']['test_data2'].chunks == (500,)
    assert vfile['version2']['test_data2'].chunks == (500,)

    assert vfile['version2']['test_data3'].chunks == (1000,)

    assert vfile['version1']['group']['test_data4'].chunks == (1000,)
    assert vfile['version2']['group']['test_data4'].chunks == (1000,)

    assert f['_version_data']['test_data']['raw_data'].chunks == (1000,)
    assert f['_version_data']['test_data2']['raw_data'].chunks == (500,)
    assert f['_version_data']['test_data3']['raw_data'].chunks == (1000,)
    assert f['_version_data']['group']['test_data4']['raw_data'].chunks == (1000,)

    # Make sure the tmp group group has been destroyed.
    assert set(f['_version_data']) == {'test_data', 'test_data2',
                                        'test_data3', 'group', 'versions'}
    assert set(f['_version_data']['group']) == {'test_data4'}

def test_modify_metadata_chunk2(vfile):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile['version1']['test_data'].chunks == (1000,)
    assert vfile['version2']['test_data'].chunks == (1000,)

    assert vfile['version1']['test_data2'].chunks == (1000,)
    assert vfile['version2']['test_data2'].chunks == (1000,)

    assert vfile['version2']['test_data3'].chunks == (1000,)

    assert vfile['version1']['group']['test_data4'].chunks == (1000,)
    assert vfile['version2']['group']['test_data4'].chunks == (1000,)

    assert f['_version_data']['test_data']['raw_data'].chunks == (1000,)
    assert f['_version_data']['test_data2']['raw_data'].chunks == (1000,)
    assert f['_version_data']['test_data3']['raw_data'].chunks == (1000,)
    assert f['_version_data']['group']['test_data4']['raw_data'].chunks == (1000,)

    modify_metadata(f, 'group/test_data4', chunks=(500,))
    check_data(vfile)

    assert vfile['version1']['test_data'].chunks == (1000,)
    assert vfile['version2']['test_data'].chunks == (1000,)

    assert vfile['version1']['test_data2'].chunks == (1000,)
    assert vfile['version2']['test_data2'].chunks == (1000,)

    assert vfile['version2']['test_data3'].chunks == (1000,)

    assert vfile['version1']['group']['test_data4'].chunks == (500,)
    assert vfile['version2']['group']['test_data4'].chunks == (500,)

    assert f['_version_data']['test_data']['raw_data'].chunks == (1000,)
    assert f['_version_data']['test_data2']['raw_data'].chunks == (1000,)
    assert f['_version_data']['test_data3']['raw_data'].chunks == (1000,)
    assert f['_version_data']['group']['test_data4']['raw_data'].chunks == (500,)

    # Make sure the tmp group group has been destroyed.
    assert set(f['_version_data']) == {'test_data', 'test_data2',
                                        'test_data3', 'group', 'versions'}
    assert set(f['_version_data']['group']) == {'test_data4'}

def test_modify_metadata_dtype(vfile):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile['version1']['test_data'].dtype == np.float64
    assert vfile['version2']['test_data'].dtype == np.float64

    assert vfile['version1']['test_data2'].dtype == np.int64
    assert vfile['version2']['test_data2'].dtype == np.int64

    assert vfile['version2']['test_data3'].dtype == np.int64

    assert vfile['version1']['group']['test_data4'].dtype == np.int64
    assert vfile['version2']['group']['test_data4'].dtype == np.int64

    assert f['_version_data']['test_data']['raw_data'].dtype == np.float64
    assert f['_version_data']['test_data2']['raw_data'].dtype == np.int64
    assert f['_version_data']['test_data3']['raw_data'].dtype == np.int64
    assert f['_version_data']['group']['test_data4']['raw_data'].dtype == np.int64

    modify_metadata(f, 'test_data2', dtype=np.float64)
    check_data(vfile)


    assert vfile['version1']['test_data'].dtype == np.float64
    assert vfile['version2']['test_data'].dtype == np.float64

    assert vfile['version1']['test_data2'].dtype == np.float64
    assert vfile['version2']['test_data2'].dtype == np.float64

    assert vfile['version2']['test_data3'].dtype == np.int64

    assert vfile['version1']['group']['test_data4'].dtype == np.int64
    assert vfile['version2']['group']['test_data4'].dtype == np.int64

    assert f['_version_data']['test_data']['raw_data'].dtype == np.float64
    assert f['_version_data']['test_data2']['raw_data'].dtype == np.float64
    assert f['_version_data']['test_data3']['raw_data'].dtype == np.int64
    assert f['_version_data']['group']['test_data4']['raw_data'].dtype == np.int64

    # Make sure the tmp group group has been destroyed.
    assert set(f['_version_data']) == {'test_data', 'test_data2',
                                        'test_data3', 'group', 'versions'}
    assert set(f['_version_data']['group']) == {'test_data4'}

def test_modify_metadata_dtype2(vfile):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile['version1']['test_data'].dtype == np.float64
    assert vfile['version2']['test_data'].dtype == np.float64

    assert vfile['version1']['test_data2'].dtype == np.int64
    assert vfile['version2']['test_data2'].dtype == np.int64

    assert vfile['version2']['test_data3'].dtype == np.int64

    assert vfile['version1']['group']['test_data4'].dtype == np.int64
    assert vfile['version2']['group']['test_data4'].dtype == np.int64

    assert f['_version_data']['test_data']['raw_data'].dtype == np.float64
    assert f['_version_data']['test_data2']['raw_data'].dtype == np.int64
    assert f['_version_data']['test_data3']['raw_data'].dtype == np.int64
    assert f['_version_data']['group']['test_data4']['raw_data'].dtype == np.int64

    modify_metadata(f, 'group/test_data4', dtype=np.float64)
    check_data(vfile)


    assert vfile['version1']['test_data'].dtype == np.float64
    assert vfile['version2']['test_data'].dtype == np.float64

    assert vfile['version1']['test_data2'].dtype == np.int64
    assert vfile['version2']['test_data2'].dtype == np.int64

    assert vfile['version2']['test_data3'].dtype == np.int64

    assert vfile['version1']['group']['test_data4'].dtype == np.float64
    assert vfile['version2']['group']['test_data4'].dtype == np.float64

    assert f['_version_data']['test_data']['raw_data'].dtype == np.float64
    assert f['_version_data']['test_data2']['raw_data'].dtype == np.int64
    assert f['_version_data']['test_data3']['raw_data'].dtype == np.int64
    assert f['_version_data']['group']['test_data4']['raw_data'].dtype == np.float64

    # Make sure the tmp group group has been destroyed.
    assert set(f['_version_data']) == {'test_data', 'test_data2',
                                        'test_data3', 'group', 'versions'}
    assert set(f['_version_data']['group']) == {'test_data4'}

def test_modify_metadata_fillvalue1(vfile):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile['version1']['test_data'].fillvalue == 1.
    assert vfile['version2']['test_data'].fillvalue == 1.

    assert vfile['version1']['test_data2'].fillvalue == 0
    assert vfile['version2']['test_data2'].fillvalue == 0

    assert vfile['version2']['test_data3'].fillvalue == 0

    assert vfile['version1']['group']['test_data4'].fillvalue == 0
    assert vfile['version2']['group']['test_data4'].fillvalue == 0

    assert f['_version_data']['test_data']['raw_data'].fillvalue == 1.
    assert f['_version_data']['test_data2']['raw_data'].fillvalue == 0
    assert f['_version_data']['test_data3']['raw_data'].fillvalue == 0
    assert f['_version_data']['group']['test_data4']['raw_data'].fillvalue == 0

    modify_metadata(f, 'test_data', fillvalue=3.)
    check_data(vfile, test_data_fillvalue=3.)

    assert vfile['version1']['test_data'].fillvalue == 3.
    assert vfile['version2']['test_data'].fillvalue == 3.

    assert vfile['version1']['test_data2'].fillvalue == 0
    assert vfile['version2']['test_data2'].fillvalue == 0

    assert vfile['version2']['test_data3'].fillvalue == 0

    assert vfile['version1']['group']['test_data4'].fillvalue == 0
    assert vfile['version2']['group']['test_data4'].fillvalue == 0

    assert f['_version_data']['test_data']['raw_data'].fillvalue == 3.
    assert f['_version_data']['test_data2']['raw_data'].fillvalue == 0
    assert f['_version_data']['test_data3']['raw_data'].fillvalue == 0
    assert f['_version_data']['group']['test_data4']['raw_data'].fillvalue == 0

    # Make sure the tmp group group has been destroyed.
    assert set(f['_version_data']) == {'test_data', 'test_data2',
                                        'test_data3', 'group', 'versions'}
    assert set(f['_version_data']['group']) == {'test_data4'}

def test_modify_metadata_fillvalue2(vfile):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile['version1']['test_data'].fillvalue == 1.
    assert vfile['version2']['test_data'].fillvalue == 1.

    assert vfile['version1']['test_data2'].fillvalue == 0
    assert vfile['version2']['test_data2'].fillvalue == 0

    assert vfile['version2']['test_data3'].fillvalue == 0

    assert vfile['version1']['group']['test_data4'].fillvalue == 0
    assert vfile['version2']['group']['test_data4'].fillvalue == 0

    assert f['_version_data']['test_data']['raw_data'].fillvalue == 1.
    assert f['_version_data']['test_data2']['raw_data'].fillvalue == 0
    assert f['_version_data']['test_data3']['raw_data'].fillvalue == 0
    assert f['_version_data']['group']['test_data4']['raw_data'].fillvalue == 0

    modify_metadata(f, 'test_data2', fillvalue=3)
    check_data(vfile)

    assert vfile['version1']['test_data'].fillvalue == 1.
    assert vfile['version2']['test_data'].fillvalue == 1.

    assert vfile['version1']['test_data2'].fillvalue == 3
    assert vfile['version2']['test_data2'].fillvalue == 3

    assert vfile['version2']['test_data3'].fillvalue == 0

    assert vfile['version1']['group']['test_data4'].fillvalue == 0
    assert vfile['version2']['group']['test_data4'].fillvalue == 0

    assert f['_version_data']['test_data']['raw_data'].fillvalue == 1.
    assert f['_version_data']['test_data2']['raw_data'].fillvalue == 3
    assert f['_version_data']['test_data3']['raw_data'].fillvalue == 0
    assert f['_version_data']['group']['test_data4']['raw_data'].fillvalue == 0

    # Make sure the tmp group group has been destroyed.
    assert set(f['_version_data']) == {'test_data', 'test_data2',
                                        'test_data3', 'group', 'versions'}
    assert set(f['_version_data']['group']) == {'test_data4'}

def test_modify_metadata_fillvalue3(vfile):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile['version1']['test_data'].fillvalue == 1.
    assert vfile['version2']['test_data'].fillvalue == 1.

    assert vfile['version1']['test_data2'].fillvalue == 0
    assert vfile['version2']['test_data2'].fillvalue == 0

    assert vfile['version2']['test_data3'].fillvalue == 0

    assert vfile['version1']['group']['test_data4'].fillvalue == 0
    assert vfile['version2']['group']['test_data4'].fillvalue == 0

    assert f['_version_data']['test_data']['raw_data'].fillvalue == 1.
    assert f['_version_data']['test_data2']['raw_data'].fillvalue == 0
    assert f['_version_data']['test_data3']['raw_data'].fillvalue == 0
    assert f['_version_data']['group']['test_data4']['raw_data'].fillvalue == 0

    modify_metadata(f, 'group/test_data4', fillvalue=2)
    check_data(vfile)

    assert vfile['version1']['test_data'].fillvalue == 1.
    assert vfile['version2']['test_data'].fillvalue == 1.

    assert vfile['version1']['test_data2'].fillvalue == 0
    assert vfile['version2']['test_data2'].fillvalue == 0

    assert vfile['version2']['test_data3'].fillvalue == 0

    assert vfile['version1']['group']['test_data4'].fillvalue == 2
    assert vfile['version2']['group']['test_data4'].fillvalue == 2

    assert f['_version_data']['test_data']['raw_data'].fillvalue == 1.
    assert f['_version_data']['test_data2']['raw_data'].fillvalue == 0
    assert f['_version_data']['test_data3']['raw_data'].fillvalue == 0
    assert f['_version_data']['group']['test_data4']['raw_data'].fillvalue == 2

    # Make sure the tmp group group has been destroyed.
    assert set(f['_version_data']) == {'test_data', 'test_data2',
                                        'test_data3', 'group', 'versions'}
    assert set(f['_version_data']['group']) == {'test_data4'}

def test_delete_version(vfile):
    setup_vfile(vfile)
    f = vfile.f

    delete_version(f, 'version2')
    check_data(vfile, version2=False)
    assert list(vfile) == ['version1']
    assert set(f['_version_data']) == {'group', 'test_data', 'test_data2', 'versions'}
    assert set(f['_version_data']['group']) == {'test_data4'}
    assert not np.isin(2., f['_version_data']['test_data']['raw_data'][:])
    assert not np.isin(5, f['_version_data']['group']['test_data4']['raw_data'][:])

def test_delete_versions(vfile):
    setup_vfile(vfile)
    with vfile.stage_version('version3') as g:
        g['test_data'][2000] = 3.
        g.create_dataset('test_data4', data=[1, 2, 3, 4], chunks=(1000,))
    f = vfile.f

    delete_versions(f, ['version2', 'version3'])
    check_data(vfile, version2=False)
    assert list(vfile) == ['version1']
    assert set(f['_version_data']) == {'group', 'test_data', 'test_data2', 'versions'}
    assert set(f['_version_data']['group']) == {'test_data4'}
    assert not np.isin(2., f['_version_data']['test_data']['raw_data'][:])
    assert not np.isin(5, f['_version_data']['group']['test_data4']['raw_data'][:])


def test_delete_versions_no_data(vfile):
    with vfile.stage_version('version1') as g:
        g.create_dataset('data', maxshape=(None, None), chunks=(20, 20), shape=(5, 5), dtype=np.dtype('int8'), fillvalue=0)

    with vfile.stage_version('version2') as g:
        g['data'][0] = 1

    f = vfile.f

    delete_versions(f, ['version2'])
    assert list(vfile) == ['version1']
    assert list(vfile['version1']) == ['data']
    assert vfile['version1']['data'].shape == (5, 5)
    assert np.all(vfile['version1']['data'][:] == 0)

def test_delete_versions_no_data2(vfile):
    with vfile.stage_version('version1') as g:
        g.create_dataset('data', maxshape=(None, None), chunks=(20, 20), shape=(5, 5), dtype=np.dtype('int8'), fillvalue=0)

    with vfile.stage_version('version2') as g:
        g['data'][0] = 1

    f = vfile.f

    delete_versions(f, ['version1'])
    assert list(vfile) == ['version2']
    assert list(vfile['version2']) == ['data']
    assert vfile['version2']['data'].shape == (5, 5)
    assert np.all(vfile['version2']['data'][1:] == 0)
    assert np.all(vfile['version2']['data'][0] == 1)

def test_delete_versions_nested_groups(vfile):
    data = []

    with vfile.stage_version('r0') as sv:
        data_group = sv.create_group('group1/group2')
        data.append(np.arange(500))
        data_group.create_dataset('test_data', maxshape=(None,), chunks=(1000), data=data[0])

    for i in range(1, 11):
        with vfile.stage_version(f'r{i}') as sv:
            data.append(np.random.randint(0, 1000, size=500))
            sv['group1']['group2']['test_data'][:] = data[-1]


    assert set(vfile) == {'r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10'}
    for i in range(11):
        assert list(vfile[f'r{i}']) == ['group1'], i
        assert list(vfile[f'r{i}']['group1']) == ['group2']
        assert list(vfile[f'r{i}']['group1']['group2']) == ['test_data']
        np.testing.assert_equal(vfile[f'r{i}']['group1']['group2']['test_data'][:], data[i])

    delete_versions(vfile, ['r3', 'r6'])

    assert set(vfile) == {'r0', 'r1', 'r2', 'r4', 'r5', 'r7', 'r8', 'r9', 'r10'}
    for i in range(11):
        if i in [3, 6]:
            continue
        assert list(vfile[f'r{i}']) == ['group1'], i
        assert list(vfile[f'r{i}']['group1']) == ['group2']
        assert list(vfile[f'r{i}']['group1']['group2']) == ['test_data']
        np.testing.assert_equal(vfile[f'r{i}']['group1']['group2']['test_data'][:], data[i])

def test_delete_versions_prev_version(vfile):
    with vfile.stage_version('r0') as g:
        g['foo'] = np.array([1, 2, 3])
    for i in range(1, 11):
        with vfile.stage_version(f'r{i}') as g:
            g['foo'][:] = np.array([1, i, 3])

    delete_versions(vfile, ['r1', 'r5', 'r8'])
    prev_versions = {
        '__first_version__': None,
        'r0': '__first_version__',
        'r2': 'r0',
        'r3': 'r2',
        'r4': 'r3',
        'r6': 'r4',
        'r7': 'r6',
        'r9': 'r7',
        'r10': 'r9',
    }

    for v in vfile:
        assert vfile[v].attrs['prev_version'] == prev_versions[v]

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

def test_delete_versions_variable_length_strings(vfile):
    with vfile.stage_version('r0') as sv:
        data = np.array(['foo'], dtype='O')
        sv.create_dataset('bar', data=data, dtype=h5py.string_dtype(encoding='ascii'))

    for i in range(1, 11):
        with vfile.stage_version('r{}'.format(i)) as sv:
            sv['bar'].resize((i+1,))
            sv['bar'][i] = 'foo'

    delete_versions(vfile, ['r2', 'r4', 'r6'])

def test_delete_versions_fillvalue_only_dataset(vfile):
    with vfile.stage_version('r0') as sv:
        sv.create_dataset('fillvalue_only', shape=(6,),
                          dtype=np.dtype('int64'), data=None,
                          maxshape=(None,), chunks=(10000,), fillvalue=0)
        sv.create_dataset('has_data', shape=(6,), dtype=np.dtype('int64'),
                          data=np.arange(6), maxshape=(None,),
                          chunks=(10000,), fillvalue=0)

    with vfile.stage_version('r1') as sv:
        sv['has_data'] = np.arange(5, -1, -1)

    delete_versions(vfile, ['r0'])

    with vfile.stage_version('r2') as sv:
        sv['fillvalue_only'][0] = 1

    assert set(vfile) == {'r1', 'r2'}
    assert set(vfile['r1']) == {'fillvalue_only', 'has_data'}
    assert set(vfile['r2']) == {'fillvalue_only', 'has_data'}
    np.testing.assert_equal(vfile['r1']['fillvalue_only'][:], 0)
    np.testing.assert_equal(vfile['r2']['fillvalue_only'][:],
                            np.array([1, 0, 0, 0, 0, 0]))
    np.testing.assert_equal(vfile['r1']['has_data'][:], np.arange(5, -1, -1))
    np.testing.assert_equal(vfile['r2']['has_data'][:], np.arange(5, -1, -1))

def test_delete_versions_current_version(vfile):
    with vfile.stage_version('r0') as sv:
        sv.create_dataset('bar', data=np.arange(10))

    for i in range(1, 11):
        with vfile.stage_version('r{}'.format(i)) as sv:
            sv['bar'] = np.arange(10 + i)

    delete_versions(vfile, ['r2', 'r4', 'r6', 'r8', 'r9', 'r10'])

    cv = vfile.current_version
    assert cv == 'r7'
    np.testing.assert_equal(vfile[cv]['bar'][:], np.arange(17))

def test_variable_length_strings(vfile):
    # Warning: this test will segfault with h5py 3.7.0
    # (https://github.com/h5py/h5py/pull/2111 fixes it)
    with vfile.stage_version('r0') as sv:
        g = sv.create_group('data')
        dt = h5py.string_dtype(encoding='ascii')
        g.create_dataset('foo', data=['foo', 'bar'], dtype=dt)

    for i in range(1, 7):
        with vfile.stage_version(f'r{i}') as sv:
            sv['data/foo'] = np.array([f'foo{i}', f'bar{i}'], dtype='O')

    delete_versions(vfile, ['r1'])

def test_delete_empty_dataset(vfile):
    """Test that deleting an empty dataset executes successfully."""
    with vfile.stage_version("r0") as sv:
        sv.create_dataset(
            "key0",
            data=np.array([]),
            maxshape=(None,),
            chunks=(10000,),
            compression="lzf",
        )

    # Raw data should be filled with fillvalue, but actual current
    # version dataset should have size 0.
    assert vfile.f['_version_data/key0/raw_data'][:].size == 10000
    assert vfile[vfile.current_version]['key0'][:].size == 0

    # Create a new version, checking again the size
    with vfile.stage_version("r1") as sv:
        sv["key0"].resize((0,))
    assert vfile.f['_version_data/key0/raw_data'][:].size == 10000
    assert vfile[vfile.current_version]['key0'][:].size == 0

    # Deleting a prior version should not change the data in the current version
    delete_versions(vfile, ["r0"])
    assert vfile.f['_version_data/key0/raw_data'][:].size == 10000
    assert vfile[vfile.current_version]['key0'][:].size == 0

    # Create a new version, then check if the data is the correct size
    with vfile.stage_version("r2") as sv:
        sv["key0"].resize((0,))

    assert vfile.f['_version_data/key0/raw_data'][:].size == 10000
    assert vfile[vfile.current_version]['key0'][:].size == 0
