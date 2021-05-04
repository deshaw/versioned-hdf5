import numpy as np

from versioned_hdf5.backend import modify_metadata

def setup_modify_metadata(file):
    with file.stage_version('version1') as g:
        data = g.create_dataset('test_data', data=None, fillvalue=1., shape=(10000,), chunks=(1000,))
        data[0] = 0.
        g.create_dataset('test_data2', data=[1, 2, 3], chunks=(1000,))

    with file.stage_version('version2') as g:
        g['test_data'][2000] = 2.
        g.create_dataset('test_data3', data=[1, 2, 3, 4], chunks=(1000,))

def check_modify_metadata_data(file, test_data_fillvalue=1.):
    assert file['version1']['test_data'].shape == (10000,)
    assert file['version1']['test_data'][0] == 0.
    assert np.all(file['version1']['test_data'][1:] == test_data_fillvalue)

    assert file['version2']['test_data'].shape == (10000,)
    assert file['version2']['test_data'][0] == 0.
    assert np.all(file['version2']['test_data'][1:2000] == test_data_fillvalue)
    assert file['version2']['test_data'][2000] == 2.
    assert np.all(file['version2']['test_data'][2001:] == test_data_fillvalue)

    assert file['version1']['test_data2'].shape == (3,)
    assert np.all(file['version1']['test_data2'][:] == [1, 2, 3])
    assert file['version2']['test_data2'].shape == (3,)
    assert np.all(file['version2']['test_data2'][:] == [1, 2, 3])

    assert 'test_data3' not in file['version1']
    assert file['version2']['test_data3'].shape == (4,)
    assert np.all(file['version2']['test_data3'][:] == [1, 2, 3, 4])

def test_modify_metadata_compression(vfile):
    setup_modify_metadata(vfile)

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
    check_modify_metadata_data(vfile)

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
    setup_modify_metadata(vfile)

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
    check_modify_metadata_data(vfile)

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
    setup_modify_metadata(vfile)

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
    check_modify_metadata_data(vfile)


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
    setup_modify_metadata(vfile)

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
    check_modify_metadata_data(vfile, test_data_fillvalue=3.)

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
    setup_modify_metadata(vfile)

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
    check_modify_metadata_data(vfile)

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
