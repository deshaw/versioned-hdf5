from versioned_hdf5.backend import modify_metadata

def setup_modify_metadata(file):
    with file.stage_version('version1') as g:
        data = g.create_dataset('test_data', data=None, fillvalue=1., shape=(10000,), chunks=(1000,))
        data[0] = 0.
        g.create_dataset('test_data2', data=[1, 2, 3])

    with file.stage_version('version2') as g:
        g['test_data'][2000] = 2.
        g.create_dataset('test_data3', data=[1, 2, 3, 4])

def test_modify_metadata(vfile):
    setup_modify_metadata(vfile)

    f = vfile.f

    assert vfile['version1']['test_data'].compression == None
    assert vfile['version2']['test_data'].compression_opts == None

    assert vfile['version1']['test_data2'].compression == None
    assert vfile['version2']['test_data2'].compression_opts == None

    assert vfile['version2']['test_data3'].compression_opts == None

    assert f['_version_data']['test_data']['raw_data'].compression == None
    assert f['_version_data']['test_data2']['raw_data'].compression == None
    assert f['_version_data']['test_data3']['raw_data'].compression == None

    assert f['_version_data']['test_data']['raw_data'].compression_opts == None
    assert f['_version_data']['test_data2']['raw_data'].compression_opts == None
    assert f['_version_data']['test_data3']['raw_data'].compression_opts == None

    modify_metadata(f, 'test_data2', compression='gzip', compression_opts=3)

    assert vfile['version1']['test_data'].compression == None
    assert vfile['version2']['test_data'].compression_opts == None

    assert vfile['version1']['test_data2'].compression == 'gzip'
    assert vfile['version2']['test_data2'].compression_opts == 3

    assert vfile['version2']['test_data3'].compression_opts == None

    assert f['_version_data']['test_data']['raw_data'].compression == None
    assert f['_version_data']['test_data2']['raw_data'].compression == 'gzip'
    assert f['_version_data']['test_data3']['raw_data'].compression == None

    assert f['_version_data']['test_data']['raw_data'].compression_opts == None
    assert f['_version_data']['test_data2']['raw_data'].compression_opts == 3
    assert f['_version_data']['test_data3']['raw_data'].compression_opts == None

    # Make sure the tmp group group has been destroyed.
    assert list(f['_version_data']) == ['test_data', 'test_data2', 'test_data3', 'versions']
