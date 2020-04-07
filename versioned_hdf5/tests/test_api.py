from pytest import raises

import numpy as np
from numpy.testing import assert_equal

from ..backend import DEFAULT_CHUNK_SIZE
from ..api import VersionedHDF5File

from .helpers import setup

def test_stage_version():
    with setup() as f:
        file = VersionedHDF5File(f)

        test_data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                                    2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                    3*np.ones((DEFAULT_CHUNK_SIZE,))))


        with file.stage_version('version1', '') as group:
            group['test_data'] = test_data

        version1 = file['version1']
        assert version1.attrs['prev_version'] == '__first_version__'
        assert_equal(version1['test_data'], test_data)

        ds = f['/_version_data/test_data/raw_data']

        assert ds.shape == (3*DEFAULT_CHUNK_SIZE,)
        assert_equal(ds[0:1*DEFAULT_CHUNK_SIZE], 1.0)
        assert_equal(ds[1*DEFAULT_CHUNK_SIZE:2*DEFAULT_CHUNK_SIZE], 2.0)
        assert_equal(ds[2*DEFAULT_CHUNK_SIZE:3*DEFAULT_CHUNK_SIZE], 3.0)

        with file.stage_version('version2', 'version1') as group:
            group['test_data'][0] = 0.0

        version2 = file['version2']
        assert version2.attrs['prev_version'] == 'version1'
        test_data[0] = 0.0
        assert_equal(version2['test_data'], test_data)

        assert ds.shape == (4*DEFAULT_CHUNK_SIZE,)
        assert_equal(ds[0:1*DEFAULT_CHUNK_SIZE], 1.0)
        assert_equal(ds[1*DEFAULT_CHUNK_SIZE:2*DEFAULT_CHUNK_SIZE], 2.0)
        assert_equal(ds[2*DEFAULT_CHUNK_SIZE:3*DEFAULT_CHUNK_SIZE], 3.0)
        assert_equal(ds[3*DEFAULT_CHUNK_SIZE], 0.0)
        assert_equal(ds[3*DEFAULT_CHUNK_SIZE+1:4*DEFAULT_CHUNK_SIZE], 1.0)

def test_stage_version_chunk_size():
    with setup() as f:
        chunk_size = 2**10
        file = VersionedHDF5File(f)

        test_data = np.concatenate((np.ones((2*chunk_size,)),
                                    2*np.ones((chunk_size,)),
                                    3*np.ones((chunk_size,))))


        with file.stage_version('version1', '') as group:
            group.create_dataset('test_data', data=test_data, chunks=(chunk_size,))

        with raises(ValueError):
            with file.stage_version('version_bad') as group:
                group.create_dataset('test_data', data=test_data, chunks=(2**9,))

        version1 = file['version1']
        assert version1.attrs['prev_version'] == '__first_version__'
        assert_equal(version1['test_data'], test_data)

        ds = f['/_version_data/test_data/raw_data']

        assert ds.shape == (3*chunk_size,)
        assert_equal(ds[0:1*chunk_size], 1.0)
        assert_equal(ds[1*chunk_size:2*chunk_size], 2.0)
        assert_equal(ds[2*chunk_size:3*chunk_size], 3.0)

        with file.stage_version('version2', 'version1') as group:
            group['test_data'][0] = 0.0

        version2 = file['version2']
        assert version2.attrs['prev_version'] == 'version1'
        test_data[0] = 0.0
        assert_equal(version2['test_data'], test_data)

        assert ds.shape == (4*chunk_size,)
        assert_equal(ds[0:1*chunk_size], 1.0)
        assert_equal(ds[1*chunk_size:2*chunk_size], 2.0)
        assert_equal(ds[2*chunk_size:3*chunk_size], 3.0)
        assert_equal(ds[3*chunk_size], 0.0)
        assert_equal(ds[3*chunk_size+1:4*chunk_size], 1.0)

def test_stage_version_compression():
    with setup() as f:
        file = VersionedHDF5File(f)

        test_data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                                    2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                    3*np.ones((DEFAULT_CHUNK_SIZE,))))


        with file.stage_version('version1', '') as group:
            group.create_dataset('test_data', data=test_data,
                                 compression='gzip', compression_opts=3)

        with raises(ValueError):
            with file.stage_version('version_bad') as group:
                group.create_dataset('test_data', data=test_data, compression='lzf')

        with raises(ValueError):
            with file.stage_version('version_bad') as group:
                group.create_dataset('test_data', data=test_data,
                                     compression='gzip', compression_opts=4)

        version1 = file['version1']
        assert version1.attrs['prev_version'] == '__first_version__'
        assert_equal(version1['test_data'], test_data)

        ds = f['/_version_data/test_data/raw_data']
        assert ds.compression == 'gzip'
        assert ds.compression_opts == 3

        with file.stage_version('version2', 'version1') as group:
            group['test_data'][0] = 0.0

        version2 = file['version2']
        assert version2.attrs['prev_version'] == 'version1'
        test_data[0] = 0.0
        assert_equal(version2['test_data'], test_data)

        assert ds.compression == 'gzip'
        assert ds.compression_opts == 3

def test_version_int_slicing():
    with setup() as f:
        file = VersionedHDF5File(f)
        test_data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                               2*np.ones((DEFAULT_CHUNK_SIZE,)),
                               3*np.ones((DEFAULT_CHUNK_SIZE,))))

        with file.stage_version('version1', '') as group:
            group['test_data'] = test_data

        with file.stage_version('version2', 'version1') as group:
            group['test_data'][0] = 2.0

        with file.stage_version('version3', 'version2') as group:
            group['test_data'][0] = 3.0

        with file.stage_version('version2_1', 'version1', make_current=False) as group:
            group['test_data'][0] = 2.0


        assert file[0]['test_data'][0] == 3.0

        with raises(KeyError):
            file['bad']

        with raises(IndexError):
            file[1]

        assert file[-1]['test_data'][0] == 2.0
        assert file[-2]['test_data'][0] == 1.0, file[-2]
        with raises(IndexError):
            file[-3]

        file.current_version = 'version2'

        assert file[0]['test_data'][0] == 2.0
        assert file[-1]['test_data'][0] == 1.0
        with raises(IndexError):
            file[-2]

        file.current_version = 'version2_1'

        assert file[0]['test_data'][0] == 2.0
        assert file[-1]['test_data'][0] == 1.0
        with raises(IndexError):
            file[-2]

        file.current_version = 'version1'

        assert file[0]['test_data'][0] == 1.0
        with raises(IndexError):
            file[-1]

def test_version_name_slicing():
    with setup() as f:
        file = VersionedHDF5File(f)
        test_data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                               2*np.ones((DEFAULT_CHUNK_SIZE,)),
                               3*np.ones((DEFAULT_CHUNK_SIZE,))))

        with file.stage_version('version1', '') as group:
            group['test_data'] = test_data

        with file.stage_version('version2', 'version1') as group:
            group['test_data'][0] = 2.0

        with file.stage_version('version3', 'version2') as group:
            group['test_data'][0] = 3.0

        with file.stage_version('version2_1', 'version1', make_current=False) as group:
            group['test_data'][0] = 2.0


        assert file[0]['test_data'][0] == 3.0

        with raises(IndexError):
            file[1]

        assert file[-1]['test_data'][0] == 2.0
        assert file[-2]['test_data'][0] == 1.0, file[-2]

def test_iter_versions():
    with setup() as f:
        file = VersionedHDF5File(f)
        test_data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                               2*np.ones((DEFAULT_CHUNK_SIZE,)),
                               3*np.ones((DEFAULT_CHUNK_SIZE,))))

        with file.stage_version('version1', '') as group:
            group['test_data'] = test_data

        with file.stage_version('version2', 'version1') as group:
            group['test_data'][0] = 2.0

        assert set(file) == {'version1', 'version2'}

        # __contains__ is implemented from __iter__ automatically
        assert 'version1' in file
        assert 'version2' in file
        assert 'version3' not in file

def test_create_dataset():
    with setup() as f:
        file = VersionedHDF5File(f)


        test_data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                                    2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                    3*np.ones((DEFAULT_CHUNK_SIZE,))))


        with file.stage_version('version1', '') as group:
            group.create_dataset('test_data', data=test_data)

        version1 = file['version1']
        assert version1.attrs['prev_version'] == '__first_version__'
        assert_equal(version1['test_data'], test_data)


        with file.stage_version('version2') as group:
            group.create_dataset('test_data2', data=test_data)

        ds = f['/_version_data/test_data/raw_data']
        assert ds.shape == (3*DEFAULT_CHUNK_SIZE,)
        assert_equal(ds[0:1*DEFAULT_CHUNK_SIZE], 1.0)
        assert_equal(ds[1*DEFAULT_CHUNK_SIZE:2*DEFAULT_CHUNK_SIZE], 2.0)
        assert_equal(ds[2*DEFAULT_CHUNK_SIZE:3*DEFAULT_CHUNK_SIZE], 3.0)

        ds = f['/_version_data/test_data2/raw_data']
        assert ds.shape == (3*DEFAULT_CHUNK_SIZE,)
        assert_equal(ds[0:1*DEFAULT_CHUNK_SIZE], 1.0)
        assert_equal(ds[1*DEFAULT_CHUNK_SIZE:2*DEFAULT_CHUNK_SIZE], 2.0)
        assert_equal(ds[2*DEFAULT_CHUNK_SIZE:3*DEFAULT_CHUNK_SIZE], 3.0)

        assert list(f['/_version_data/versions/__first_version__']) == []
        assert list(f['/_version_data/versions/version1']) == list(file['version1']) == ['test_data']
        assert list(f['/_version_data/versions/version2']) == list(file['version2']) == ['test_data', 'test_data2']

def test_small_dataset():
    # Test creating a dataset that is smaller than the chunk size
    with setup() as f:
        file = VersionedHDF5File(f)

        data = np.ones((100,))

        with file.stage_version("version1") as group:
            group.create_dataset("test", data=data, chunks=(2**14,))

        assert_equal(file['version1']['test'], data)

def test_unmodified():
    with setup() as f:
        file = VersionedHDF5File(f)

        test_data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                                    2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                    3*np.ones((DEFAULT_CHUNK_SIZE,))))

        with file.stage_version('version1') as group:
            group.create_dataset('test_data', data=test_data)
            group.create_dataset('test_data2', data=test_data)

        assert set(f['_version_data/versions/version1']) == {'test_data', 'test_data2'}
        assert set(file['version1']) == {'test_data', 'test_data2'}
        assert_equal(file['version1']['test_data'], test_data)
        assert_equal(file['version1']['test_data2'], test_data)
        assert file['version1'].datasets().keys() == {'test_data', 'test_data2'}

        with file.stage_version('version2') as group:
            group['test_data2'][0] = 0.0

        assert set(f['_version_data/versions/version2']) == {'test_data', 'test_data2'}
        assert set(file['version2']) == {'test_data', 'test_data2'}
        assert_equal(file['version2']['test_data'], test_data)
        assert_equal(file['version2']['test_data2'][0], 0.0)
        assert_equal(file['version2']['test_data2'][1:], test_data[1:])

def test_delete():
    with setup() as f:
        file = VersionedHDF5File(f)

        test_data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                                    2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                    3*np.ones((DEFAULT_CHUNK_SIZE,))))

        with file.stage_version('version1') as group:
            group.create_dataset('test_data', data=test_data)
            group.create_dataset('test_data2', data=test_data)

        with file.stage_version('version2') as group:
            del group['test_data2']

        assert set(f['_version_data/versions/version2']) == {'test_data'}
        assert set(file['version2']) == {'test_data'}
        assert_equal(file['version2']['test_data'], test_data)
        assert file['version2'].datasets().keys() == {'test_data'}

def test_resize():
    with setup() as f:
        file = VersionedHDF5File(f)

        no_offset_data = np.ones((2*DEFAULT_CHUNK_SIZE,))

        offset_data = np.concatenate((np.ones((DEFAULT_CHUNK_SIZE,)),
                                      np.ones((2,))))

        with file.stage_version('version1') as group:
            group.create_dataset('no_offset', data=no_offset_data)
            group.create_dataset('offset', data=offset_data)

        group = file['version1']
        assert group['no_offset'].shape == (2*DEFAULT_CHUNK_SIZE,)
        assert group['offset'].shape == (DEFAULT_CHUNK_SIZE + 2,)
        assert_equal(group['no_offset'][:2*DEFAULT_CHUNK_SIZE], 1.0)
        assert_equal(group['offset'][:DEFAULT_CHUNK_SIZE + 2], 1.0)

        # Resize larger, chunk multiple
        with file.stage_version('larger_chunk_multiple') as group:
            group['no_offset'].resize((3*DEFAULT_CHUNK_SIZE,))
            group['offset'].resize((3*DEFAULT_CHUNK_SIZE,))

        group = file['larger_chunk_multiple']
        assert group['no_offset'].shape == (3*DEFAULT_CHUNK_SIZE,)
        assert group['offset'].shape == (3*DEFAULT_CHUNK_SIZE,)
        assert_equal(group['no_offset'][:2*DEFAULT_CHUNK_SIZE], 1.0)
        assert_equal(group['no_offset'][2*DEFAULT_CHUNK_SIZE:], 0.0)
        assert_equal(group['offset'][:DEFAULT_CHUNK_SIZE + 2], 1.0)
        assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 2:], 0.0)


        # Resize larger, non-chunk multiple
        with file.stage_version('larger_chunk_non_multiple', 'version1') as group:
            group['no_offset'].resize((3*DEFAULT_CHUNK_SIZE + 2,))
            group['offset'].resize((3*DEFAULT_CHUNK_SIZE + 2,))

        group = file['larger_chunk_non_multiple']
        assert group['no_offset'].shape == (3*DEFAULT_CHUNK_SIZE + 2,)
        assert group['offset'].shape == (3*DEFAULT_CHUNK_SIZE + 2,)
        assert_equal(group['no_offset'][:2*DEFAULT_CHUNK_SIZE], 1.0)
        assert_equal(group['no_offset'][2*DEFAULT_CHUNK_SIZE:], 0.0)
        assert_equal(group['offset'][:DEFAULT_CHUNK_SIZE + 2], 1.0)
        assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 2:], 0.0)

        # Resize smaller, chunk multiple
        with file.stage_version('smaller_chunk_multiple', 'version1') as group:
            group['no_offset'].resize((DEFAULT_CHUNK_SIZE,))
            group['offset'].resize((DEFAULT_CHUNK_SIZE,))

        group = file['smaller_chunk_multiple']
        assert group['no_offset'].shape == (DEFAULT_CHUNK_SIZE,)
        assert group['offset'].shape == (DEFAULT_CHUNK_SIZE,)
        assert_equal(group['no_offset'][:], 1.0)
        assert_equal(group['offset'][:], 1.0)


        # Resize smaller, chunk non-multiple
        with file.stage_version('smaller_chunk_non_multiple', 'version1') as group:
            group['no_offset'].resize((DEFAULT_CHUNK_SIZE + 2,))
            group['offset'].resize((DEFAULT_CHUNK_SIZE + 2,))

        group = file['smaller_chunk_non_multiple']
        assert group['no_offset'].shape == (DEFAULT_CHUNK_SIZE + 2,)
        assert group['offset'].shape == (DEFAULT_CHUNK_SIZE + 2,)
        assert_equal(group['no_offset'][:], 1.0)
        assert_equal(group['offset'][:], 1.0)

        # Resize after creation
        with file.stage_version('version2', 'version1') as group:
            # Cover the case where some data is already read in
            group['offset'][-1] = 2.0

            group['no_offset'].resize((3*DEFAULT_CHUNK_SIZE + 2,))
            group['offset'].resize((3*DEFAULT_CHUNK_SIZE + 2,))

            assert group['no_offset'].shape == (3*DEFAULT_CHUNK_SIZE + 2,)
            assert group['offset'].shape == (3*DEFAULT_CHUNK_SIZE + 2,)
            assert_equal(group['no_offset'][:2*DEFAULT_CHUNK_SIZE], 1.0)
            assert_equal(group['no_offset'][2*DEFAULT_CHUNK_SIZE:], 0.0)
            assert_equal(group['offset'][:DEFAULT_CHUNK_SIZE + 1], 1.0)
            assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 1], 2.0)
            assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 2:], 0.0)

            group['no_offset'].resize((3*DEFAULT_CHUNK_SIZE,))
            group['offset'].resize((3*DEFAULT_CHUNK_SIZE,))

            assert group['no_offset'].shape == (3*DEFAULT_CHUNK_SIZE,)
            assert group['offset'].shape == (3*DEFAULT_CHUNK_SIZE,)
            assert_equal(group['no_offset'][:2*DEFAULT_CHUNK_SIZE], 1.0)
            assert_equal(group['no_offset'][2*DEFAULT_CHUNK_SIZE:], 0.0)
            assert_equal(group['offset'][:DEFAULT_CHUNK_SIZE + 1], 1.0)
            assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 1], 2.0)
            assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 2:], 0.0)

            group['no_offset'].resize((DEFAULT_CHUNK_SIZE + 2,))
            group['offset'].resize((DEFAULT_CHUNK_SIZE + 2,))

            assert group['no_offset'].shape == (DEFAULT_CHUNK_SIZE + 2,)
            assert group['offset'].shape == (DEFAULT_CHUNK_SIZE + 2,)
            assert_equal(group['no_offset'][:], 1.0)
            assert_equal(group['offset'][:DEFAULT_CHUNK_SIZE + 1], 1.0)
            assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 1], 2.0)

            group['no_offset'].resize((DEFAULT_CHUNK_SIZE,))
            group['offset'].resize((DEFAULT_CHUNK_SIZE,))

            assert group['no_offset'].shape == (DEFAULT_CHUNK_SIZE,)
            assert group['offset'].shape == (DEFAULT_CHUNK_SIZE,)
            assert_equal(group['no_offset'][:], 1.0)
            assert_equal(group['offset'][:], 1.0)

        group = file['version2']
        assert group['no_offset'].shape == (DEFAULT_CHUNK_SIZE,)
        assert group['offset'].shape == (DEFAULT_CHUNK_SIZE,)
        assert_equal(group['no_offset'][:], 1.0)
        assert_equal(group['offset'][:], 1.0)

        # Resize smaller than a chunk
        small_data = np.array([1, 2, 3])

        with file.stage_version('version1_small', '') as group:
            group.create_dataset('small', data=small_data)

        with file.stage_version('version2_small', 'version1_small') as group:
            group['small'].resize((5,))
            assert_equal(group['small'], np.array([1, 2, 3, 0, 0]))
            group['small'][3:] = np.array([4, 5])
            assert_equal(group['small'], np.array([1, 2, 3, 4, 5]))

        group = file['version1_small']
        assert_equal(group['small'], np.array([1, 2, 3]))
        group = file['version2_small']
        assert_equal(group['small'], np.array([1, 2, 3, 4, 5]))


def test_resize_unaligned():
    with setup() as f:
        file = VersionedHDF5File(f)
        ds_name = 'test_resize_unaligned'
        with file.stage_version('0') as group:
            group.create_dataset(ds_name, data=np.arange(1000))

        for i in range(1, 10):
            with file.stage_version(str(i)) as group:
                l = len(group[ds_name])
                assert_equal(group[ds_name][:], np.arange(i * 1000))
                group[ds_name].resize((l + 1000,))
                group[ds_name][-1000:] = np.arange(l, l + 1000)
                assert_equal(group[ds_name][:], np.arange((i + 1) * 1000))


def test_getitem():
    with setup() as f:
        file = VersionedHDF5File(f)

        data = np.arange(2*DEFAULT_CHUNK_SIZE)

        with file.stage_version('version1') as group:
            group.create_dataset('test_data', data=data)

            test_data = group['test_data']
            assert test_data.shape == (2*DEFAULT_CHUNK_SIZE,)
            assert_equal(test_data[0], 0)
            assert test_data[0].dtype == np.int64
            assert_equal(test_data[:], data)
            assert_equal(test_data[:DEFAULT_CHUNK_SIZE+1], data[:DEFAULT_CHUNK_SIZE+1])


        with file.stage_version('version2') as group:
            test_data = group['test_data']
            assert test_data.shape == (2*DEFAULT_CHUNK_SIZE,)
            assert_equal(test_data[0], 0)
            assert test_data[0].dtype == np.int64
            assert_equal(test_data[:], data)
            assert_equal(test_data[:DEFAULT_CHUNK_SIZE+1], data[:DEFAULT_CHUNK_SIZE+1])
