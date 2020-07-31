import os
import itertools

from pytest import raises

import h5py

import datetime

import numpy as np
from numpy.testing import assert_equal

from ..backend import DEFAULT_CHUNK_SIZE
from ..api import VersionedHDF5File
from ..versions import TIMESTAMP_FMT
from ..wrappers import InMemoryArrayDataset, InMemoryDataset

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

        file.close()


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

        file.close()

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

        file.close()

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

        file.close()

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

        file.close()

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

        file.close()

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

        file.close()

def test_changes_dataset():
    # Testcase similar to those on generate_data.py
    test_data = np.ones((2*DEFAULT_CHUNK_SIZE,))

    name = "testname"

    with setup() as f:
        file = VersionedHDF5File(f)

        with file.stage_version('version1', '') as group:
            group.create_dataset(f'{name}/key', data=test_data)
            group.create_dataset(f'{name}/val', data=test_data)

        version1 = file['version1']
        assert version1.attrs['prev_version'] == '__first_version__'
        assert_equal(version1[f'{name}/key'], test_data)
        assert_equal(version1[f'{name}/val'], test_data)

        with file.stage_version('version2') as group:
            key_ds = group[f'{name}/key']
            val_ds = group[f'{name}/val']
            val_ds[0] = -1
            key_ds[0] = 0

        key = file['version2'][f'{name}/key']
        assert key.shape == (2*DEFAULT_CHUNK_SIZE,)
        assert_equal(key[0], 0)
        assert_equal(key[1:2*DEFAULT_CHUNK_SIZE], 1.0)

        val = file['version2'][f'{name}/val']
        assert val.shape == (2*DEFAULT_CHUNK_SIZE,)
        assert_equal(val[0], -1.0)
        assert_equal(val[1:2*DEFAULT_CHUNK_SIZE], 1.0)

        assert list(f['_version_data/versions/__first_version__']) == []
        assert list(f['_version_data/versions/version1']) == list(file['version1']) == [name]
        assert list(f['_version_data/versions/version2']) == list(file['version2']) == [name]

        file.close()

def test_small_dataset():
    # Test creating a dataset that is smaller than the chunk size
    with setup() as f:
        file = VersionedHDF5File(f)

        data = np.ones((100,))

        with file.stage_version("version1") as group:
            group.create_dataset("test", data=data, chunks=(2**14,))

        assert_equal(file['version1']['test'], data)

        file.close()

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

        file.close()

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

        file.close()

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

        # Resize after calling create_dataset, larger
        with file.stage_version('resize_after_create_larger', '') as group:
            group.create_dataset('data', data=offset_data)
            group['data'].resize((DEFAULT_CHUNK_SIZE + 4,))

            assert group['data'].shape == (DEFAULT_CHUNK_SIZE + 4,)
            assert_equal(group['data'][:DEFAULT_CHUNK_SIZE + 2], 1.0)
            assert_equal(group['data'][DEFAULT_CHUNK_SIZE + 2:], 0.0)

        group = file['resize_after_create_larger']
        assert group['data'].shape == (DEFAULT_CHUNK_SIZE + 4,)
        assert_equal(group['data'][:DEFAULT_CHUNK_SIZE + 2], 1.0)
        assert_equal(group['data'][DEFAULT_CHUNK_SIZE + 2:], 0.0)

        # Resize after calling create_dataset, smaller
        with file.stage_version('resize_after_create_smaller', '') as group:
            group.create_dataset('data', data=offset_data)
            group['data'].resize((DEFAULT_CHUNK_SIZE - 4,))

            assert group['data'].shape == (DEFAULT_CHUNK_SIZE - 4,)
            assert_equal(group['data'][:], 1.0)

        group = file['resize_after_create_smaller']
        assert group['data'].shape == (DEFAULT_CHUNK_SIZE - 4,)
        assert_equal(group['data'][:], 1.0)

        file.close()

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

        file.close()

def test_resize_multiple_dimensions():
    # Test semantics against raw HDF5

    shapes = range(5, 25, 5) # 5, 10, 15, 20
    chunks = (10, 10, 10)
    with setup() as f:
        file = VersionedHDF5File(f)
        for i, (oldshape, newshape) in\
            enumerate(itertools.combinations_with_replacement(itertools.product(shapes, repeat=3), 2)):
            data = np.arange(np.product(oldshape)).reshape(oldshape)
            # Get the ground truth from h5py
            f.create_dataset(f'data{i}', data=data, fillvalue=-1, chunks=chunks,
                             maxshape=(None, None, None))
            f[f'data{i}'].resize(newshape)
            new_data = f[f'data{i}'][()]

            # resize after creation
            with file.stage_version(f'version1_{i}') as group:
                group.create_dataset(f'dataset1_{i}', data=data, chunks=chunks,
                                     fillvalue=-1)
                group[f'dataset1_{i}'].resize(newshape)
                assert group[f'dataset1_{i}'].shape == newshape
                assert_equal(group[f'dataset1_{i}'][()], new_data)

            version1 = file[f'version1_{i}']
            assert version1[f'dataset1_{i}'].shape == newshape
            assert_equal(version1[f'dataset1_{i}'][()], new_data)

            # resize in a new version
            with file.stage_version(f'version2_1_{i}', '') as group:
                group.create_dataset(f'dataset2_{i}', data=data, chunks=chunks,
                                     fillvalue=-1)
            with file.stage_version(f'version2_2_{i}', f'version2_1_{i}') as group:
                group[f'dataset2_{i}'].resize(newshape)
                assert group[f'dataset2_{i}'].shape == newshape
                assert_equal(group[f'dataset2_{i}'][()], new_data, str((oldshape, newshape)))

            version2_2 = file[f'version2_2_{i}']
            assert version2_2[f'dataset2_{i}'].shape == newshape
            assert_equal(version2_2[f'dataset2_{i}'][()], new_data)

            # resize after some data is read in
            with file.stage_version(f'version3_1_{i}', '') as group:
                group.create_dataset(f'dataset3_{i}', data=data, chunks=chunks,
                                     fillvalue=-1)
            with file.stage_version(f'version3_2_{i}', f'version3_1_{i}') as group:
                # read in first and last chunks
                group[f'dataset3_{i}'][0, 0, 0]
                group[f'dataset3_{i}'][-1, -1, -1]
                group[f'dataset3_{i}'].resize(newshape)
                assert group[f'dataset3_{i}'].shape == newshape
                assert_equal(group[f'dataset3_{i}'][()], new_data)

            version3_2 = file[f'version3_2_{i}']
            assert version3_2[f'dataset3_{i}'].shape == newshape
            assert_equal(version3_2[f'dataset3_{i}'][()], new_data)

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

        file.close()


def test_timestamp_auto():
    with setup() as f:
        file = VersionedHDF5File(f)

        data = np.ones((2*DEFAULT_CHUNK_SIZE,))

        with file.stage_version('version1') as group:
            group.create_dataset('test_data', data=data)

        assert isinstance(file['version1'].attrs['timestamp'], str)


def test_timestamp_manual():
    with setup() as f:
        file = VersionedHDF5File(f)

        data1 = np.ones((2*DEFAULT_CHUNK_SIZE,))
        data2 = np.ones((3*DEFAULT_CHUNK_SIZE))

        ts1 = datetime.datetime(2020, 6, 29, 20, 12, 56, 2560, tzinfo=datetime.timezone.utc)
        ts2 = datetime.datetime(2020, 6, 29, 22, 12, 56, 2560)
        with file.stage_version('version1', timestamp=ts1) as group:
            group['test_data_1'] = data1

        assert file['version1'].attrs['timestamp'] == ts1.strftime(TIMESTAMP_FMT)

        with raises(ValueError):
            with file.stage_version('version2', timestamp=ts2) as group:
                group['test_data_2'] = data2

        with raises(TypeError):
            with file.stage_version('version3', timestamp='2020-6-29') as group:
                group['test_data_3'] = data1


def test_getitem_by_timestamp():
    with setup() as f:
        file = VersionedHDF5File(f)

        data = np.arange(2*DEFAULT_CHUNK_SIZE)

        with file.stage_version('version1') as group:
            group.create_dataset('test_data', data=data)

        v = file['version1']
        ts = datetime.datetime.strptime(v.attrs['timestamp'], TIMESTAMP_FMT)
        assert file[ts] == v

        dt = np.datetime64(ts.replace(tzinfo=None))
        assert file[dt] == v

        file.close()


def test_nonroot():
    with setup() as f:
        g = f.create_group('subgroup')
        file = VersionedHDF5File(g)

        test_data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                                    2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                    3*np.ones((DEFAULT_CHUNK_SIZE,))))


        with file.stage_version('version1', '') as group:
            group['test_data'] = test_data

        version1 = file['version1']
        assert version1.attrs['prev_version'] == '__first_version__'
        assert_equal(version1['test_data'], test_data)

        ds = f['/subgroup/_version_data/test_data/raw_data']

        assert ds.shape == (3*DEFAULT_CHUNK_SIZE,)
        assert_equal(ds[0:1*DEFAULT_CHUNK_SIZE], 1.0)
        assert_equal(ds[1*DEFAULT_CHUNK_SIZE:2*DEFAULT_CHUNK_SIZE], 2.0)
        assert_equal(ds[2*DEFAULT_CHUNK_SIZE:3*DEFAULT_CHUNK_SIZE], 3.0)

        file.close()

def test_attrs():
    with setup() as f:
        file = VersionedHDF5File(f)

        data = np.arange(2*DEFAULT_CHUNK_SIZE)

        with file.stage_version('version1') as group:
            group.create_dataset('test_data', data=data)

            test_data = group['test_data']
            assert test_data.attrs == {}
            test_data.attrs['test_attr'] = 0

        assert file['version1']['test_data'].attrs == \
            dict(f['_version_data']['versions']['version1']['test_data'].attrs) == \
                {'test_attr': 0}

        with file.stage_version('version2') as group:
            test_data = group['test_data']
            assert test_data.attrs == {'test_attr': 0}
            test_data.attrs['test_attr'] = 1

        assert file['version1']['test_data'].attrs == \
            dict(f['_version_data']['versions']['version1']['test_data'].attrs) == \
                {'test_attr': 0}

        assert file['version2']['test_data'].attrs == \
            dict(f['_version_data']['versions']['version2']['test_data'].attrs) == \
                {'test_attr': 1}

        file.close()

def test_auto_delete():
    with setup() as f:
        file = VersionedHDF5File(f)
        try:
            with file.stage_version('version1') as group:
                raise RuntimeError
        except RuntimeError:
            pass
        else:
            raise AssertionError("did not raise")

        # Make sure the version got deleted so that we can make it again

        data = np.arange(2*DEFAULT_CHUNK_SIZE)

        with file.stage_version('version1') as group:
            group.create_dataset('test_data', data=data)

        assert_equal(file['version1']['test_data'], data)

        file.close()

def test_delitem():
    with setup() as f:
        file = VersionedHDF5File(f)

        data = np.arange(2*DEFAULT_CHUNK_SIZE)

        with file.stage_version('version1') as group:
            group.create_dataset('test_data', data=data)

        with file.stage_version('version2') as group:
            group.create_dataset('test_data2', data=data)

        del file['version2']

        assert list(file) == ['version1']
        assert file.current_version == 'version1'

        with raises(KeyError):
            del file['version2']

        del file['version1']

        assert list(file) == []
        assert file.current_version == '__first_version__'

        file.close()

def test_groups():
    with setup() as f:
        file = VersionedHDF5File(f)

        data = np.ones(2*DEFAULT_CHUNK_SIZE)

        with file.stage_version('version1') as group:
            group.create_group('group1')
            group.create_dataset('group1/test_data', data=data)
            assert_equal(group['group1']['test_data'], data)
            assert_equal(group['group1/test_data'], data)

        version = file['version1']
        assert_equal(version['group1']['test_data'], data)
        assert_equal(version['group1/test_data'], data)

        with file.stage_version('version2', '') as group:
            group.create_dataset('group1/test_data', data=data)
            assert_equal(group['group1']['test_data'], data)
            assert_equal(group['group1/test_data'], data)

        version = file['version2']
        assert_equal(version['group1']['test_data'], data)
        assert_equal(version['group1/test_data'], data)

        with file.stage_version('version3', 'version1') as group:
            group['group1']['test_data'][0] = 0
            group['group1/test_data'][1] = 0

            assert_equal(group['group1']['test_data'][:2], 0)
            assert_equal(group['group1']['test_data'][2:], 1)

            assert_equal(group['group1/test_data'][:2], 0)
            assert_equal(group['group1/test_data'][2:], 1)

        version = file['version3']
        assert_equal(version['group1']['test_data'][:2], 0)
        assert_equal(version['group1']['test_data'][2:], 1)

        assert_equal(version['group1/test_data'][:2], 0)
        assert_equal(version['group1/test_data'][2:], 1)

        assert list(version) == ['group1']
        assert list(version['group1']) == ['test_data']

        with file.stage_version('version4', 'version3') as group:
            group.create_dataset('group2/test_data', data=2*data)

            assert_equal(group['group1']['test_data'][:2], 0)
            assert_equal(group['group1']['test_data'][2:], 1)
            assert_equal(group['group2']['test_data'][:], 2)

            assert_equal(group['group1/test_data'][:2], 0)
            assert_equal(group['group1/test_data'][2:], 1)
            assert_equal(group['group2/test_data'][:], 2)

        version = file['version4']
        assert_equal(version['group1']['test_data'][:2], 0)
        assert_equal(version['group1']['test_data'][2:], 1)
        assert_equal(group['group2']['test_data'][:], 2)

        assert_equal(version['group1/test_data'][:2], 0)
        assert_equal(version['group1/test_data'][2:], 1)
        assert_equal(group['group2/test_data'][:], 2)

        assert list(version) == ['group1', 'group2']
        assert list(version['group1']) == ['test_data']
        assert list(version['group2']) == ['test_data']

        with file.stage_version('version5', '') as group:
            group.create_dataset('group1/group2/test_data', data=data)
            assert_equal(group['group1']['group2']['test_data'], data)
            assert_equal(group['group1/group2']['test_data'], data)
            assert_equal(group['group1']['group2/test_data'], data)
            assert_equal(group['group1/group2/test_data'], data)

        version = file['version5']
        assert_equal(version['group1']['group2']['test_data'], data)
        assert_equal(version['group1/group2']['test_data'], data)
        assert_equal(version['group1']['group2/test_data'], data)
        assert_equal(version['group1/group2/test_data'], data)

        with file.stage_version('version6', '') as group:
            group.create_dataset('group1/test_data1', data=data)
            group.create_dataset('group1/group2/test_data2', data=2*data)
            group.create_dataset('group1/group2/group3/test_data3', data=3*data)
            group.create_dataset('group1/group2/test_data4', data=4*data)

            assert_equal(group['group1']['test_data1'], data)
            assert_equal(group['group1/test_data1'], data)

            assert_equal(group['group1']['group2']['test_data2'], 2*data)
            assert_equal(group['group1/group2']['test_data2'], 2*data)
            assert_equal(group['group1']['group2/test_data2'], 2*data)
            assert_equal(group['group1/group2/test_data2'], 2*data)

            assert_equal(group['group1']['group2']['group3']['test_data3'], 3*data)
            assert_equal(group['group1/group2']['group3']['test_data3'], 3*data)
            assert_equal(group['group1/group2']['group3/test_data3'], 3*data)
            assert_equal(group['group1']['group2/group3/test_data3'], 3*data)
            assert_equal(group['group1/group2/group3/test_data3'], 3*data)

            assert_equal(group['group1']['group2']['test_data4'], 4*data)
            assert_equal(group['group1/group2']['test_data4'], 4*data)
            assert_equal(group['group1']['group2/test_data4'], 4*data)
            assert_equal(group['group1/group2/test_data4'], 4*data)

            assert list(group) == ['group1']
            assert set(group['group1']) == {'group2', 'test_data1'}
            assert set(group['group1']['group2']) == set(group['group1/group2']) == {'group3', 'test_data2', 'test_data4'}
            assert list(group['group1']['group2']['group3']) == list(group['group1/group2/group3']) == ['test_data3']

        version = file['version6']
        assert_equal(version['group1']['test_data1'], data)
        assert_equal(version['group1/test_data1'], data)

        assert_equal(version['group1']['group2']['test_data2'], 2*data)
        assert_equal(version['group1/group2']['test_data2'], 2*data)
        assert_equal(version['group1']['group2/test_data2'], 2*data)
        assert_equal(version['group1/group2/test_data2'], 2*data)

        assert_equal(version['group1']['group2']['group3']['test_data3'], 3*data)
        assert_equal(version['group1/group2']['group3']['test_data3'], 3*data)
        assert_equal(version['group1/group2']['group3/test_data3'], 3*data)
        assert_equal(version['group1']['group2/group3/test_data3'], 3*data)
        assert_equal(version['group1/group2/group3/test_data3'], 3*data)

        assert_equal(version['group1']['group2']['test_data4'], 4*data)
        assert_equal(version['group1/group2']['test_data4'], 4*data)
        assert_equal(version['group1']['group2/test_data4'], 4*data)
        assert_equal(version['group1/group2/test_data4'], 4*data)

        assert list(version) == ['group1']
        assert set(version['group1']) == {'group2', 'test_data1'}
        assert set(version['group1']['group2']) == set(version['group1/group2']) == {'group3', 'test_data2', 'test_data4'}
        assert list(version['group1']['group2']['group3']) == list(version['group1/group2/group3']) == ['test_data3']

        with file.stage_version('version-bad', '') as group:
            raises(ValueError, lambda: group.create_dataset('/group1/test_data', data=data))
            raises(ValueError, lambda: group.create_group('/group1'))

        file.close()

def test_group_contains():
    with setup() as f:
        file = VersionedHDF5File(f)

        data = np.ones(2*DEFAULT_CHUNK_SIZE)

        with file.stage_version('version1') as group:
            group.create_dataset('group1/group2/test_data', data=data)
            assert 'group1' in group
            assert 'group2' in group['group1']
            assert 'test_data' in group['group1/group2']
            assert 'test_data' not in group
            assert 'test_data' not in group['group1']
            assert 'group1/group2' in group
            assert 'group1/group3' not in group
            assert 'group1/group2/test_data' in group
            assert 'group1/group3/test_data' not in group
            assert 'group1/group3/test_data2' not in group

        with file.stage_version('version2') as group:
            group.create_dataset('group1/group3/test_data2', data=data)
            assert 'group1' in group
            assert 'group2' in group['group1']
            assert 'group3' in group['group1']
            assert 'test_data' in group['group1/group2']
            assert 'test_data' not in group
            assert 'test_data' not in group['group1']
            assert 'test_data2' in group['group1/group3']
            assert 'test_data2' not in group['group1/group2']
            assert 'group1/group2' in group
            assert 'group1/group3' in group
            assert 'group1/group2/test_data' in group
            assert 'group1/group3/test_data' not in group
            assert 'group1/group3/test_data2' in group

        version1 = file['version1']
        version2 = file['version2']
        assert 'group1' in version1
        assert 'group1' in version2
        assert 'group2' in version1['group1']
        assert 'group2' in version2['group1']
        assert 'group3' not in version1['group1']
        assert 'group3' in version2['group1']
        assert 'group1/group2' in version1
        assert 'group1/group2' in version2
        assert 'group1/group3' not in version1
        assert 'group1/group3' in version2
        assert 'group1/group2/test_data' in version1
        assert 'group1/group2/test_data' in version2
        assert 'group1/group3/test_data' not in version1
        assert 'group1/group3/test_data' not in version2
        assert 'group1/group3/test_data2' not in version1
        assert 'group1/group3/test_data2' in version2
        assert 'test_data' in version1['group1/group2']
        assert 'test_data' in version2['group1/group2']
        assert 'test_data' not in version1
        assert 'test_data' not in version2
        assert 'test_data' not in version1['group1']
        assert 'test_data' not in version2['group1']
        assert 'test_data2' in version2['group1/group3']
        assert 'test_data2' not in version1['group1/group2']
        assert 'test_data2' not in version2['group1/group2']

        file.close()

def test_moved_file():
    # See issue #28. Make sure the virtual datasets do not hard-code the filename.
    with setup(file_name='test.hdf5') as f:
        file = VersionedHDF5File(f)

        data = np.ones(2*DEFAULT_CHUNK_SIZE)

        with file.stage_version('version1') as group:
            group['dataset'] = data
        file.close()

    with h5py.File('test.hdf5', 'r') as f:
        file = VersionedHDF5File(f)
        assert_equal(file['version1']['dataset'][:], data)
        file.close()

    # XXX: os.replace
    os.rename('test.hdf5', 'test2.hdf5')

    with h5py.File('test2.hdf5', 'r') as f:
        file = VersionedHDF5File(f)
        assert_equal(file['version1']['dataset'][:], data)

def test_list_assign():
    with setup() as f:
        file = VersionedHDF5File(f)

        data = [1, 2, 3]

        with file.stage_version('version1') as group:
            group['dataset'] = data

            assert_equal(group['dataset'][:], data)

        assert_equal(file['version1']['dataset'][:], data)
        file.close()

def test_nested_group():
    # Issue #66
    with setup() as f:
        file = VersionedHDF5File(f)

        data1 = np.array([1, 1])
        data2 = np.array([2, 2])

        with file.stage_version('1') as sv:
            sv.create_dataset('bar/baz', data=data1)
            assert_equal(sv['bar/baz'][:], data1)

        assert_equal(sv['bar/baz'][:], data1)

        with file.stage_version('2') as sv:
            sv.create_dataset('bar/bon/1/data/axes/date', data=data2)
            assert_equal(sv['bar/baz'][:], data1)
            assert_equal(sv['bar/bon/1/data/axes/date'][:], data2)

        version1 = file['1']
        version2 = file['2']
        assert_equal(version1['bar/baz'][:], data1)
        assert_equal(version2['bar/baz'][:], data1)
        assert 'bar/bon/1/data/axes/date' not in version1
        assert_equal(version2['bar/bon/1/data/axes/date'][:], data2)

        file.close()

def test_fillvalue():
    # Based on test_resize(), but only the resize largers that use the fill
    # value
    with setup() as f:
        file = VersionedHDF5File(f)

        fillvalue = 5.0

        no_offset_data = np.ones((2*DEFAULT_CHUNK_SIZE,))

        offset_data = np.concatenate((np.ones((DEFAULT_CHUNK_SIZE,)),
                                      np.ones((2,))))

        with file.stage_version('version1') as group:
            group.create_dataset('no_offset', data=no_offset_data, fillvalue=fillvalue)
            group.create_dataset('offset', data=offset_data, fillvalue=fillvalue)

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
        assert_equal(group['no_offset'][2*DEFAULT_CHUNK_SIZE:], fillvalue)
        assert_equal(group['offset'][:DEFAULT_CHUNK_SIZE + 2], 1.0)
        assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 2:], fillvalue)


        # Resize larger, non-chunk multiple
        with file.stage_version('larger_chunk_non_multiple', 'version1') as group:
            group['no_offset'].resize((3*DEFAULT_CHUNK_SIZE + 2,))
            group['offset'].resize((3*DEFAULT_CHUNK_SIZE + 2,))

        group = file['larger_chunk_non_multiple']
        assert group['no_offset'].shape == (3*DEFAULT_CHUNK_SIZE + 2,)
        assert group['offset'].shape == (3*DEFAULT_CHUNK_SIZE + 2,)
        assert_equal(group['no_offset'][:2*DEFAULT_CHUNK_SIZE], 1.0)
        assert_equal(group['no_offset'][2*DEFAULT_CHUNK_SIZE:], fillvalue)
        assert_equal(group['offset'][:DEFAULT_CHUNK_SIZE + 2], 1.0)
        assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 2:], fillvalue)

        # Resize after creation
        with file.stage_version('version2', 'version1') as group:
            # Cover the case where some data is already read in
            group['offset'][-1] = 2.0

            group['no_offset'].resize((3*DEFAULT_CHUNK_SIZE + 2,))
            group['offset'].resize((3*DEFAULT_CHUNK_SIZE + 2,))

            assert group['no_offset'].shape == (3*DEFAULT_CHUNK_SIZE + 2,)
            assert group['offset'].shape == (3*DEFAULT_CHUNK_SIZE + 2,)
            assert_equal(group['no_offset'][:2*DEFAULT_CHUNK_SIZE], 1.0)
            assert_equal(group['no_offset'][2*DEFAULT_CHUNK_SIZE:], fillvalue)
            assert_equal(group['offset'][:DEFAULT_CHUNK_SIZE + 1], 1.0)
            assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 1], 2.0)
            assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 2:], fillvalue)

            group['no_offset'].resize((3*DEFAULT_CHUNK_SIZE,))
            group['offset'].resize((3*DEFAULT_CHUNK_SIZE,))

            assert group['no_offset'].shape == (3*DEFAULT_CHUNK_SIZE,)
            assert group['offset'].shape == (3*DEFAULT_CHUNK_SIZE,)
            assert_equal(group['no_offset'][:2*DEFAULT_CHUNK_SIZE], 1.0)
            assert_equal(group['no_offset'][2*DEFAULT_CHUNK_SIZE:], fillvalue)
            assert_equal(group['offset'][:DEFAULT_CHUNK_SIZE + 1], 1.0)
            assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 1], 2.0)
            assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 2:], fillvalue)

        group = file['version2']
        assert group['no_offset'].shape == (3*DEFAULT_CHUNK_SIZE,)
        assert group['offset'].shape == (3*DEFAULT_CHUNK_SIZE,)
        assert_equal(group['no_offset'][:2*DEFAULT_CHUNK_SIZE], 1.0)
        assert_equal(group['no_offset'][2*DEFAULT_CHUNK_SIZE:], fillvalue)
        assert_equal(group['offset'][:DEFAULT_CHUNK_SIZE + 1], 1.0)
        assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 1], 2.0)
        assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 2:], fillvalue)

        # Resize after calling create_dataset, larger
        with file.stage_version('resize_after_create_larger', '') as group:
            group.create_dataset('data', data=offset_data,
                                 fillvalue=fillvalue)
            group['data'].resize((DEFAULT_CHUNK_SIZE + 4,))

            assert group['data'].shape == (DEFAULT_CHUNK_SIZE + 4,)
            assert_equal(group['data'][:DEFAULT_CHUNK_SIZE + 2], 1.0)
            assert_equal(group['data'][DEFAULT_CHUNK_SIZE + 2:], fillvalue)

        group = file['resize_after_create_larger']
        assert group['data'].shape == (DEFAULT_CHUNK_SIZE + 4,)
        assert_equal(group['data'][:DEFAULT_CHUNK_SIZE + 2], 1.0)
        assert_equal(group['data'][DEFAULT_CHUNK_SIZE + 2:], fillvalue)

def test_multidimsional():
    with setup() as f:
        file = VersionedHDF5File(f)

        data = np.ones((2*DEFAULT_CHUNK_SIZE, 5))

        with file.stage_version('version1') as g:
            g.create_dataset('test_data', data=data,
                             chunks=(DEFAULT_CHUNK_SIZE, 2))
            assert_equal(g['test_data'][()], data)

        version1 = file['version1']
        assert_equal(version1['test_data'][()], data)

        data2 = data.copy()
        data2[0, 1] = 2

        with file.stage_version('version2') as g:
            g['test_data'][0, 1] = 2
            assert g['test_data'][0, 1] == 2
            assert_equal(g['test_data'][()], data2)

        version2 = file['version2']
        assert version2['test_data'][0, 1] == 2
        assert_equal(version2['test_data'][()], data2)

        data3 = data.copy()
        data3[0:1] = 3

        with file.stage_version('version3', 'version1') as g:
            g['test_data'][0:1] = 3
            assert_equal(g['test_data'][0:1], 3)
            assert_equal(g['test_data'][()], data3)

        version3 = file['version3']
        assert_equal(version3['test_data'][0:1], 3)
        assert_equal(version3['test_data'][()], data3)

def test_group_chunks_compression():
    # Chunks and compression are similar, so test them both at the same time.
    with setup() as f:
        file = VersionedHDF5File(f)

        data = np.ones((2*DEFAULT_CHUNK_SIZE, 5))

        with file.stage_version('version1') as g:
            g2 = g.create_group('group')
            g2.create_dataset('test_data', data=data,
                              chunks=(DEFAULT_CHUNK_SIZE, 2),
                              compression='gzip',
                              compression_opts=3)
            assert_equal(g2['test_data'][()], data)
            assert_equal(g['group/test_data'][()], data)
            assert_equal(g['group']['test_data'][()], data)

        version1 = file['version1']
        assert_equal(version1['group']['test_data'][()], data)
        assert_equal(version1['group/test_data'][()], data)

        raw_data = f['/_version_data/group/test_data/raw_data']
        assert raw_data.compression == 'gzip'
        assert raw_data.compression_opts == 3

        file.close()


def test_closes():
    with setup() as f:
        file = VersionedHDF5File(f)

        data = np.ones((DEFAULT_CHUNK_SIZE,))

        with file.stage_version('version1') as g:
            g.create_dataset('test_data', data=data)
        assert file._closed == False

        version_data = file._version_data
        versions = file._versions

        file.close()

        assert file._closed == True
        raises(AttributeError, lambda: file.f)
        raises(AttributeError, lambda: file._version_data)
        raises(AttributeError, lambda: file._versions)
        assert file.__repr__() == "<Closed VersionedHDF5File>"

        reopened_file = VersionedHDF5File(f)
        assert list(reopened_file['/_version_data/versions/__first_version__']) == []
        assert list(reopened_file['/_version_data/versions/version1']) == list(reopened_file['version1']) == ['test_data']
        assert_equal(reopened_file['version1']['test_data'][()], data)

        assert reopened_file._version_data == version_data
        assert reopened_file._versions == versions

def test_store_binary_as_void():
    with setup() as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version('version1') as sv:
            sv['test_store_binary_data'] = [np.void(b'1111')]

        version1 = vf['version1']
        assert_equal(version1['test_store_binary_data'][0], np.void(b'1111'))

        with vf.stage_version('version2') as sv:
            sv['test_store_binary_data'][:] = [np.void(b'1234567890')]

        version2 = vf['version2']
        assert_equal(version2['test_store_binary_data'][0], np.void(b'1234'))

def test_check_committed():
    with setup() as f:
        file = VersionedHDF5File(f)

        data = np.ones((DEFAULT_CHUNK_SIZE,))

        with file.stage_version('version1') as g:
            g.create_dataset('test_data', data=data)

        with raises(ValueError, match="committed"):
            g['data'] = data

        with raises(ValueError, match="committed"):
            g.create_dataset('data', data=data)

        with raises(ValueError, match="committed"):
            g.create_group('subgruop')

        with raises(ValueError, match="committed"):
            del g['test_data']

        # Incorrectly uses g from the previous version (InMemoryArrayDataset)
        with raises(ValueError, match="committed"):
            with file.stage_version('version2'):
                assert isinstance(g['test_data'], InMemoryArrayDataset)
                g['test_data'][0] = 1

        with raises(ValueError, match="committed"):
            with file.stage_version('version2'):
                assert isinstance(g['test_data'], InMemoryArrayDataset)
                g['test_data'].resize((100,))

        with file.stage_version('version2') as g2:
            pass

        # Incorrectly uses g from the previous version (InMemoryDataset)
        with raises(ValueError, match="committed"):
            with file.stage_version('version3'):
                assert isinstance(g2['test_data'], InMemoryDataset)
                g2['test_data'][0] = 1

        with raises(ValueError, match="committed"):
            with file.stage_version('version3'):
                assert isinstance(g2['test_data'], InMemoryDataset)
                g2['test_data'].resize((100,))

        assert repr(g) == '<Committed InMemoryGroup "/_version_data/versions/version1">'

def test_set_chunks_nested():
    with setup() as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version('0') as sv:
            data_group = sv.create_group('data')
            data_group.create_dataset('bar', data=np.arange(4))

        with vf.stage_version('1') as sv:
            data_group = sv['data']
            data_group.create_dataset('props/1/bar', data=np.arange(0, 4, 2))

def test_InMemoryArrayDataset_chunks():
    with setup() as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version('0') as sv:
            data_group = sv.create_group('data')
            data_group.create_dataset('g/bar', data=np.arange(4),
    chunks=(100,), compression='gzip', compression_opts=3)
            assert isinstance(data_group['g/bar'], InMemoryArrayDataset)
            assert data_group['g/bar'].chunks == (100,)
            assert data_group['g/bar'].compression == 'gzip'
            assert data_group['g/bar'].compression_opts == 3

def test_string_dtypes():
    # Make sure the fillvalue logic works correctly for custom h5py string dtypes.
    for typ, dt in [
            (str, h5py.string_dtype('utf-8')),
            (bytes, h5py.string_dtype('ascii')),
            # h5py uses bytes here
            (bytes, h5py.string_dtype('utf-8', length=20)),
            (bytes, h5py.string_dtype('ascii', length=20)),
            ]:

        if typ == str:
            data = np.full(10, 'hello world', dtype=dt)
        else:
            data = np.full(10, b'hello world', dtype=dt)
        with setup() as f:
            file = VersionedHDF5File(f)
            with file.stage_version('0') as sv:
                sv.create_dataset("name", shape=(10,), dtype=dt, data=data)
                assert isinstance(sv['name'], InMemoryArrayDataset)
                sv['name'].resize((11,))

            assert file['0']['name'].dtype == dt
            assert_equal(file['0']['name'][:10], data)
            assert file['0']['name'][10] == typ(), dt.metadata

            with file.stage_version('1') as sv:
                assert isinstance(sv['name'], InMemoryDataset)
                sv['name'].resize((12,))

            assert file['1']['name'].dtype == dt
            assert_equal(file['1']['name'][:10], data, str(dt.metadata))
            assert file['1']['name'][10] == typ(), dt.metadata
            assert file['1']['name'][11] == typ(), dt.metadata
