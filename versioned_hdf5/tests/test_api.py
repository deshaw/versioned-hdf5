import logging
import os
import itertools

from pytest import raises, mark

import h5py

import datetime

import numpy as np
from numpy.testing import assert_equal

from .helpers import setup_vfile
from ..backend import DEFAULT_CHUNK_SIZE
from ..api import VersionedHDF5File
from ..versions import TIMESTAMP_FMT
from ..wrappers import (InMemoryArrayDataset, InMemoryDataset,
                        InMemorySparseDataset, DatasetWrapper, InMemoryGroup)


def test_stage_version(vfile):

    test_data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                                2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                3*np.ones((DEFAULT_CHUNK_SIZE,))))

    with vfile.stage_version('version1', '') as group:
        group['test_data'] = test_data

    version1 = vfile['version1']
    assert version1.attrs['prev_version'] == '__first_version__'
    assert_equal(version1['test_data'], test_data)

    ds = vfile.f['/_version_data/test_data/raw_data']

    assert ds.shape == (3*DEFAULT_CHUNK_SIZE,)
    assert_equal(ds[0:1*DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(ds[1*DEFAULT_CHUNK_SIZE:2*DEFAULT_CHUNK_SIZE], 2.0)
    assert_equal(ds[2*DEFAULT_CHUNK_SIZE:3*DEFAULT_CHUNK_SIZE], 3.0)

    with vfile.stage_version('version2', 'version1') as group:
        group['test_data'][0] = 0.0

    version2 = vfile['version2']
    assert version2.attrs['prev_version'] == 'version1'
    test_data[0] = 0.0
    assert_equal(version2['test_data'], test_data)

    assert ds.shape == (4*DEFAULT_CHUNK_SIZE,)
    assert_equal(ds[0:1*DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(ds[1*DEFAULT_CHUNK_SIZE:2*DEFAULT_CHUNK_SIZE], 2.0)
    assert_equal(ds[2*DEFAULT_CHUNK_SIZE:3*DEFAULT_CHUNK_SIZE], 3.0)
    assert_equal(ds[3*DEFAULT_CHUNK_SIZE], 0.0)
    assert_equal(ds[3*DEFAULT_CHUNK_SIZE+1:4*DEFAULT_CHUNK_SIZE], 1.0)


def test_stage_version_chunk_size(vfile):
    chunk_size = 2**10

    test_data = np.concatenate((np.ones((2*chunk_size,)),
                                2*np.ones((chunk_size,)),
                                3*np.ones((chunk_size,))))

    with vfile.stage_version('version1', '') as group:
        group.create_dataset('test_data', data=test_data, chunks=(chunk_size,))

    with raises(ValueError):
        with vfile.stage_version('version_bad') as group:
            group.create_dataset('test_data', data=test_data, chunks=(2**9,))

    version1 = vfile['version1']
    assert version1.attrs['prev_version'] == '__first_version__'
    assert_equal(version1['test_data'], test_data)

    ds = vfile.f['/_version_data/test_data/raw_data']

    assert ds.shape == (3*chunk_size,)
    assert_equal(ds[0:1*chunk_size], 1.0)
    assert_equal(ds[1*chunk_size:2*chunk_size], 2.0)
    assert_equal(ds[2*chunk_size:3*chunk_size], 3.0)

    with vfile.stage_version('version2', 'version1') as group:
        group['test_data'][0] = 0.0

    version2 = vfile['version2']
    assert version2.attrs['prev_version'] == 'version1'
    test_data[0] = 0.0
    assert_equal(version2['test_data'], test_data)

    assert ds.shape == (4*chunk_size,)
    assert_equal(ds[0:1*chunk_size], 1.0)
    assert_equal(ds[1*chunk_size:2*chunk_size], 2.0)
    assert_equal(ds[2*chunk_size:3*chunk_size], 3.0)
    assert_equal(ds[3*chunk_size], 0.0)
    assert_equal(ds[3*chunk_size+1:4*chunk_size], 1.0)


def test_stage_version_compression(vfile):

    test_data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                                2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                3*np.ones((DEFAULT_CHUNK_SIZE,))))

    with vfile.stage_version('version1', '') as group:
        group.create_dataset('test_data', data=test_data,
                             compression='gzip', compression_opts=3)

    with raises(ValueError):
        with vfile.stage_version('version_bad') as group:
            group.create_dataset('test_data', data=test_data, compression='lzf')

    with raises(ValueError):
        with vfile.stage_version('version_bad') as group:
            group.create_dataset('test_data', data=test_data,
                                 compression='gzip', compression_opts=4)

    version1 = vfile['version1']
    assert version1.attrs['prev_version'] == '__first_version__'
    assert_equal(version1['test_data'], test_data)

    ds = vfile.f['/_version_data/test_data/raw_data']
    assert ds.compression == 'gzip'
    assert ds.compression_opts == 3

    with vfile.stage_version('version2', 'version1') as group:
        group['test_data'][0] = 0.0

    version2 = vfile['version2']
    assert version2.attrs['prev_version'] == 'version1'
    test_data[0] = 0.0
    assert_equal(version2['test_data'], test_data)

    assert ds.compression == 'gzip'
    assert ds.compression_opts == 3


def test_version_int_slicing(vfile):
    test_data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                                2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                3*np.ones((DEFAULT_CHUNK_SIZE,))))

    with vfile.stage_version('version1', '') as group:
        group['test_data'] = test_data

    with vfile.stage_version('version2', 'version1') as group:
        group['test_data'][0] = 2.0

    with vfile.stage_version('version3', 'version2') as group:
        group['test_data'][0] = 3.0

    with vfile.stage_version('version2_1', 'version1', make_current=False) as group:
        group['test_data'][0] = 2.0

    assert vfile[0]['test_data'][0] == 3.0

    with raises(KeyError):
        vfile['bad']

    with raises(IndexError):
        vfile[1]

    assert vfile[-1]['test_data'][0] == 2.0
    assert vfile[-2]['test_data'][0] == 1.0, vfile[-2]
    with raises(IndexError):
        vfile[-3]

    vfile.current_version = 'version2'

    assert vfile[0]['test_data'][0] == 2.0
    assert vfile[-1]['test_data'][0] == 1.0
    with raises(IndexError):
        vfile[-2]

    vfile.current_version = 'version2_1'

    assert vfile[0]['test_data'][0] == 2.0
    assert vfile[-1]['test_data'][0] == 1.0
    with raises(IndexError):
        vfile[-2]

    vfile.current_version = 'version1'

    assert vfile[0]['test_data'][0] == 1.0
    with raises(IndexError):
        vfile[-1]


def test_version_name_slicing(vfile):
    test_data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                                2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                3*np.ones((DEFAULT_CHUNK_SIZE,))))

    with vfile.stage_version('version1', '') as group:
        group['test_data'] = test_data

    with vfile.stage_version('version2', 'version1') as group:
        group['test_data'][0] = 2.0

    with vfile.stage_version('version3', 'version2') as group:
        group['test_data'][0] = 3.0

    with vfile.stage_version('version2_1', 'version1', make_current=False) as group:
        group['test_data'][0] = 2.0

    assert vfile[0]['test_data'][0] == 3.0

    with raises(IndexError):
        vfile[1]

    assert vfile[-1]['test_data'][0] == 2.0
    assert vfile[-2]['test_data'][0] == 1.0, vfile[-2]

    with raises(ValueError):
        vfile['/_version_data']

def test_iter_versions(vfile):
    test_data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                                2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                3*np.ones((DEFAULT_CHUNK_SIZE,))))

    with vfile.stage_version('version1', '') as group:
        group['test_data'] = test_data

    with vfile.stage_version('version2', 'version1') as group:
        group['test_data'][0] = 2.0

    assert set(vfile) == {'version1', 'version2'}

    # __contains__ is implemented from __iter__ automatically
    assert 'version1' in vfile
    assert 'version2' in vfile
    assert 'version3' not in vfile


def test_create_dataset(vfile):
    test_data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                                2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                3*np.ones((DEFAULT_CHUNK_SIZE,))))

    with vfile.stage_version('version1', '') as group:
        group.create_dataset('test_data', data=test_data)

    version1 = vfile['version1']
    assert version1.attrs['prev_version'] == '__first_version__'
    assert_equal(version1['test_data'], test_data)

    with vfile.stage_version('version2') as group:
        group.create_dataset('test_data2', data=test_data)

    ds = vfile.f['/_version_data/test_data/raw_data']
    assert ds.shape == (3*DEFAULT_CHUNK_SIZE,)
    assert_equal(ds[0:1*DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(ds[1*DEFAULT_CHUNK_SIZE:2*DEFAULT_CHUNK_SIZE], 2.0)
    assert_equal(ds[2*DEFAULT_CHUNK_SIZE:3*DEFAULT_CHUNK_SIZE], 3.0)

    ds = vfile.f['/_version_data/test_data2/raw_data']
    assert ds.shape == (3*DEFAULT_CHUNK_SIZE,)
    assert_equal(ds[0:1*DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(ds[1*DEFAULT_CHUNK_SIZE:2*DEFAULT_CHUNK_SIZE], 2.0)
    assert_equal(ds[2*DEFAULT_CHUNK_SIZE:3*DEFAULT_CHUNK_SIZE], 3.0)

    assert list(vfile.f['/_version_data/versions/__first_version__']) == []
    assert list(vfile.f['/_version_data/versions/version1']) == list(vfile['version1']) == ['test_data']
    assert list(vfile.f['/_version_data/versions/version2']) == list(vfile['version2']) == ['test_data', 'test_data2']


def test_changes_dataset(vfile):
    # Testcase similar to those on generate_data.py
    test_data = np.ones((2*DEFAULT_CHUNK_SIZE,))

    name = "testname"

    with vfile.stage_version('version1', '') as group:
        group.create_dataset(f'{name}/key', data=test_data)
        group.create_dataset(f'{name}/val', data=test_data)

    version1 = vfile['version1']
    assert version1.attrs['prev_version'] == '__first_version__'
    assert_equal(version1[f'{name}/key'], test_data)
    assert_equal(version1[f'{name}/val'], test_data)

    with vfile.stage_version('version2') as group:
        key_ds = group[f'{name}/key']
        val_ds = group[f'{name}/val']
        val_ds[0] = -1
        key_ds[0] = 0

    key = vfile['version2'][f'{name}/key']
    assert key.shape == (2*DEFAULT_CHUNK_SIZE,)
    assert_equal(key[0], 0)
    assert_equal(key[1:2*DEFAULT_CHUNK_SIZE], 1.0)

    val = vfile['version2'][f'{name}/val']
    assert val.shape == (2*DEFAULT_CHUNK_SIZE,)
    assert_equal(val[0], -1.0)
    assert_equal(val[1:2*DEFAULT_CHUNK_SIZE], 1.0)

    assert list(vfile.f['_version_data/versions/__first_version__']) == []
    assert list(vfile.f['_version_data/versions/version1']) == list(vfile['version1']) == [name]
    assert list(vfile.f['_version_data/versions/version2']) == list(vfile['version2']) == [name]


def test_small_dataset(vfile):
    # Test creating a dataset that is smaller than the chunk size
    data = np.ones((100,))

    with vfile.stage_version("version1") as group:
        group.create_dataset("test", data=data, chunks=(2**14,))

    assert_equal(vfile['version1']['test'], data)


def test_unmodified(vfile):
    test_data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                                2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                3*np.ones((DEFAULT_CHUNK_SIZE,))))

    with vfile.stage_version('version1') as group:
        group.create_dataset('test_data', data=test_data)
        group.create_dataset('test_data2', data=test_data)

    assert set(vfile.f['_version_data/versions/version1']) == {'test_data', 'test_data2'}
    assert set(vfile['version1']) == {'test_data', 'test_data2'}
    assert_equal(vfile['version1']['test_data'], test_data)
    assert_equal(vfile['version1']['test_data2'], test_data)
    assert vfile['version1'].datasets().keys() == {'test_data', 'test_data2'}

    with vfile.stage_version('version2') as group:
        group['test_data2'][0] = 0.0

    assert set(vfile.f['_version_data/versions/version2']) == {'test_data', 'test_data2'}
    assert set(vfile['version2']) == {'test_data', 'test_data2'}
    assert_equal(vfile['version2']['test_data'], test_data)
    assert_equal(vfile['version2']['test_data2'][0], 0.0)
    assert_equal(vfile['version2']['test_data2'][1:], test_data[1:])


def test_delete_version(vfile):
    test_data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                                2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                3*np.ones((DEFAULT_CHUNK_SIZE,))))

    with vfile.stage_version('version1') as group:
        group.create_dataset('test_data', data=test_data)
        group.create_dataset('test_data2', data=test_data)

    with vfile.stage_version('version2') as group:
        del group['test_data2']

    assert set(vfile.f['_version_data/versions/version2']) == {'test_data'}
    assert set(vfile['version2']) == {'test_data'}
    assert_equal(vfile['version2']['test_data'], test_data)
    assert vfile['version2'].datasets().keys() == {'test_data'}


def test_resize(vfile):
    no_offset_data = np.ones((2*DEFAULT_CHUNK_SIZE,))

    offset_data = np.concatenate((np.ones((DEFAULT_CHUNK_SIZE,)),
                                  np.ones((2,))))

    with vfile.stage_version('version1') as group:
        group.create_dataset('no_offset', data=no_offset_data)
        group.create_dataset('offset', data=offset_data)

    group = vfile['version1']
    assert group['no_offset'].shape == (2*DEFAULT_CHUNK_SIZE,)
    assert group['offset'].shape == (DEFAULT_CHUNK_SIZE + 2,)
    assert_equal(group['no_offset'][:2*DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(group['offset'][:DEFAULT_CHUNK_SIZE + 2], 1.0)

    # Resize larger, chunk multiple
    with vfile.stage_version('larger_chunk_multiple') as group:
        group['no_offset'].resize((3*DEFAULT_CHUNK_SIZE,))
        group['offset'].resize((3*DEFAULT_CHUNK_SIZE,))

    group = vfile['larger_chunk_multiple']
    assert group['no_offset'].shape == (3*DEFAULT_CHUNK_SIZE,)
    assert group['offset'].shape == (3*DEFAULT_CHUNK_SIZE,)
    assert_equal(group['no_offset'][:2*DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(group['no_offset'][2*DEFAULT_CHUNK_SIZE:], 0.0)
    assert_equal(group['offset'][:DEFAULT_CHUNK_SIZE + 2], 1.0)
    assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 2:], 0.0)

    # Resize larger, non-chunk multiple
    with vfile.stage_version('larger_chunk_non_multiple', 'version1') as group:
        group['no_offset'].resize((3*DEFAULT_CHUNK_SIZE + 2,))
        group['offset'].resize((3*DEFAULT_CHUNK_SIZE + 2,))

    group = vfile['larger_chunk_non_multiple']
    assert group['no_offset'].shape == (3*DEFAULT_CHUNK_SIZE + 2,)
    assert group['offset'].shape == (3*DEFAULT_CHUNK_SIZE + 2,)
    assert_equal(group['no_offset'][:2*DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(group['no_offset'][2*DEFAULT_CHUNK_SIZE:], 0.0)
    assert_equal(group['offset'][:DEFAULT_CHUNK_SIZE + 2], 1.0)
    assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 2:], 0.0)

    # Resize smaller, chunk multiple
    with vfile.stage_version('smaller_chunk_multiple', 'version1') as group:
        group['no_offset'].resize((DEFAULT_CHUNK_SIZE,))
        group['offset'].resize((DEFAULT_CHUNK_SIZE,))

    group = vfile['smaller_chunk_multiple']
    assert group['no_offset'].shape == (DEFAULT_CHUNK_SIZE,)
    assert group['offset'].shape == (DEFAULT_CHUNK_SIZE,)
    assert_equal(group['no_offset'][:], 1.0)
    assert_equal(group['offset'][:], 1.0)

    # Resize smaller, chunk non-multiple
    with vfile.stage_version('smaller_chunk_non_multiple', 'version1') as group:
        group['no_offset'].resize((DEFAULT_CHUNK_SIZE + 2,))
        group['offset'].resize((DEFAULT_CHUNK_SIZE + 2,))

    group = vfile['smaller_chunk_non_multiple']
    assert group['no_offset'].shape == (DEFAULT_CHUNK_SIZE + 2,)
    assert group['offset'].shape == (DEFAULT_CHUNK_SIZE + 2,)
    assert_equal(group['no_offset'][:], 1.0)
    assert_equal(group['offset'][:], 1.0)

    # Resize after creation
    with vfile.stage_version('version2', 'version1') as group:
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

    group = vfile['version2']
    assert group['no_offset'].shape == (DEFAULT_CHUNK_SIZE,)
    assert group['offset'].shape == (DEFAULT_CHUNK_SIZE,)
    assert_equal(group['no_offset'][:], 1.0)
    assert_equal(group['offset'][:], 1.0)

    # Resize smaller than a chunk
    small_data = np.array([1, 2, 3])

    with vfile.stage_version('version1_small', '') as group:
        group.create_dataset('small', data=small_data)

    with vfile.stage_version('version2_small', 'version1_small') as group:
        group['small'].resize((5,))
        assert_equal(group['small'], np.array([1, 2, 3, 0, 0]))
        group['small'][3:] = np.array([4, 5])
        assert_equal(group['small'], np.array([1, 2, 3, 4, 5]))

    group = vfile['version1_small']
    assert_equal(group['small'], np.array([1, 2, 3]))
    group = vfile['version2_small']
    assert_equal(group['small'], np.array([1, 2, 3, 4, 5]))

    # Resize after calling create_dataset, larger
    with vfile.stage_version('resize_after_create_larger', '') as group:
        group.create_dataset('data', data=offset_data)
        group['data'].resize((DEFAULT_CHUNK_SIZE + 4,))

        assert group['data'].shape == (DEFAULT_CHUNK_SIZE + 4,)
        assert_equal(group['data'][:DEFAULT_CHUNK_SIZE + 2], 1.0)
        assert_equal(group['data'][DEFAULT_CHUNK_SIZE + 2:], 0.0)

    group = vfile['resize_after_create_larger']
    assert group['data'].shape == (DEFAULT_CHUNK_SIZE + 4,)
    assert_equal(group['data'][:DEFAULT_CHUNK_SIZE + 2], 1.0)
    assert_equal(group['data'][DEFAULT_CHUNK_SIZE + 2:], 0.0)

    # Resize after calling create_dataset, smaller
    with vfile.stage_version('resize_after_create_smaller', '') as group:
        group.create_dataset('data', data=offset_data)
        group['data'].resize((DEFAULT_CHUNK_SIZE - 4,))

        assert group['data'].shape == (DEFAULT_CHUNK_SIZE - 4,)
        assert_equal(group['data'][:], 1.0)

    group = vfile['resize_after_create_smaller']
    assert group['data'].shape == (DEFAULT_CHUNK_SIZE - 4,)
    assert_equal(group['data'][:], 1.0)


def test_resize_unaligned(vfile):
    ds_name = 'test_resize_unaligned'
    with vfile.stage_version('0') as group:
        group.create_dataset(ds_name, data=np.arange(1000))

    for i in range(1, 10):
        with vfile.stage_version(str(i)) as group:
            l = len(group[ds_name])
            assert_equal(group[ds_name][:], np.arange(i * 1000))
            group[ds_name].resize((l + 1000,))
            group[ds_name][-1000:] = np.arange(l, l + 1000)
            assert_equal(group[ds_name][:], np.arange((i + 1) * 1000))


@mark.slow
def test_resize_multiple_dimensions(tmp_path, h5file):
    # Test semantics against raw HDF5

    vfile = VersionedHDF5File(h5file)
    shapes = range(5, 25, 5)  # 5, 10, 15, 20
    chunks = (10, 10, 10)
    for i, (oldshape, newshape) in\
            enumerate(itertools.combinations_with_replacement(itertools.product(shapes, repeat=3), 2)):
        data = np.arange(np.product(oldshape)).reshape(oldshape)
        # Get the ground truth from h5py
        vfile.f.create_dataset(f'data{i}', data=data, fillvalue=-1, chunks=chunks,
                               maxshape=(None, None, None))
        vfile.f[f'data{i}'].resize(newshape)
        new_data = vfile.f[f'data{i}'][()]

        # resize after creation
        with vfile.stage_version(f'version1_{i}') as group:
            group.create_dataset(f'dataset1_{i}', data=data, chunks=chunks,
                                 fillvalue=-1)
            group[f'dataset1_{i}'].resize(newshape)
            assert group[f'dataset1_{i}'].shape == newshape
            assert_equal(group[f'dataset1_{i}'][()], new_data)

        version1 = vfile[f'version1_{i}']
        assert version1[f'dataset1_{i}'].shape == newshape
        assert_equal(version1[f'dataset1_{i}'][()], new_data)

        # resize in a new version
        with vfile.stage_version(f'version2_1_{i}', '') as group:
            group.create_dataset(f'dataset2_{i}', data=data, chunks=chunks,
                                 fillvalue=-1)
        with vfile.stage_version(f'version2_2_{i}', f'version2_1_{i}') as group:
            group[f'dataset2_{i}'].resize(newshape)
            assert group[f'dataset2_{i}'].shape == newshape
            assert_equal(group[f'dataset2_{i}'][()], new_data, str((oldshape, newshape)))

        version2_2 = vfile[f'version2_2_{i}']
        assert version2_2[f'dataset2_{i}'].shape == newshape
        assert_equal(version2_2[f'dataset2_{i}'][()], new_data)

        # resize after some data is read in
        with vfile.stage_version(f'version3_1_{i}', '') as group:
            group.create_dataset(f'dataset3_{i}', data=data, chunks=chunks,
                                 fillvalue=-1)
        with vfile.stage_version(f'version3_2_{i}', f'version3_1_{i}') as group:
            # read in first and last chunks
            group[f'dataset3_{i}'][0, 0, 0]
            group[f'dataset3_{i}'][-1, -1, -1]
            group[f'dataset3_{i}'].resize(newshape)
            assert group[f'dataset3_{i}'].shape == newshape
            assert_equal(group[f'dataset3_{i}'][()], new_data)

        version3_2 = vfile[f'version3_2_{i}']
        assert version3_2[f'dataset3_{i}'].shape == newshape
        assert_equal(version3_2[f'dataset3_{i}'][()], new_data)

def test_getitem(vfile):
    data = np.arange(2*DEFAULT_CHUNK_SIZE)

    with vfile.stage_version('version1') as group:
        group.create_dataset('test_data', data=data)

        test_data = group['test_data']
        assert test_data.shape == (2*DEFAULT_CHUNK_SIZE,)
        assert_equal(test_data[0], 0)
        assert test_data[0].dtype == np.int64
        assert_equal(test_data[:], data)
        assert_equal(test_data[:DEFAULT_CHUNK_SIZE+1], data[:DEFAULT_CHUNK_SIZE+1])

    with vfile.stage_version('version2') as group:
        test_data = group['test_data']
        assert test_data.shape == (2*DEFAULT_CHUNK_SIZE,)
        assert_equal(test_data[0], 0)
        assert test_data[0].dtype == np.int64
        assert_equal(test_data[:], data)
        assert_equal(test_data[:DEFAULT_CHUNK_SIZE+1], data[:DEFAULT_CHUNK_SIZE+1])


def test_timestamp_auto(vfile):
    data = np.ones((2*DEFAULT_CHUNK_SIZE,))

    with vfile.stage_version('version1') as group:
        group.create_dataset('test_data', data=data)

    assert isinstance(vfile['version1'].attrs['timestamp'], str)


def test_timestamp_manual(vfile):
    data1 = np.ones((2*DEFAULT_CHUNK_SIZE,))
    data2 = np.ones((3*DEFAULT_CHUNK_SIZE))

    ts1 = datetime.datetime(2020, 6, 29, 20, 12, 56, tzinfo=datetime.timezone.utc)
    ts2 = datetime.datetime(2020, 6, 29, 22, 12, 56)
    with vfile.stage_version('version1', timestamp=ts1) as group:
        group['test_data_1'] = data1

    assert vfile['version1'].attrs['timestamp'] == ts1.strftime(TIMESTAMP_FMT)

    with raises(ValueError):
        with vfile.stage_version('version2', timestamp=ts2) as group:
            group['test_data_2'] = data2

    with raises(TypeError):
        with vfile.stage_version('version3', timestamp='2020-6-29') as group:
            group['test_data_3'] = data1


def test_timestamp_manual_datetime64(vfile):
    data = np.ones((2*DEFAULT_CHUNK_SIZE,))

    # Also tests that it works correctly for 0 fractional part (issue #190).
    ts = datetime.datetime(2020, 6, 29, 20, 12, 56, tzinfo=datetime.timezone.utc)
    npts = np.datetime64(ts.replace(tzinfo=None))

    with vfile.stage_version('version1', timestamp=npts) as group:
        group['test_data'] = data

    v1 = vfile['version1']

    assert v1.attrs['timestamp'] == ts.strftime(TIMESTAMP_FMT)

    assert vfile[npts] == v1
    assert vfile[ts] == v1
    assert vfile.get_version_by_timestamp(npts, exact=True) == v1
    assert vfile.get_version_by_timestamp(ts, exact=True) == v1


def test_getitem_by_timestamp(vfile):
    data = np.arange(2*DEFAULT_CHUNK_SIZE)

    with vfile.stage_version('version1') as group:
        group.create_dataset('test_data', data=data)

    v1 = vfile['version1']
    ts1 = datetime.datetime.strptime(v1.attrs['timestamp'], TIMESTAMP_FMT)
    assert vfile[ts1] == v1
    assert vfile.get_version_by_timestamp(ts1) == v1
    assert vfile.get_version_by_timestamp(ts1, exact=True) == v1

    dt1 = np.datetime64(ts1.replace(tzinfo=None))
    assert vfile[dt1] == v1
    assert vfile.get_version_by_timestamp(dt1) == v1
    assert vfile.get_version_by_timestamp(dt1, exact=True) == v1

    minute = datetime.timedelta(minutes=1)
    second = datetime.timedelta(seconds=1)

    ts2 = ts1 + minute
    dt2 = np.datetime64(ts2.replace(tzinfo=None))

    with vfile.stage_version('version2', timestamp=ts2) as group:
        group['test_data'][0] += 1

    v2 = vfile['version2']
    assert vfile[ts2] == v2
    assert vfile.get_version_by_timestamp(ts2) == v2
    assert vfile.get_version_by_timestamp(ts2, exact=True) == v2

    assert vfile[dt2] == v2
    assert vfile.get_version_by_timestamp(dt2) == v2
    assert vfile.get_version_by_timestamp(dt2, exact=True) == v2


    ts2_1 = ts2 + second
    dt2_1 = np.datetime64(ts2_1.replace(tzinfo=None))

    assert vfile[ts2_1] == v2
    assert vfile.get_version_by_timestamp(ts2_1) == v2
    raises(KeyError, lambda: vfile.get_version_by_timestamp(ts2_1, exact=True))

    assert vfile[dt2_1] == v2
    assert vfile.get_version_by_timestamp(dt2_1) == v2
    raises(KeyError, lambda: vfile.get_version_by_timestamp(dt2_1, exact=True))

    ts1_1 = ts1 + second
    dt1_1 = np.datetime64(ts1_1.replace(tzinfo=None))

    assert vfile[ts1_1] == v1
    assert vfile.get_version_by_timestamp(ts1_1) == v1
    raises(KeyError, lambda: vfile.get_version_by_timestamp(ts1_1, exact=True))

    assert vfile[dt1_1] == v1
    assert vfile.get_version_by_timestamp(dt1_1) == v1
    raises(KeyError, lambda: vfile.get_version_by_timestamp(dt1_1, exact=True))

    ts0 = ts1 - second
    dt0 = np.datetime64(ts0.replace(tzinfo=None))

    raises(KeyError, lambda: vfile[ts0] == v1)
    raises(KeyError, lambda: vfile.get_version_by_timestamp(ts0) == v1)
    raises(KeyError, lambda: vfile.get_version_by_timestamp(ts0, exact=True))

    raises(KeyError, lambda: vfile[dt0] == v1)
    raises(KeyError, lambda: vfile.get_version_by_timestamp(dt0) == v1)
    raises(KeyError, lambda: vfile.get_version_by_timestamp(dt0, exact=True))

def test_nonroot(vfile):
    g = vfile.f.create_group('subgroup')
    file = VersionedHDF5File(g)

    test_data = np.concatenate((np.ones((2*DEFAULT_CHUNK_SIZE,)),
                                2*np.ones((DEFAULT_CHUNK_SIZE,)),
                                3*np.ones((DEFAULT_CHUNK_SIZE,))))

    with file.stage_version('version1', '') as group:
        group['test_data'] = test_data

    version1 = file['version1']
    assert version1.attrs['prev_version'] == '__first_version__'
    assert_equal(version1['test_data'], test_data)

    ds = vfile.f['/subgroup/_version_data/test_data/raw_data']

    assert ds.shape == (3*DEFAULT_CHUNK_SIZE,)
    assert_equal(ds[0:1*DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(ds[1*DEFAULT_CHUNK_SIZE:2*DEFAULT_CHUNK_SIZE], 2.0)
    assert_equal(ds[2*DEFAULT_CHUNK_SIZE:3*DEFAULT_CHUNK_SIZE], 3.0)


def test_attrs(vfile):
    data = np.arange(2*DEFAULT_CHUNK_SIZE)

    with vfile.stage_version('version1') as group:
        group.create_dataset('test_data', data=data)

        test_data = group['test_data']
        assert 'test_attr' not in test_data.attrs
        test_data.attrs['test_attr'] = 0

    assert vfile['version1']['test_data'].attrs['test_attr'] == \
        vfile.f['_version_data']['versions']['version1']['test_data'].attrs['test_attr'] == 0


    with vfile.stage_version('version2') as group:
        test_data = group['test_data']
        assert test_data.attrs['test_attr'] == 0
        test_data.attrs['test_attr'] = 1

    assert vfile['version1']['test_data'].attrs['test_attr'] == \
        vfile.f['_version_data']['versions']['version1']['test_data'].attrs['test_attr'] == 0


    assert vfile['version2']['test_data'].attrs['test_attr'] == \
        vfile.f['_version_data']['versions']['version2']['test_data'].attrs['test_attr'] == 1


def test_auto_delete(vfile):
    try:
        with vfile.stage_version('version1') as group:
            raise RuntimeError
    except RuntimeError:
        pass
    else:
        raise AssertionError("did not raise")

    # Make sure the version got deleted so that we can make it again
    data = np.arange(2*DEFAULT_CHUNK_SIZE)

    with vfile.stage_version('version1') as group:
        group.create_dataset('test_data', data=data)

    assert_equal(vfile['version1']['test_data'], data)


def test_delitem(vfile):
    data = np.arange(2*DEFAULT_CHUNK_SIZE)

    with vfile.stage_version('version1') as group:
        group.create_dataset('test_data', data=data)

    with vfile.stage_version('version2') as group:
        group.create_dataset('test_data2', data=data)

    del vfile['version2']

    assert list(vfile) == ['version1']
    assert vfile.current_version == 'version1'

    with raises(KeyError):
        del vfile['version2']

    del vfile['version1']

    assert list(vfile) == []
    assert vfile.current_version == '__first_version__'


def test_groups(vfile):
    data = np.ones(2*DEFAULT_CHUNK_SIZE)

    with vfile.stage_version('version1') as group:
        group.create_group('group1')
        group.create_dataset('group1/test_data', data=data)
        assert_equal(group['group1']['test_data'], data)
        assert_equal(group['group1/test_data'], data)

    version = vfile['version1']
    assert_equal(version['group1']['test_data'], data)
    assert_equal(version['group1/test_data'], data)

    with vfile.stage_version('version2', '') as group:
        group.create_dataset('group1/test_data', data=data)
        assert_equal(group['group1']['test_data'], data)
        assert_equal(group['group1/test_data'], data)

    version = vfile['version2']
    assert_equal(version['group1']['test_data'], data)
    assert_equal(version['group1/test_data'], data)

    with vfile.stage_version('version3', 'version1') as group:
        group['group1']['test_data'][0] = 0
        group['group1/test_data'][1] = 0

        assert_equal(group['group1']['test_data'][:2], 0)
        assert_equal(group['group1']['test_data'][2:], 1)

        assert_equal(group['group1/test_data'][:2], 0)
        assert_equal(group['group1/test_data'][2:], 1)

    version = vfile['version3']
    assert_equal(version['group1']['test_data'][:2], 0)
    assert_equal(version['group1']['test_data'][2:], 1)

    assert_equal(version['group1/test_data'][:2], 0)
    assert_equal(version['group1/test_data'][2:], 1)

    assert list(version) == ['group1']
    assert list(version['group1']) == ['test_data']

    with vfile.stage_version('version4', 'version3') as group:
        group.create_dataset('group2/test_data', data=2*data)

        assert_equal(group['group1']['test_data'][:2], 0)
        assert_equal(group['group1']['test_data'][2:], 1)
        assert_equal(group['group2']['test_data'][:], 2)

        assert_equal(group['group1/test_data'][:2], 0)
        assert_equal(group['group1/test_data'][2:], 1)
        assert_equal(group['group2/test_data'][:], 2)

    version = vfile['version4']
    assert_equal(version['group1']['test_data'][:2], 0)
    assert_equal(version['group1']['test_data'][2:], 1)
    assert_equal(group['group2']['test_data'][:], 2)

    assert_equal(version['group1/test_data'][:2], 0)
    assert_equal(version['group1/test_data'][2:], 1)
    assert_equal(group['group2/test_data'][:], 2)

    assert list(version) == ['group1', 'group2']
    assert list(version['group1']) == ['test_data']
    assert list(version['group2']) == ['test_data']

    with vfile.stage_version('version5', '') as group:
        group.create_dataset('group1/group2/test_data', data=data)
        assert_equal(group['group1']['group2']['test_data'], data)
        assert_equal(group['group1/group2']['test_data'], data)
        assert_equal(group['group1']['group2/test_data'], data)
        assert_equal(group['group1/group2/test_data'], data)

    version = vfile['version5']
    assert_equal(version['group1']['group2']['test_data'], data)
    assert_equal(version['group1/group2']['test_data'], data)
    assert_equal(version['group1']['group2/test_data'], data)
    assert_equal(version['group1/group2/test_data'], data)

    with vfile.stage_version('version6', '') as group:
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

    version = vfile['version6']
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

    with vfile.stage_version('version-bad', '') as group:
        raises(ValueError, lambda: group.create_dataset('/group1/test_data', data=data))
        raises(ValueError, lambda: group.create_group('/group1'))

def test_group_contains(vfile):
    data = np.ones(2*DEFAULT_CHUNK_SIZE)

    with vfile.stage_version('version1') as group:
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

    with vfile.stage_version('version2') as group:
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

    version1 = vfile['version1']
    version2 = vfile['version2']
    assert 'group1' in version1
    assert 'group1/' in version1
    assert 'group1' in version2
    assert 'group1/' in version2
    assert 'group2' in version1['group1']
    assert 'group2/' in version1['group1']
    assert 'group2' in version2['group1']
    assert 'group2/' in version2['group1']
    assert 'group3' not in version1['group1']
    assert 'group3/' not in version1['group1']
    assert 'group3' in version2['group1']
    assert 'group3/' in version2['group1']
    assert 'group1/group2' in version1
    assert 'group1/group2/' in version1
    assert 'group1/group2' in version2
    assert 'group1/group2/' in version2
    assert 'group1/group3' not in version1
    assert 'group1/group3/' not in version1
    assert 'group1/group3' in version2
    assert 'group1/group3/' in version2
    assert 'group1/group2/test_data' in version1
    assert 'group1/group2/test_data/' in version1
    assert 'group1/group2/test_data' in version2
    assert 'group1/group2/test_data/' in version2
    assert 'group1/group3/test_data' not in version1
    assert 'group1/group3/test_data/' not in version1
    assert 'group1/group3/test_data' not in version2
    assert 'group1/group3/test_data/' not in version2
    assert 'group1/group3/test_data2' not in version1
    assert 'group1/group3/test_data2/' not in version1
    assert 'group1/group3/test_data2' in version2
    assert 'group1/group3/test_data2/' in version2
    assert 'test_data' in version1['group1/group2']
    assert 'test_data' in version2['group1/group2']
    assert 'test_data' not in version1
    assert 'test_data' not in version2
    assert 'test_data' not in version1['group1']
    assert 'test_data' not in version2['group1']
    assert 'test_data2' in version2['group1/group3']
    assert 'test_data2' not in version1['group1/group2']
    assert 'test_data2' not in version2['group1/group2']

    assert '/_version_data/versions/version1/' in version1
    assert '/_version_data/versions/version1' in version1
    assert '/_version_data/versions/version1/' not in version2
    assert '/_version_data/versions/version1' not in version2
    assert '/_version_data/versions/version1/group1' in version1
    assert '/_version_data/versions/version1/group1' not in version2
    assert '/_version_data/versions/version1/group1/group2' in version1
    assert '/_version_data/versions/version1/group1/group2' not in version2

@mark.setup_args(file_name='test.hdf5')
def test_moved_file(tmp_path, h5file):
    # See issue #28. Make sure the virtual datasets do not hard-code the filename.
    file = VersionedHDF5File(h5file)
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
        file.close()


def test_list_assign(vfile):
    data = [1, 2, 3]

    with vfile.stage_version('version1') as group:
        group['dataset'] = data

        assert_equal(group['dataset'][:], data)

    assert_equal(vfile['version1']['dataset'][:], data)


def test_nested_group(vfile):
    # Issue #66
    data1 = np.array([1, 1])
    data2 = np.array([2, 2])

    with vfile.stage_version('1') as sv:
        sv.create_dataset('bar/baz', data=data1)
        assert_equal(sv['bar/baz'][:], data1)

    assert_equal(sv['bar/baz'][:], data1)

    with vfile.stage_version('2') as sv:
        sv.create_dataset('bar/bon/1/data/axes/date', data=data2)
        assert_equal(sv['bar/baz'][:], data1)
        assert_equal(sv['bar/bon/1/data/axes/date'][:], data2)

    version1 = vfile['1']
    version2 = vfile['2']
    assert_equal(version1['bar/baz'][:], data1)
    assert_equal(version2['bar/baz'][:], data1)
    assert 'bar/bon/1/data/axes/date' not in version1
    assert_equal(version2['bar/bon/1/data/axes/date'][:], data2)


def test_fillvalue(vfile):
    # Based on test_resize(), but only the resize largers that use the fill
    # value
    fillvalue = 5.0

    no_offset_data = np.ones((2*DEFAULT_CHUNK_SIZE,))

    offset_data = np.concatenate((np.ones((DEFAULT_CHUNK_SIZE,)),
                                  np.ones((2,))))

    with vfile.stage_version('version1') as group:
        group.create_dataset('no_offset', data=no_offset_data, fillvalue=fillvalue)
        group.create_dataset('offset', data=offset_data, fillvalue=fillvalue)

    group = vfile['version1']
    assert group['no_offset'].shape == (2*DEFAULT_CHUNK_SIZE,)
    assert group['offset'].shape == (DEFAULT_CHUNK_SIZE + 2,)
    assert_equal(group['no_offset'][:2*DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(group['offset'][:DEFAULT_CHUNK_SIZE + 2], 1.0)

    # Resize larger, chunk multiple
    with vfile.stage_version('larger_chunk_multiple') as group:
        group['no_offset'].resize((3*DEFAULT_CHUNK_SIZE,))
        group['offset'].resize((3*DEFAULT_CHUNK_SIZE,))

    group = vfile['larger_chunk_multiple']
    assert group['no_offset'].shape == (3*DEFAULT_CHUNK_SIZE,)
    assert group['offset'].shape == (3*DEFAULT_CHUNK_SIZE,)
    assert_equal(group['no_offset'][:2*DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(group['no_offset'][2*DEFAULT_CHUNK_SIZE:], fillvalue)
    assert_equal(group['offset'][:DEFAULT_CHUNK_SIZE + 2], 1.0)
    assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 2:], fillvalue)

    # Resize larger, non-chunk multiple
    with vfile.stage_version('larger_chunk_non_multiple', 'version1') as group:
        group['no_offset'].resize((3*DEFAULT_CHUNK_SIZE + 2,))
        group['offset'].resize((3*DEFAULT_CHUNK_SIZE + 2,))

    group = vfile['larger_chunk_non_multiple']
    assert group['no_offset'].shape == (3*DEFAULT_CHUNK_SIZE + 2,)
    assert group['offset'].shape == (3*DEFAULT_CHUNK_SIZE + 2,)
    assert_equal(group['no_offset'][:2*DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(group['no_offset'][2*DEFAULT_CHUNK_SIZE:], fillvalue)
    assert_equal(group['offset'][:DEFAULT_CHUNK_SIZE + 2], 1.0)
    assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 2:], fillvalue)

    # Resize after creation
    with vfile.stage_version('version2', 'version1') as group:
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

    group = vfile['version2']
    assert group['no_offset'].shape == (3*DEFAULT_CHUNK_SIZE,)
    assert group['offset'].shape == (3*DEFAULT_CHUNK_SIZE,)
    assert_equal(group['no_offset'][:2*DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(group['no_offset'][2*DEFAULT_CHUNK_SIZE:], fillvalue)
    assert_equal(group['offset'][:DEFAULT_CHUNK_SIZE + 1], 1.0)
    assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 1], 2.0)
    assert_equal(group['offset'][DEFAULT_CHUNK_SIZE + 2:], fillvalue)

    # Resize after calling create_dataset, larger
    with vfile.stage_version('resize_after_create_larger', '') as group:
        group.create_dataset('data', data=offset_data,
                             fillvalue=fillvalue)
        group['data'].resize((DEFAULT_CHUNK_SIZE + 4,))

        assert group['data'].shape == (DEFAULT_CHUNK_SIZE + 4,)
        assert_equal(group['data'][:DEFAULT_CHUNK_SIZE + 2], 1.0)
        assert_equal(group['data'][DEFAULT_CHUNK_SIZE + 2:], fillvalue)


def test_multidimsional(vfile):

    data = np.ones((2*DEFAULT_CHUNK_SIZE, 5))

    with vfile.stage_version('version1') as g:
        g.create_dataset('test_data', data=data,
                         chunks=(DEFAULT_CHUNK_SIZE, 2))
        assert_equal(g['test_data'][()], data)

    version1 = vfile['version1']
    assert_equal(version1['test_data'][()], data)

    data2 = data.copy()
    data2[0, 1] = 2

    with vfile.stage_version('version2') as g:
        g['test_data'][0, 1] = 2
        assert g['test_data'][0, 1] == 2
        assert_equal(g['test_data'][()], data2)

    version2 = vfile['version2']
    assert version2['test_data'][0, 1] == 2
    assert_equal(version2['test_data'][()], data2)

    data3 = data.copy()
    data3[0:1] = 3

    with vfile.stage_version('version3', 'version1') as g:
        g['test_data'][0:1] = 3
        assert_equal(g['test_data'][0:1], 3)
        assert_equal(g['test_data'][()], data3)

    version3 = vfile['version3']
    assert_equal(version3['test_data'][0:1], 3)
    assert_equal(version3['test_data'][()], data3)


def test_group_chunks_compression(vfile):
    # Chunks and compression are similar, so test them both at the same time.
    data = np.ones((2*DEFAULT_CHUNK_SIZE, 5))

    with vfile.stage_version('version1') as g:
        g2 = g.create_group('group')
        g2.create_dataset('test_data', data=data,
                          chunks=(DEFAULT_CHUNK_SIZE, 2),
                          compression='gzip',
                          compression_opts=3)
        assert_equal(g2['test_data'][()], data)
        assert_equal(g['group/test_data'][()], data)
        assert_equal(g['group']['test_data'][()], data)

    version1 = vfile['version1']
    assert_equal(version1['group']['test_data'][()], data)
    assert_equal(version1['group/test_data'][()], data)

    raw_data = vfile.f['/_version_data/group/test_data/raw_data']
    assert raw_data.compression == 'gzip'
    assert raw_data.compression_opts == 3


def test_closes(vfile):
    data = np.ones((DEFAULT_CHUNK_SIZE,))

    with vfile.stage_version('version1') as g:
        g.create_dataset('test_data', data=data)
    assert vfile._closed is False
    assert vfile.closed is False

    version_data = vfile._version_data
    versions = vfile._versions

    h5pyfile = vfile.f
    vfile.close()

    assert vfile._closed is True
    assert vfile.closed is True
    raises(AttributeError, lambda: vfile.f)
    raises(AttributeError, lambda: vfile._version_data)
    raises(AttributeError, lambda: vfile._versions)
    assert repr(vfile) == "<Closed VersionedHDF5File>"

    reopened_file = VersionedHDF5File(h5pyfile)
    assert list(reopened_file['version1']) == ['test_data']
    assert_equal(reopened_file['version1']['test_data'][()], data)

    assert reopened_file._version_data == version_data
    assert reopened_file._versions == versions

    # Close the underlying file
    h5pyfile.close()
    assert vfile.closed is True
    raises(ValueError, lambda: vfile['version1'])
    raises(ValueError, lambda: vfile['version2'])
    assert repr(vfile) == "<Closed VersionedHDF5File>"

def test_scalar_dataset():
    for data1, data2 in [
            (b'baz', b'foo'),
            (np.asarray('baz', dtype='S'), np.asarray('foo', dtype='S')),
            (1.5, 2.3),
            (1, 0)
    ]:

        dt = np.asarray(data1).dtype
        with setup_vfile() as f:
            file = VersionedHDF5File(f)
            with file.stage_version('v1') as group:
                group['scalar_ds'] = data1

            v1_ds = file['v1']['scalar_ds']
            assert v1_ds[()] == data1
            assert v1_ds.shape == ()
            assert v1_ds.dtype == dt

            with file.stage_version('v2') as group:
                group['scalar_ds'] = data2

            v2_ds = file['v2']['scalar_ds']
            assert v2_ds[()] == data2
            assert v2_ds.shape == ()
            assert v2_ds.dtype == dt

        file.close()


def test_store_binary_as_void(vfile):
    with vfile.stage_version('version1') as sv:
        sv['test_store_binary_data'] = [np.void(b'1111')]

    version1 = vfile['version1']
    assert_equal(version1['test_store_binary_data'][0], np.void(b'1111'))

    with vfile.stage_version('version2') as sv:
        sv['test_store_binary_data'][:] = [np.void(b'1234567890')]

    version2 = vfile['version2']
    assert_equal(version2['test_store_binary_data'][0], np.void(b'1234'))


def test_check_committed(vfile):

    data = np.ones((DEFAULT_CHUNK_SIZE,))

    with vfile.stage_version('version1') as g:
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
        with vfile.stage_version('version2'):
            assert isinstance(g['test_data'], InMemoryArrayDataset)
            g['test_data'][0] = 1

    with raises(ValueError, match="committed"):
        with vfile.stage_version('version2'):
            assert isinstance(g['test_data'], InMemoryArrayDataset)
            g['test_data'].resize((100,))

    with vfile.stage_version('version2') as g2:
        pass

    # Incorrectly uses g from the previous version (InMemoryDataset)
    with raises(ValueError, match="committed"):
        with vfile.stage_version('version3'):
            assert isinstance(g2['test_data'], DatasetWrapper)
            assert isinstance(g2['test_data'].dataset, InMemoryDataset)
            g2['test_data'][0] = 1

    with raises(ValueError, match="committed"):
        with vfile.stage_version('version3'):
            assert isinstance(g2['test_data'], DatasetWrapper)
            assert isinstance(g2['test_data'].dataset, InMemoryDataset)
            g2['test_data'].resize((100,))

    assert repr(g) == '<Committed InMemoryGroup "/_version_data/versions/version1">'


def test_set_chunks_nested(vfile):
    with vfile.stage_version('0') as sv:
        data_group = sv.create_group('data')
        data_group.create_dataset('bar', data=np.arange(4))

    with vfile.stage_version('1') as sv:
        data_group = sv['data']
        data_group.create_dataset('props/1/bar', data=np.arange(0, 4, 2))


def test_InMemoryArrayDataset_chunks(vfile):
    with vfile.stage_version('0') as sv:
        data_group = sv.create_group('data')
        data_group.create_dataset('g/bar', data=np.arange(4),
                                  chunks=(100,), compression='gzip', compression_opts=3)
        assert isinstance(data_group['g/bar'], InMemoryArrayDataset)
        assert data_group['g/bar'].chunks == (100,)
        assert data_group['g/bar'].compression == 'gzip'
        assert data_group['g/bar'].compression_opts == 3


def test_string_dtypes():

    # Make sure the fillvalue logic works correctly for custom h5py string
    # dtypes.

    # h5py 3 changed variable-length UTF-8 strings to be read in as bytes
    # instead of str. See
    # https://docs.h5py.org/en/stable/whatsnew/3.0.html#breaking-changes-deprecations
    h5py_str_type = bytes if h5py.__version__.startswith('3') else str

    for typ, dt in [
            (h5py_str_type, h5py.string_dtype('utf-8')),
            (bytes, h5py.string_dtype('ascii')),
            # h5py uses bytes here
            (bytes, h5py.string_dtype('utf-8', length=20)),
            (bytes, h5py.string_dtype('ascii', length=20)),
            ]:

        if typ == str:
            data = np.full(10, 'hello world', dtype=dt)
        else:
            data = np.full(10, b'hello world', dtype=dt)

        with setup_vfile() as f:
            file = VersionedHDF5File(f)
            with file.stage_version('0') as sv:
                sv.create_dataset("name", shape=(10,), dtype=dt, data=data)
                assert isinstance(sv['name'], InMemoryArrayDataset)
                sv['name'].resize((11,))

            assert file['0']['name'].dtype == dt
            assert_equal(file['0']['name'][:10], data)
            assert file['0']['name'][10] == typ(), dt.metadata

            with file.stage_version('1') as sv:
                assert isinstance(sv['name'], DatasetWrapper)
                assert isinstance(sv['name'].dataset, InMemoryDataset)
                sv['name'].resize((12,))

            assert file['1']['name'].dtype == dt
            assert_equal(file['1']['name'][:10], data, str(dt.metadata))
            assert file['1']['name'][10] == typ(), dt.metadata
            assert file['1']['name'][11] == typ(), dt.metadata

            # Make sure we are matching the pure h5py behavior
            f.create_dataset('name', shape=(10,), dtype=dt, data=data,
                             chunks=(10,), maxshape=(None,))
            f['name'].resize((11,))
            assert f['name'].dtype == dt
            assert_equal(f['name'][:10], data)
            assert f['name'][10] == typ(), dt.metadata

def test_empty(vfile):
    with vfile.stage_version('version1') as g:
        g['data'] = np.arange(10)
        g.create_dataset('data2', data=np.empty((1, 0, 2)), chunks=(5, 5, 5))
        assert_equal(g['data2'][()], np.empty((1, 0, 2)))
    assert_equal(vfile['version1']['data2'][()], np.empty((1, 0, 2)))

    with vfile.stage_version('version2') as g:
        g['data'].resize((0,))
        assert_equal(g['data'][()], np.empty((0,)))

    assert_equal(vfile['version2']['data'][()], np.empty((0,)))
    assert_equal(vfile['version2']['data2'][()], np.empty((1, 0, 2)))


def test_read_only():
    with setup_vfile('test.hdf5') as f:
        file = VersionedHDF5File(f)
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        with file.stage_version('version1', timestamp=timestamp) as g:
            g['data'] = [0, 1, 2]

        with raises(ValueError):
            g['data'][0] = 1
        with raises(ValueError):
            g['data2'] = [1, 2, 3]

        with raises(ValueError):
            file['version1']['data'][0] = 1
        with raises(ValueError):
            file['version1']['data2'] = [1, 2, 3]

        with raises(ValueError):
            file[timestamp]['data'][0] = 1
        with raises(ValueError):
            file[timestamp]['data2'] = [1, 2, 3]

    with h5py.File('test.hdf5', 'r+') as f:
        file = VersionedHDF5File(f)

        with raises(ValueError):
            file['version1']['data'][0] = 1
        with raises(ValueError):
            file['version1']['data2'] = [1, 2, 3]

        with raises(ValueError):
            file[timestamp]['data'][0] = 1
        with raises(ValueError):
            file[timestamp]['data2'] = [1, 2, 3]

def test_delete_datasets(vfile):
    data1 = np.arange(10)
    data2 = np.zeros(20, dtype=int)
    with vfile.stage_version('version1') as g:
        g['data'] = data1
        g.create_group('group1/group2')
        g['group1']['group2']['data1'] = data1

    with vfile.stage_version('del_data') as g:
        del g['data']

    with vfile.stage_version('del_data1', 'version1') as g:
        del g['group1/group2/data1']

    with vfile.stage_version('del_group2', 'version1') as g:
        del g['group1/group2']

    with vfile.stage_version('del_group1', 'version1') as g:
        del g['group1/']

    with vfile.stage_version('version2', 'del_data') as g:
        g['data'] = np.zeros(20, dtype=int)

    with vfile.stage_version('version3', 'del_data1') as g:
        g['group1/group2/data1'] = data2

    with vfile.stage_version('version4', 'del_group2') as g:
        g.create_group('group1/group2')
        g['group1/group2/data1'] = data2

    with vfile.stage_version('version5', 'del_group1') as g:
        g.create_group('group1/group2')
        g['group1/group2/data1'] = data2

    assert set(vfile['version1']) == {'group1', 'data'}
    assert list(vfile['version1']['group1']) == ['group2']
    assert list(vfile['version1']['group1']['group2']) == ['data1']
    assert_equal(vfile['version1']['data'][:], data1)
    assert_equal(vfile['version1']['group1/group2/data1'][:], data1)

    assert list(vfile['del_data']) == ['group1']
    assert list(vfile['del_data']['group1']) == ['group2']
    assert list(vfile['del_data']['group1']['group2']) == ['data1']
    assert_equal(vfile['del_data']['group1/group2/data1'][:], data1)

    assert set(vfile['del_data1']) == {'group1', 'data'}
    assert list(vfile['del_data1']['group1']) == ['group2']
    assert list(vfile['del_data1']['group1']['group2']) == []
    assert_equal(vfile['del_data1']['data'][:], data1)

    assert set(vfile['del_group2']) == {'group1', 'data'}
    assert list(vfile['del_group2']['group1']) == []
    assert_equal(vfile['del_group2']['data'][:], data1)

    assert list(vfile['del_group1']) == ['data']
    assert_equal(vfile['del_group1']['data'][:], data1)

    assert set(vfile['version2']) == {'group1', 'data'}
    assert list(vfile['version2']['group1']) == ['group2']
    assert list(vfile['version2']['group1']['group2']) == ['data1']
    assert_equal(vfile['version2']['data'][:], data2)
    assert_equal(vfile['version2']['group1/group2/data1'][:], data1)

    assert set(vfile['version3']) == {'group1', 'data'}
    assert list(vfile['version3']['group1']) == ['group2']
    assert list(vfile['version3']['group1']['group2']) == ['data1']
    assert_equal(vfile['version3']['data'][:], data1)
    assert_equal(vfile['version3']['group1/group2/data1'][:], data2)

    assert set(vfile['version4']) == {'group1', 'data'}
    assert list(vfile['version4']['group1']) == ['group2']
    assert list(vfile['version4']['group1']['group2']) == ['data1']
    assert_equal(vfile['version4']['data'][:], data1)
    assert_equal(vfile['version4']['group1/group2/data1'][:], data2)

    assert set(vfile['version5']) == {'group1', 'data'}
    assert list(vfile['version5']['group1']) == ['group2']
    assert list(vfile['version5']['group1']['group2']) == ['data1']
    assert_equal(vfile['version5']['data'][:], data1)
    assert_equal(vfile['version5']['group1/group2/data1'][:], data2)

def test_auto_create_group(vfile):
    with vfile.stage_version('version1') as g:
        g['a/b/c'] = [0, 1, 2]
        assert_equal(g['a']['b']['c'][:], [0, 1, 2])

    assert_equal(vfile['version1']['a']['b']['c'][:], [0, 1, 2])

def test_scalar():
    with setup_vfile('test.hdf5') as f:
        vfile = VersionedHDF5File(f)
        with vfile.stage_version('version1') as g:
            dtype = h5py.special_dtype(vlen=bytes)
            g.create_dataset('bar', data=np.array(['aaa'], dtype='O'), dtype=dtype)

    with h5py.File('test.hdf5', 'r+') as f:
        vfile = VersionedHDF5File(f)
        assert isinstance(vfile['version1']['bar'], DatasetWrapper)
        assert isinstance(vfile['version1']['bar'].dataset, InMemoryDataset)
        # Should return a scalar, not a shape () array
        assert isinstance(vfile['version1']['bar'][0], bytes)

    with h5py.File('test.hdf5', 'r') as f:
        vfile = VersionedHDF5File(f)
        assert isinstance(vfile['version1']['bar'], h5py.Dataset)
        # Should return a scalar, not a shape () array
        assert isinstance(vfile['version1']['bar'][0], bytes)

def test_sparse(vfile):
    with vfile.stage_version('version1') as g:
        g.create_dataset('test_data', shape=(10_000, 10_000), dtype=np.dtype('int64'), data=None,
              chunks=(100, 100), fillvalue=1)
        assert isinstance(g['test_data'], InMemorySparseDataset)
        assert g['test_data'][0, 0] == 1
        assert g['test_data'][0, 1] == 1
        assert g['test_data'][200, 1] == 1

        g['test_data'][0, 0] = 2
        assert g['test_data'][0, 0] == 2
        assert g['test_data'][0, 1] == 1
        assert g['test_data'][200, 1] == 1


    with vfile.stage_version('version2') as g:
        assert isinstance(g['test_data'], DatasetWrapper)
        assert isinstance(g['test_data'].dataset, InMemoryDataset)
        assert g['test_data'][0, 0] == 2
        assert g['test_data'][0, 1] == 1
        assert g['test_data'][200, 1] == 1

        g['test_data'][200, 1] = 3

        assert g['test_data'][0, 0] == 2
        assert g['test_data'][0, 1] == 1
        assert g['test_data'][200, 1] == 3

    assert vfile['version1']['test_data'][0, 0] == 2
    assert vfile['version1']['test_data'][0, 1] == 1
    assert vfile['version1']['test_data'][200, 1] == 1

    assert vfile['version2']['test_data'][0, 0] == 2
    assert vfile['version2']['test_data'][0, 1] == 1
    assert vfile['version2']['test_data'][200, 1] == 3

def test_sparse_empty(vfile):
    with vfile.stage_version('version1') as g:
        g.create_dataset('test_data', shape=(10_000, 10_000), dtype=np.dtype('int64'), data=None,
              chunks=(100, 100), fillvalue=1)
        # Don't read or write any data from the sparse dataset

    assert vfile['version1']['test_data'][0, 0] == 1
    assert vfile['version1']['test_data'][0, 1] == 1
    assert vfile['version1']['test_data'][200, 1] == 1

    with vfile.stage_version('version2') as g:
        assert isinstance(g['test_data'], DatasetWrapper)
        assert isinstance(g['test_data'].dataset, InMemoryDataset)
        assert g['test_data'][0, 0] == 1
        assert g['test_data'][0, 1] == 1
        assert g['test_data'][200, 1] == 1

        g['test_data'][0, 0] = 2
        g['test_data'][200, 1] = 2

        assert g['test_data'][0, 0] == 2
        assert g['test_data'][0, 1] == 1
        assert g['test_data'][200, 1] == 2

    assert vfile['version1']['test_data'][0, 0] == 1
    assert vfile['version1']['test_data'][0, 1] == 1
    assert vfile['version1']['test_data'][200, 1] == 1

    assert vfile['version2']['test_data'][0, 0] == 2
    assert vfile['version2']['test_data'][0, 1] == 1
    assert vfile['version2']['test_data'][200, 1] == 2

def test_sparse_large(vfile):
    # This is currently inefficient in terms of time, but test that it isn't
    # inefficient in terms of memory.
    with vfile.stage_version('version1') as g:
        # test_data would be 100GB if stored entirely in memory. We use a huge
        # chunk size to avoid taking too long with the current code that loops
        # over all chunk indices.
        g.create_dataset('test_data', shape=(100_000_000_000,), data=None,
                         chunks=(10_000_000,), fillvalue=0.)
        assert isinstance(g['test_data'], InMemorySparseDataset)
        assert g['test_data'][0] == 0
        assert g['test_data'][1] == 0
        assert g['test_data'][20_000_000] == 0

        g['test_data'][0] = 1
        assert g['test_data'][0] == 1
        assert g['test_data'][1] == 0
        assert g['test_data'][20_000_000] == 0


    with vfile.stage_version('version2') as g:
        assert isinstance(g['test_data'], DatasetWrapper)
        assert isinstance(g['test_data'].dataset, InMemoryDataset)
        assert g['test_data'][0] == 1
        assert g['test_data'][1] == 0
        assert g['test_data'][20_000_000] == 0

        g['test_data'][20_000_000] = 2

        assert g['test_data'][0] == 1
        assert g['test_data'][1] == 0
        assert g['test_data'][20_000_000] == 2

    assert vfile['version1']['test_data'][0] == 1
    assert vfile['version1']['test_data'][1] == 0
    assert vfile['version1']['test_data'][20_000_000] == 0

    assert vfile['version2']['test_data'][0] == 1
    assert vfile['version2']['test_data'][1] == 0
    assert vfile['version2']['test_data'][20_000_000] == 2

def test_no_recursive_version_group_access(vfile):
    timestamp1 = datetime.datetime.now(datetime.timezone.utc)
    with vfile.stage_version('version1', timestamp=timestamp1) as g:
        g.create_dataset('test', data=[1, 2, 3])

    timestamp2 = datetime.datetime.now(datetime.timezone.utc)
    minute = datetime.timedelta(minutes=1)
    with vfile.stage_version('version2', timestamp=timestamp2) as g:
        vfile['version1'] # Doesn't raise
        raises(ValueError, lambda: vfile['version2'])

        vfile[timestamp1] # Doesn't raise
        # Without +minute, it will pick the previous version, as the
        # uncommitted group only has a placeholder timestamp, which will be
        # after timestamp2. Since this isn't supposed to work in the first
        # place, this isn't a big deal.
        raises(ValueError, lambda: vfile[timestamp2+minute])

def test_empty_dataset_str_dtype(vfile):
    # Issue #161. Make sure the dtype is maintained correctly for empty
    # datasets with custom string dtypes.
    with vfile.stage_version('version1') as g:
        g.create_dataset('bar', data=np.array(['a', 'b', 'c'], dtype='S5'), dtype=np.dtype('S5'))
        g['bar'].resize((0,))
    with vfile.stage_version('version2') as g:
        g['bar'].resize((3,))
        g['bar'][:] = np.array(['a', 'b', 'c'], dtype='S5')

def test_datasetwrapper(vfile):
    with vfile.stage_version('r0') as sv:
        sv.create_dataset('bar', data=[1, 2, 3], chunks=(2,))
        sv['bar'].attrs['key'] = 0
        assert isinstance(sv['bar'], InMemoryArrayDataset)
        assert dict(sv['bar'].attrs) == {'key': 0}
        assert sv['bar'].chunks == (2,)

    with vfile.stage_version('r1') as sv:
        assert isinstance(sv['bar'], DatasetWrapper)
        assert isinstance(sv['bar'].dataset, InMemoryDataset)
        assert sv['bar'].attrs['key'] == 0
        sv['bar'].attrs['key'] = 1
        assert sv['bar'].attrs['key'] == 1
        assert sv['bar'].chunks == (2,)

        sv['bar'][:] = [4, 5, 6]
        assert isinstance(sv['bar'], DatasetWrapper)
        assert isinstance(sv['bar'].dataset, InMemoryArrayDataset)
        assert sv['bar'].attrs['key'] == 1
        assert sv['bar'].chunks == (2,)

def test_mask_reading(tmp_path):
    # Reading a virtual dataset with a mask does not work in HDF5, so make
    # sure it still works for versioned datasets.
    file_name = os.path.join(tmp_path, 'file.hdf5')
    mask = np.array([True, True, False], dtype='bool')

    with h5py.File(file_name, 'w') as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version('r0') as sv:
            sv.create_dataset('bar', data=[1, 2, 3], chunks=(2,))
            b = sv['bar'][mask]
            assert_equal(b, [1, 2])

        b = vf['r0']['bar'][mask]
        assert_equal(b, [1, 2])

    with h5py.File(file_name, 'r+') as f:
        vf = VersionedHDF5File(f)
        sv = vf['r0']
        b = sv['bar'][mask]
        assert_equal(b, [1, 2])

# This fails prior to h5py 3.3 because read-only files return the virtual
# dataset directly, but h5py <3.3 does not support mask indices on virtual
# datasets.
@mark.xfail(h5py.__version__[0] == '2'
            or h5py.__version__[0] == '3' and int(h5py.__version__[2]) < 3,
            reason='h5py 2 does not support masks on virtual datasets')
def test_mask_reading_read_only(tmp_path):
    # Reading a virtual dataset with a mask does not work in HDF5, so make
    # sure it still works for versioned datasets.
    file_name = os.path.join(tmp_path, 'file.hdf5')
    mask = np.array([True, True, False], dtype='bool')

    with h5py.File(file_name, 'w') as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version('r0') as sv:
            sv.create_dataset('bar', data=[1, 2, 3], chunks=(2,))
            b = sv['bar'][mask]
            assert_equal(b, [1, 2])

        b = vf['r0']['bar'][mask]
        assert_equal(b, [1, 2])

    with h5py.File(file_name, 'r') as f:
        vf = VersionedHDF5File(f)
        sv = vf['r0']
        b = sv['bar'][mask]
        assert_equal(b, [1, 2])

def test_read_only_no_wrappers():
    # Read-only files should not use the wrapper classes
    with setup_vfile('test.hdf5') as f:
        vfile = VersionedHDF5File(f)
        with vfile.stage_version('version1') as g:
            g.create_dataset('bar', data=np.array([0, 1, 2]))

    with h5py.File('test.hdf5', 'r+') as f:
        vfile = VersionedHDF5File(f)
        assert isinstance(vfile['version1'], InMemoryGroup)
        assert isinstance(vfile['version1']['bar'], DatasetWrapper)
        assert isinstance(vfile['version1']['bar'].dataset, InMemoryDataset)

    with h5py.File('test.hdf5', 'r') as f:
        vfile = VersionedHDF5File(f)
        assert isinstance(vfile['version1'], h5py.Group)
        assert isinstance(vfile['version1']['bar'], h5py.Dataset)


def test_stage_version_log_stats(tmp_path, caplog):
    """Test that stage_version logs stats after writing data."""
    caplog.set_level(logging.DEBUG)
    file_name = os.path.join(tmp_path, 'file.hdf5')

    with h5py.File(file_name, 'w') as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version('r0') as sv:
            bar_shape_r0 = (2, 15220, 2)
            bar_chunks_r0 = (300, 100, 2)
            baz_shape_r0 = (1, 10, 2)
            baz_chunks_r0 = (600, 2, 4)

            sv.create_dataset(
                'bar',
                bar_shape_r0,
                chunks=bar_chunks_r0,
                data=np.full((2, 15220, 2), 0)
            )
            sv.create_dataset(
                'baz',
                baz_shape_r0,
                chunks=baz_chunks_r0,
                data=np.full((1, 10, 2), 0)
            )

        assert caplog.records
        assert str(bar_shape_r0) in caplog.records[-1].getMessage()
        assert str(bar_chunks_r0) in caplog.records[-1].getMessage()
        assert str(baz_shape_r0) in caplog.records[-1].getMessage()
        assert str(baz_chunks_r0) in caplog.records[-1].getMessage()

        with vf.stage_version('r1') as sv:
            bar_shape_r1 = (3, 15222, 2)
            baz_shape_r1 = (1, (4) * 10, 2)

            bar = sv['bar']
            bar.resize(bar_shape_r1)
            baz = sv['baz']
            baz.resize(baz_shape_r1)
            baz[:, -10:, :] = np.full((1, 10, 2), 3)

        assert str(bar_shape_r1) in caplog.records[-1].getMessage()
        assert str(baz_shape_r1) in caplog.records[-1].getMessage()
