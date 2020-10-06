import itertools

import numpy as np
from numpy.testing import assert_equal

from ..wrappers import InMemoryArrayDataset, InMemorySparseDataset, InMemoryGroup
import pytest


def test_InMemoryArrayDataset(h5file):
    group = h5file.create_group('group')
    parent = InMemoryGroup(group.id)
    a = np.arange(100,).reshape((50, 2))
    dataset = InMemoryArrayDataset('data', a, parent=parent)
    assert dataset.name == 'data'
    assert_equal(dataset.array, a)
    assert dataset.attrs == {}
    assert dataset.shape == a.shape
    assert dataset.dtype == a.dtype
    assert dataset.ndim == 2

    # __array__
    assert_equal(np.array(dataset), a)

    # Length of the first axis, matching h5py.Dataset
    assert len(dataset) == dataset.len() == 50

    # Test __iter__
    assert_equal(list(dataset), list(a))

    assert dataset[30, 0] == a[30, 0] == 60
    dataset[30, 0] = 1000
    assert dataset[30, 0] == 1000
    assert dataset.array[30, 0] == 1000

    assert dataset.size == 100


def test_InMemoryArrayDataset_resize(h5file):
    group = h5file.create_group('group')
    parent = InMemoryGroup(group.id)
    a = np.arange(100)
    dataset = InMemoryArrayDataset('data', a, parent=parent)
    dataset.resize((110,))

    assert len(dataset) == 110
    assert_equal(dataset[:100], dataset.array[:100])
    assert_equal(dataset[:100], a)
    assert_equal(dataset[100:], dataset.array[100:])
    assert_equal(dataset[100:], 0)
    assert dataset.shape == dataset.array.shape == (110,)

    a = np.arange(100)
    dataset = InMemoryArrayDataset('data', a, parent=parent)
    dataset.resize((90,))

    assert len(dataset) == 90
    assert_equal(dataset, dataset.array)
    assert_equal(dataset, np.arange(90))
    assert dataset.shape == dataset.array.shape == (90,)

def test_InMemorySparseDataset(h5file):
    group = h5file.create_group('group')
    parent = InMemoryGroup(group.id)
    d = InMemorySparseDataset('data', shape=(1000,), dtype=np.float64,
                              parent=parent, fillvalue=1.0)
    assert d.shape == (1000,)
    assert d.name == 'data'
    assert d.dtype == np.float64
    assert d.fillvalue == np.float64(1.0)

def test_InMemorySparseDataset_getitem(h5file):
    group = h5file.create_group('group')
    parent = InMemoryGroup(group.id)
    d = InMemorySparseDataset('data', shape=(1000,), dtype=np.float64,
                              parent=parent, fillvalue=1.0)
    assert_equal(d[0], 1.0)
    assert_equal(d[:], np.ones((1000,)))
    assert_equal(d[10:20], np.ones((10,)))

shapes = range(5, 25, 5) # 5, 10, 15, 20
chunks = (10, 10, 10)
@pytest.mark.parametrize('oldshape,newshape',
                         itertools.combinations_with_replacement(itertools.product(shapes, repeat=3), 2))
def test_InMemoryArrayDataset_resize_multidimension(oldshape, newshape, h5file):
    # Test semantics against raw HDF5
    a = np.arange(np.product(oldshape)).reshape(oldshape)

    group = h5file.create_group('group')
    parent = InMemoryGroup(group.id)

    dataset = InMemoryArrayDataset('data', a, parent=parent, fillvalue=-1)
    dataset.resize(newshape)

    h5file.create_dataset('data', data=a, fillvalue=-1,
                     chunks=chunks, maxshape=(None, None, None))
    h5file['data'].resize(newshape)
    assert_equal(dataset[()], h5file['data'][()], str(newshape))
