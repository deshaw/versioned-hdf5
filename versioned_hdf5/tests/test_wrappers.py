import numpy as np
from numpy.testing import assert_equal

from ..wrappers import InMemoryArrayDataset, InMemorySparseDataset

def test_InMemoryArrayDataset():
    a = np.arange(100,).reshape((50, 2))
    dataset = InMemoryArrayDataset('data', a)
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

def test_InMemoryArrayDataset_resize():
    a = np.arange(100)
    dataset = InMemoryArrayDataset('data', a)
    dataset.resize((110,))

    assert len(dataset) == 110
    assert_equal(dataset[:100], dataset.array[:100])
    assert_equal(dataset[:100], a)
    assert_equal(dataset[100:], dataset.array[100:])
    assert_equal(dataset[100:], 0)
    assert dataset.shape == dataset.array.shape == (110,)

    a = np.arange(100)
    dataset = InMemoryArrayDataset('data', a)
    dataset.resize((90,))

    assert len(dataset) == 90
    assert_equal(dataset, dataset.array)
    assert_equal(dataset, np.arange(90))
    assert dataset.shape == dataset.array.shape == (90,)

def test_InMemorySparseDataset():
    d = InMemorySparseDataset('data', (1000,), np.float64, fillvalue=1.0)
    assert d.shape == (1000,)
    assert d.name == 'data'
    assert d.dtype == np.float64
    assert d.fillvalue == np.float64(1.0)

def test_InMemorySparseDataset_getitem():
    d = InMemorySparseDataset('data', (1000,), np.float64, fillvalue=1.0)
    assert_equal(d[0], 1.0)
    assert_equal(d[:], np.ones((1000,)))
    assert_equal(d[10:20], np.ones((10,)))
