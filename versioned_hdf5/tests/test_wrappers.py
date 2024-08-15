import itertools

import h5py
import numpy as np
import pytest
from numpy.testing import assert_equal

from ..api import VersionedHDF5File
from ..wrappers import InMemoryArrayDataset, InMemoryGroup, InMemorySparseDataset


@pytest.fixture()
def premade_group(h5file):
    group = h5file.create_group("group")
    return InMemoryGroup(group.id)


def test_InMemoryArrayDataset(h5file):
    group = h5file.create_group("group")
    parent = InMemoryGroup(group.id)
    a = np.arange(
        100,
    ).reshape((50, 2))
    dataset = InMemoryArrayDataset("data", a, parent=parent)
    assert dataset.name == "data"
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
    group = h5file.create_group("group")
    parent = InMemoryGroup(group.id)
    a = np.arange(100)
    dataset = InMemoryArrayDataset("data", a, parent=parent)
    dataset.resize((110,))

    assert len(dataset) == 110
    assert_equal(dataset[:100], dataset.array[:100])
    assert_equal(dataset[:100], a)
    assert_equal(dataset[100:], dataset.array[100:])
    assert_equal(dataset[100:], 0)
    assert dataset.shape == dataset.array.shape == (110,)

    a = np.arange(100)
    dataset = InMemoryArrayDataset("data", a, parent=parent)
    dataset.resize((90,))

    assert len(dataset) == 90
    assert_equal(dataset, dataset.array)
    assert_equal(dataset, np.arange(90))
    assert dataset.shape == dataset.array.shape == (90,)


def test_InMemorySparseDataset(h5file):
    group = h5file.create_group("group")
    parent = InMemoryGroup(group.id)
    d = InMemorySparseDataset(
        "data", shape=(1000,), dtype=np.float64, parent=parent, fillvalue=1.0
    )
    assert d.shape == (1000,)
    assert d.name == "data"
    assert d.dtype == np.float64
    assert d.fillvalue == np.float64(1.0)


def test_InMemorySparseDataset_getitem(h5file):
    group = h5file.create_group("group")
    parent = InMemoryGroup(group.id)
    d = InMemorySparseDataset(
        "data", shape=(1000,), dtype=np.float64, parent=parent, fillvalue=1.0
    )
    assert_equal(d[0], 1.0)
    assert_equal(d[:], np.ones((1000,)))
    assert_equal(d[10:20], np.ones((10,)))


@pytest.mark.parametrize(
    "oldshape,newshape",
    itertools.combinations_with_replacement(
        itertools.product(range(5, 25, 5), repeat=3), 2
    ),
)
@pytest.mark.slow
def test_InMemoryArrayDataset_resize_multidimension(oldshape, newshape, h5file):
    # Test semantics against raw HDF5
    a = np.arange(np.prod(oldshape)).reshape(oldshape)

    group = h5file.create_group("group")
    parent = InMemoryGroup(group.id)

    dataset = InMemoryArrayDataset("data", a, parent=parent, fillvalue=-1)
    dataset.resize(newshape)

    h5file.create_dataset(
        "data", data=a, fillvalue=-1, chunks=(10, 10, 10), maxshape=(None, None, None)
    )
    h5file["data"].resize(newshape)
    assert_equal(dataset[()], h5file["data"][()], str(newshape))


def test_group_repr(premade_group):
    """Test that repr(InMemoryGroup) also shows reprs of child objects."""
    foo = premade_group.create_dataset(
        "foo",
        data=np.array([1, 2, 3, 4, 5, np.nan]),
    )
    bar = premade_group.create_dataset(
        "bar",
        data=np.array([1, 2, 3, 4, 5, np.nan]),
    )
    baz = premade_group.create_dataset(
        "baz",
        data=np.array([1, 2, 3, 4, 5, np.nan]),
    )

    result = repr(premade_group)
    assert repr(foo) in result
    assert repr(bar) in result
    assert repr(baz) in result


def test_committed_propagation():
    """Check that InMemoryGroup propagates the '_committed' state to child instances."""
    name = "testname"
    test_data = np.ones((10,))
    f = h5py.File("foo.h5", "w")
    vfile = VersionedHDF5File(f)

    # Commit some data to nested groups
    with vfile.stage_version("version1", "") as group:
        group.create_dataset(f"{name}/key", data=test_data)
        group.create_dataset(f"{name}/val", data=test_data)

    assert vfile["version1"]._committed
    assert vfile["version1"][name]._committed

    with vfile.stage_version("version2") as group:
        key_ds = group[f"{name}/key"]
        val_ds = group[f"{name}/val"]
        val_ds[0] = -1
        key_ds[0] = 0

    assert vfile["version2"]._committed
    assert vfile["version2"][name]._committed
