import itertools

import h5py
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal

from versioned_hdf5 import VersionedHDF5File
from versioned_hdf5.typing_ import ArrayProtocol
from versioned_hdf5.wrappers import (
    DatasetWrapper,
    InMemoryArrayDataset,
    InMemoryDataset,
    InMemoryGroup,
    InMemorySparseDataset,
)


@pytest.fixture
def premade_group(h5file):
    group = h5file.create_group("group")
    return InMemoryGroup(group.id)


def test_InMemoryArrayDataset(h5file):
    group = h5file.create_group("group")
    parent = InMemoryGroup(group.id)
    a = np.arange(100).reshape((50, 2))
    dataset = InMemoryArrayDataset("data", a, parent=parent)
    assert dataset.name == "data"
    assert_equal(dataset._buffer, a)
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
    assert isinstance(dataset[30, 0], np.generic)
    dataset[30, 0] = 1000
    assert dataset[30, 0] == 1000
    assert dataset._buffer[30, 0] == 1000
    assert (dataset[30, :] == a[30, :]).all()
    assert isinstance(dataset[30, :], np.ndarray)

    assert dataset.size == 100


def test_InMemoryArrayDataset_resize(h5file):
    group = h5file.create_group("group")
    parent = InMemoryGroup(group.id)
    a = np.arange(100)
    dataset = DatasetWrapper(  # Can't enlarge an InMemoryArrayDataset directly
        InMemoryArrayDataset("data", a, parent=parent, chunks=(50,))
    )
    dataset.resize((110,))

    assert len(dataset) == 110
    assert_equal(dataset[:100], dataset._buffer[:100])
    assert_equal(dataset[:100], a)
    assert_equal(dataset[100:], dataset._buffer[100:])
    assert_equal(dataset[100:], 0)
    assert dataset.shape == dataset._buffer.shape == (110,)

    a = np.arange(100)
    dataset = InMemoryArrayDataset("data", a, parent=parent)
    dataset.resize((90,))

    assert len(dataset) == 90
    assert_equal(dataset, dataset._buffer)
    assert_equal(dataset, np.arange(90))
    assert dataset.shape == dataset._buffer.shape == (90,)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(data=[0], dtype="i2"),
        dict(data=[0], dtype=np.int16),
        dict(data=np.asarray([0], dtype=np.int16)),
        dict(data=np.asarray([0], dtype=np.int32), dtype=np.int16),
        dict(shape=(1,), dtype=np.int16),
    ],
)
def test_InMemoryArrayDataset_dtype(vfile, kwargs):
    with vfile.stage_version("r0") as group:
        ds = group.create_dataset("x", **kwargs)
        assert isinstance(ds.dtype, np.dtype)
        assert ds.dtype == np.int16
        assert ds[:].dtype == np.int16
        assert np.asarray(ds).dtype == np.int16


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
    ("oldshape", "newshape"),
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

    dataset = DatasetWrapper(  # Can't enlarge an InMemoryArrayDataset directly
        InMemoryArrayDataset("data", a, parent=parent, fillvalue=-1, chunks=(7, 4, 11))
    )
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


def test_readonly_data(vfile):
    """Read-only data is copied upon creation of a new version."""
    data = np.array([0, 1, 2])
    data.flags.writeable = False
    with vfile.stage_version("r0") as group:
        dset = group.create_dataset("x", data=data)
        dset[0] = 3

    assert_array_equal(data, np.array([0, 1, 2]))
    assert_array_equal(vfile[None]["x"], np.array([3, 1, 2]))


@pytest.mark.parametrize("dtype", ["i1", "i2"])  # no-op view vs. actual conversion
def test_astype_dense(vfile, dtype):
    with vfile.stage_version("r0") as group:
        dset = group.create_dataset("x", data=np.array([0, 1, 2], dtype="i1"))
        assert isinstance(dset.dataset, InMemoryArrayDataset)

        a = dset.astype(dtype)
        assert isinstance(a, ArrayProtocol)
        assert a.dtype == dtype
        assert dset.dtype == "i1"
        with pytest.raises(ValueError, match="read-only"):
            a[0] = 123
        assert_array_equal(a, np.array([0, 1, 2], dtype=dtype), strict=True)
        # The read-only flag has only been applied to the view
        dset[0] = 3

    with vfile.stage_version("r1") as group:
        dset = group["x"]
        assert isinstance(dset, DatasetWrapper)
        assert isinstance(dset.dataset, InMemoryDataset)
        assert dset.dtype == "i1"

        a = dset.astype(dtype)
        assert isinstance(a, ArrayProtocol)
        assert a.dtype == dtype
        assert dset.dtype == "i1"
        with pytest.raises(ValueError, match="read-only"):
            a[0] = 123
        assert_array_equal(a, np.array([3, 1, 2], dtype=dtype), strict=True)
        # The read-only flag has only been applied to the view
        dset[1] = 4

    dset = vfile[None]["x"]
    assert_array_equal(dset, np.array([3, 4, 2], dtype="i1"), strict=True)


@pytest.mark.parametrize("dtype", ["i1", "i2"])  # no-op view vs. actual conversion
def test_astype_sparse(vfile, dtype):
    with vfile.stage_version("r0") as group:
        dset = group.create_dataset("x", shape=(3,), chunks=(2,), dtype="i1")
        assert isinstance(dset, InMemorySparseDataset)
        dset[0] = 1

        a = dset.astype(dtype)
        assert isinstance(a, ArrayProtocol)
        assert a.dtype == dtype
        assert dset.dtype == "i1"
        with pytest.raises(ValueError, match="read-only"):
            a[0] = 123
        assert_array_equal(a, np.array([1, 0, 0], dtype=dtype), strict=True)
        # The read-only flag has only been applied to the view
        dset[1] = 2

    dset = vfile[None]["x"]
    assert_array_equal(dset, np.array([1, 2, 0], dtype="i1"), strict=True)
