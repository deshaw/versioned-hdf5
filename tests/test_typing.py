import h5py
import numpy as np
import numpy.ma as ma
import pytest
from packaging.version import Version

from versioned_hdf5.h5py_compat import h5py_astype
from versioned_hdf5.staged_changes import StagedChangesArray
from versioned_hdf5.typing_ import ArrayProtocol, MutableArrayProtocol


class MinimalArray:
    """Minimal read-only NumPy array-like implementing the ArrayProtocol"""

    def __init__(self, arr):
        self._array = np.asarray(arr)
        self._array.flags.writeable = False

    @property
    def shape(self):
        return self._array.shape

    @property
    def size(self):
        return self._array.size

    @property
    def ndim(self):
        return self._array.ndim

    @property
    def dtype(self):
        return self._array.dtype

    def __getitem__(self, idx):
        return type(self)(self._array[idx])

    def __array__(self, dtype=None, copy=None):
        """Needed to qualify as an ArrayLike and to be accepted
        as the RHS of numpy.ndarray.__setitem__.
        """
        assert copy is not False
        return self._array


class MinimalMutableArray(MinimalArray):
    """Minimal writeable NumPy array-like implementing the ArrayProtocol"""

    def __setitem__(self, idx, val):
        self._array[idx] = val


def test_array_protocol():
    assert isinstance(MinimalArray(1), ArrayProtocol)
    assert not isinstance(MinimalArray(1), MutableArrayProtocol)
    assert isinstance(MinimalMutableArray(1), ArrayProtocol)
    assert isinstance(MinimalMutableArray(1), MutableArrayProtocol)
    assert isinstance(np.array(1), ArrayProtocol)
    assert isinstance(np.array(1), MutableArrayProtocol)
    assert isinstance(np.int64(1), ArrayProtocol)
    assert not isinstance(np.int64(1), MutableArrayProtocol)
    assert not isinstance(1, ArrayProtocol)
    assert not isinstance([1], ArrayProtocol)

    # numpy subclasses implement ArrayProtocol
    x = ma.masked_array([1, -1], mask=[0, 1], dtype="i2")
    assert isinstance(x, ArrayProtocol)
    assert isinstance(x, MutableArrayProtocol)


def test_array_protocol_h5_dataset(h5file):
    """Test that h5py.Dataset is a ArrayProtocol"""
    dset = h5file.create_dataset("x", shape=(10,), dtype="i2")
    assert isinstance(dset, ArrayProtocol)
    assert isinstance(dset, MutableArrayProtocol)


@pytest.mark.skipif(Version(h5py.__version__) < Version("3.13"), reason="h5py#2550")
def test_array_protocol_h5_astypeview(h5file):
    """Test that h5py AsTypeView is a ArrayProtocol"""
    dset = h5file.create_dataset("x", shape=(10,), dtype="i2")
    view = dset.astype("i4")
    assert isinstance(view, ArrayProtocol)
    assert not isinstance(view, MutableArrayProtocol)


def test_array_protocol_h5_astypeview_compat(h5file):
    """Test that h5py_astype() returns an ArrayProtocol, also on older h5py versions.
    TODO delete this test when dropping support for h5py <3.13.
    """
    dset = h5file.create_dataset("x", shape=(10,), dtype="i2")
    view = h5py_astype(dset, "i4")
    assert isinstance(view, ArrayProtocol)
    assert not isinstance(view, MutableArrayProtocol)


def array_protocol_staged_changes():
    arr = StagedChangesArray.full((3, 3), chunk_size=(3, 1), dtype="f4")
    assert isinstance(arr, ArrayProtocol)
    assert isinstance(arr, MutableArrayProtocol)
