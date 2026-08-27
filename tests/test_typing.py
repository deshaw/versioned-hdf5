import h5py
import numpy as np
import numpy.ma as ma
import pytest
from packaging.version import Version
from versioned_hdf5.slicetools import RawDataView

from versioned_hdf5.h5py_compat import h5py_astype
from versioned_hdf5.staged_changes import StagedChangesArray
from versioned_hdf5.typing_ import is_array_protocol


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

    def __init__(self, arr):
        self._array = np.asarray(arr)

    def __setitem__(self, idx, val):
        self._array[idx] = val


def test_array_protocol():
    assert is_array_protocol(MinimalArray(1))
    assert not is_array_protocol(MinimalArray(1), mutable=True)
    assert is_array_protocol(MinimalMutableArray(1))
    assert is_array_protocol(MinimalMutableArray(1), mutable=True)
    assert is_array_protocol(np.array(1))
    assert is_array_protocol(np.array(1), mutable=True)
    assert is_array_protocol(np.int64(1))
    assert not is_array_protocol(np.int64(1), mutable=True)
    assert not is_array_protocol(1)
    assert not is_array_protocol([1])

    # numpy subclasses implement ArrayProtocol
    x = ma.masked_array([1, -1], mask=[0, 1], dtype="i2")
    assert is_array_protocol(x)
    assert is_array_protocol(x, mutable=True)


def test_array_protocol_h5_dataset(h5file):
    """Test that h5py.Dataset is a ArrayProtocol"""
    dset = h5file.create_dataset("x", shape=(10,), dtype="i2")
    assert is_array_protocol(dset)
    assert is_array_protocol(dset, mutable=True)


@pytest.mark.skipif(Version(h5py.__version__) < Version("3.13"), reason="h5py#2550")
def test_array_protocol_h5_astypeview(h5file):
    """Test that h5py AsTypeView is an ArrayProtocol"""
    dset = h5file.create_dataset("x", shape=(10,), dtype="i2")
    view = dset.astype("i4")
    assert is_array_protocol(view)
    assert not is_array_protocol(view, mutable=True)
    assert is_array_protocol(view)


def test_array_protocol_h5_astypeview_compat(h5file):
    """Test that h5py_astype() returns an ArrayProtocol, also on older h5py versions.
    TODO delete this test when dropping support for h5py <3.13.
    """
    dset = h5file.create_dataset("x", shape=(10,), dtype="i2")
    view = h5py_astype(dset, "i4")
    assert is_array_protocol(view)
    assert not is_array_protocol(view, mutable=True)
    assert is_array_protocol(view)


def array_protocol_staged_changes():
    arr = StagedChangesArray.full((3, 3), chunk_size=(3, 1), dtype="f4")
    assert is_array_protocol(arr)
    assert is_array_protocol(arr, mutable=True)


def test_array_protocol_rawdataview():
    """RawDataView conforms to the writeable MutableArrayProtocol, so that it satisfies
    the return type of StagedChangesArray.commit()'s ``empty`` callback, even though its
    __getitem__/__setitem__/__array__ raise (it must only be used via read_many_slices).
    """
    view = RawDataView(np.zeros((4, 3)), offset=1, dtype=np.dtype("i8"))
    assert is_array_protocol(view)
    assert is_array_protocol(view, mutable=True)
