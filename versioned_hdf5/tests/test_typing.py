import h5py
import numpy as np
import numpy.ma as ma
import pytest
from packaging.version import Version

from ..typing_ import ArrayProtocol


class MinimalArray:
    """Minimal read-only NumPy array-like implementing the ArrayProtocol"""

    def __init__(self, arr):
        self._array = np.asarray(arr)
        self._array.flags.writeable = False

    @property
    def shape(self):
        return self._array.shape

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


def test_array_protocol():
    assert isinstance(MinimalArray(1), ArrayProtocol)
    assert isinstance(np.array(1), ArrayProtocol)
    assert isinstance(np.int64(1), ArrayProtocol)
    assert not isinstance(1, ArrayProtocol)
    assert not isinstance([1], ArrayProtocol)

    # numpy subclasses implement ArrayProtocol
    x = ma.masked_array([1, -1], mask=[0, 1], dtype="i2")
    assert isinstance(x, ArrayProtocol)


def test_array_protocol_h5_dataset(h5file):
    """Test that h5py.Dataset is a ArrayProtocol"""
    dset = h5file.create_dataset("x", shape=(10,), dtype="i2")
    assert isinstance(dset, ArrayProtocol)


@pytest.mark.skipif(Version(h5py.__version__) < Version("3.13"), reason="h5py#2550")
def test_array_protocol_h5_astypeview(h5file):
    """Test that h5py AsTypeView is a ArrayProtocol"""
    dset = h5file.create_dataset("x", shape=(10,), dtype="i2")
    assert isinstance(dset.astype("i4"), ArrayProtocol)
