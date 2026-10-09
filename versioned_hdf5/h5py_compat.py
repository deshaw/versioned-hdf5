from __future__ import annotations

import h5py
import numpy as np
from numpy.typing import DTypeLike

from versioned_hdf5.tools import NP_GE_200
from versioned_hdf5.typing_ import ArrayProtocol

H5PY_VERSION = tuple(int(i) for i in h5py.__version__.split(".")[:2])
# Native variable-width strings support (NpyStrings a.k.a StringDType)
HAS_NPYSTRINGS = H5PY_VERSION >= (3, 14) and NP_GE_200


class FixedStringToObjectView:
    """Read a fixed-width string dataset as an array of ``object`` strings.

    HDF5 has no conversion path from fixed-width to variable-length strings, so
    ``h5py.Dataset.astype(object)`` itself raises::

        OSError: Can't synchronously read data (no appropriate function for
        conversion path)

    Read the data as-is and let NumPy do the conversion instead.
    """

    def __init__(self, ds: h5py.Dataset):
        self._ds = ds

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, item) -> np.ndarray | np.generic:
        return np.asarray(self._ds[item]).astype(object)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(object)

    @property
    def ndim(self) -> int:
        return self._ds.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self._ds.shape

    @property
    def size(self) -> int:
        return self._ds.size

    def __array__(
        self,
        dtype: DTypeLike | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        if copy is False:
            raise ValueError("Cannot return an ndarray view of a Dataset")
        return np.asarray(self[()], dtype=dtype or self.dtype)


if H5PY_VERSION >= (3, 13):

    def _ds_astype(ds: h5py.Dataset, dtype: DTypeLike) -> ArrayProtocol:
        return ds.astype(dtype)

else:
    # Backport AsTypeView to h5py <3.13

    class AsTypeView:
        """Wrap around AstypeWrapper, which exclusively defined
        __getitem__ and __len__.
        """

        def __init__(self, wrapper):
            self._wrapper = wrapper

        def __len__(self) -> int:
            return len(self._wrapper)

        def __getitem__(self, item) -> np.ndarray | np.generic:
            return self._wrapper[item]

        @property
        def dtype(self) -> np.dtype:
            return self._wrapper._dtype

        @property
        def ndim(self) -> int:
            return self._wrapper._dset.ndim

        @property
        def shape(self) -> tuple[int, ...]:
            return self._wrapper._dset.shape

        @property
        def size(self) -> int:
            return self._wrapper._dset.size

        def __array__(
            self,
            dtype: DTypeLike | None = None,
            copy: bool | None = None,
        ) -> np.ndarray:
            if copy is False:
                raise ValueError("Cannot return a ndarray view of a Dataset")
            # If self.ndim == 0, convert np.generic back to np.ndarray
            return np.asarray(self[()], dtype=dtype or self.dtype)

    def _ds_astype(ds: h5py.Dataset, dtype: DTypeLike) -> ArrayProtocol:
        return AsTypeView(ds.astype(dtype))


def h5py_astype(ds: h5py.Dataset, dtype: DTypeLike) -> ArrayProtocol:
    """Like ``ds.astype(dtype)``, but working around HDF5's lack of a conversion
    path from fixed-width strings to variable-length (``object``) strings."""
    if np.dtype(dtype) == object and ds.dtype.kind in ("S", "U"):
        return FixedStringToObjectView(ds)
    return _ds_astype(ds, dtype)
