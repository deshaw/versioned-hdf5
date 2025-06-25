from __future__ import annotations

import h5py
import numpy as np
from numpy.typing import DTypeLike

from versioned_hdf5.tools import NP_VERSION
from versioned_hdf5.typing_ import ArrayProtocol

H5PY_VERSION = tuple(int(i) for i in h5py.__version__.split(".")[:2])
HAS_NPYSTRINGS = H5PY_VERSION >= (3, 14) and NP_VERSION >= (2,)  # a.k.a StringDType


if H5PY_VERSION >= (3, 13):

    def h5py_astype(ds: h5py.Dataset, dtype: DTypeLike) -> ArrayProtocol:
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

    def h5py_astype(ds: h5py.Dataset, dtype: DTypeLike) -> ArrayProtocol:
        return AsTypeView(ds.astype(dtype))
