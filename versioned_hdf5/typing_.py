"""Type annotations.

Note: This module cannot be called 'typing' or 'types' as it will cause a
collision in Cython with the standard library 'typing' and 'types' modules.
(In Python, this issue was fixed in 3.0).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray


@runtime_checkable
class ArrayProtocol(Protocol):
    """Minimal read-only NumPy array-like interface.

    Not to be confused with numpy.typing.ArrayLike, which is any object that
    can be coerced into a numpy array, including a nested list.

    Note that this is quite a lot laxer than the Array API definition of an array:
    https://data-apis.org/array-api/latest/API_specification/array_object.html
    Notably, it misses all arithmetic dunder methods.
    h5py does not implement the Array API, so no point feature-matching it here.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def size(self) -> int: ...

    @property
    def ndim(self) -> int: ...

    @property
    def dtype(self) -> np.dtype: ...

    def __getitem__(self, index: Any) -> ArrayProtocol: ...

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> NDArray: ...


@runtime_checkable  # Does not support inheritance
class MutableArrayProtocol(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def size(self) -> int: ...

    @property
    def ndim(self) -> int: ...

    @property
    def dtype(self) -> np.dtype: ...

    def __getitem__(self, index: Any) -> MutableArrayProtocol: ...

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> NDArray: ...

    def __setitem__(self, index: Any, value: ArrayLike) -> None: ...


class Default(Enum):
    """Sentinel for default argument values."""

    DEFAULT = "default"


DEFAULT = Default.DEFAULT
