"""Type annotations.

Note: This module cannot be called 'typing' or 'types' as it will cause a
collision in Cython with the standard library 'typing' and 'types' modules.
(In Python, this issue was fixed in 3.0).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import DTypeLike, NDArray


@runtime_checkable
class ArrayProtocol(Protocol):
    """Minimal read-only NumPy array-like interface.

    Not to be confused with numpy.typing.ArrayLike, which is any object that
    can be coerced into a numpy array, including a nested list.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def dtype(self) -> np.dtype: ...

    def __getitem__(self, key: Any) -> ArrayProtocol: ...

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> NDArray: ...
