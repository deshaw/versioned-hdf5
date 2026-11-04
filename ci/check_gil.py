"""Fail loudly if a free-threaded pixi environment has the GIL enabled.

The ``py314t``/``py315t`` environments set both ``PYTHON_GIL=0`` and
``VERSIONED_HDF5_REQUIRE_GIL_DISABLED=1``. Importing a compiled extension that does
not declare ``Py_mod_gil`` re-enables the GIL for the whole process. This standalone
check imports h5py and every versioned-hdf5 Cython module before asserting; smoke and
the root pytest conftest check again after real operations and at session end.
"""

from __future__ import annotations

import importlib
import os
import sys

ENV_VAR = "VERSIONED_HDF5_REQUIRE_GIL_DISABLED"


def is_required() -> bool:
    return os.environ.get(ENV_VAR) == "1"


def assert_gil_disabled() -> str:
    """Return a status line, or raise RuntimeError if the GIL is enabled."""
    if not is_required():
        return "GIL check: disabled (not a free-threaded pixi environment)"
    if not hasattr(sys, "_is_gil_enabled"):
        raise RuntimeError(
            f"{ENV_VAR}=1 but this interpreter was not built with free threading"
        )
    if sys._is_gil_enabled():
        raise RuntimeError(
            f"{ENV_VAR}=1 but the GIL is enabled. Check PYTHON_GIL=0 and make sure "
            "every compiled extension declares Py_mod_gil / freethreading_compatible."
        )
    return "GIL check: GIL disabled (free-threaded build)"


def main() -> None:
    for module in (
        "h5py",
        "versioned_hdf5",
        "versioned_hdf5.cytools",
        "versioned_hdf5.hash",
        "versioned_hdf5.slicetools",
        "versioned_hdf5.staged_changes",
        "versioned_hdf5.subchunk_map",
    ):
        importlib.import_module(module)
    print(assert_gil_disabled())


if __name__ == "__main__":
    main()
