import os
import sys

import pytest

# Kept in sync with ci/check_gil.py, which cannot be imported from here because
# pytest runs with --import-mode=importlib and does not put the rootdir on sys.path.
_REQUIRE_GIL_DISABLED = "VERSIONED_HDF5_REQUIRE_GIL_DISABLED"


def _gil_enabled() -> bool:
    # sys._is_gil_enabled only exists on free-threaded builds (Python >=3.13).
    is_enabled = getattr(sys, "_is_gil_enabled", None)
    return True if is_enabled is None else bool(is_enabled())


def pytest_addoption(parser):
    """Register pytest-run-parallel's ini keys when the plugin is absent.

    The wheel smoke-test job runs pytest from this checkout without installing the
    plugin. With ``filterwarnings = ["error"]``, the two plugin-only ini keys in
    pyproject.toml would otherwise fail collection as unknown options. Registering
    them here is harmless when the plugin is installed because it uses the same
    names, types, and defaults.
    """
    parser.addini(
        "thread_unsafe_fixtures",
        "list of thread-unsafe fixture names that cause a test to be run sequentially",
        type="linelist",
        default=[],
    )
    parser.addini(
        "thread_unsafe_functions",
        "list of thread-unsafe fully-qualified named functions that cause "
        "a test to run on one thread",
        type="linelist",
        default=[],
    )


def pytest_report_header(config):  # noqa: ARG001
    import h5py
    import ndindex

    gil = "enabled" if _gil_enabled() else "disabled"
    return (
        f"project deps: h5py-{h5py.__version__}, ndindex-{ndindex.__version__}\n"
        f"GIL: {gil}"
    )


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Make a GIL re-enabled at any point during the run a visible failure."""
    if os.environ.get(_REQUIRE_GIL_DISABLED) == "1" and _gil_enabled():
        pytest.exit(
            "The GIL was re-enabled during the test session in a free-threaded "
            "environment. A compiled extension is missing Py_mod_gil, or "
            "PYTHON_GIL=0 was not set.",
            returncode=1,
        )
