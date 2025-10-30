from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path

import h5py
import pytest
from numpy.testing import assert_array_equal

from versioned_hdf5 import VersionedHDF5File
from versioned_hdf5.backend import initialize


# Run tests marked with @pytest.mark.slow last. See
# https://stackoverflow.com/questions/61533694/run-slow-pytest-commands-at-the-end-of-the-test-suite
def by_slow_marker(item):
    return bool(item.get_closest_marker("slow"))


def pytest_collection_modifyitems(items):
    items.sort(key=by_slow_marker)


@pytest.fixture
def h5file(
    setup_vfile: Callable[..., h5py.File], request: pytest.FixtureRequest
) -> Generator[h5py.File]:
    m = request.node.get_closest_marker("setup_args")
    kwargs = m.kwargs if m is not None else {}
    version_name = kwargs.get("version_name", None)
    f = setup_vfile(version_name=version_name)
    yield f
    try:
        f.close()
    # Workaround upstream h5py bug. https://github.com/deshaw/versioned-hdf5/issues/162
    except ValueError as e:
        if e.args[0] == "Unrecognized type code -1":
            return
        raise
    except RuntimeError as e:
        if e.args[0] in [
            "Can't increment id ref count (can't locate ID)",
            "Unspecified error in H5Iget_type (return value <0)",
            "Can't retrieve file id (invalid data ID)",
        ]:
            return
        raise


@pytest.fixture
def vfile(h5file: h5py.File) -> Generator[VersionedHDF5File]:
    file = VersionedHDF5File(h5file)
    yield file
    file.close()


@pytest.fixture
def setup_vfile(tmp_path: Path) -> Callable[..., h5py.File]:
    """Fixture which provides a function that creates an hdf5 file, optionally
    with groups.
    """

    def _setup_vfile(*, version_name: str | list[str] | None = None) -> h5py.File:
        f = h5py.File(tmp_path / "file.h5", "w")
        initialize(f)
        if version_name:
            if isinstance(version_name, str):
                version_name = [version_name]
            for name in version_name:
                f["_version_data/versions"].create_group(name)
        return f

    return _setup_vfile


def assert_slab_offsets(version, name, expect):
    """Assert that the StagedChangesArray.slab_offsets matches the expectations.
    This is useful to test chunk reuse.
    """
    ds = version[name]
    assert_array_equal(ds.staged_changes.slab_offsets, expect)
