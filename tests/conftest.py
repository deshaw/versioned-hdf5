from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import h5py
import pytest
from numpy.testing import assert_array_equal

from versioned_hdf5 import VersionedHDF5File
from versioned_hdf5.backend import initialize

MAIN_THREAD_ID = threading.get_ident()


def make_name(template_or_prefix: str = "foo", /) -> str:
    """Return a deterministic name, suffixed with the thread ID in worker threads.

    Adapted from h5py/tests/common.py (h5py PR #2655). pytest-run-parallel runs the
    same test function in several threads, which share the fixture instances. Tests
    that share a file must use this to name datasets, groups, versions and attributes,
    otherwise the threads fight over the same names.

    In a serial run this returns ``template_or_prefix`` unchanged. In a worker thread
    it appends ``-<thread id>`` (or substitutes it for ``{}``).
    """
    tid = threading.get_ident()
    suffix = "" if tid == MAIN_THREAD_ID else f"-{tid}"
    if "{}" in template_or_prefix:
        return template_or_prefix.format(suffix)
    return template_or_prefix + suffix


@pytest.fixture(name="make_name")
def make_name_fixture() -> Callable[..., str]:
    """Fixture form of :func:`make_name`."""
    return make_name


# Run tests marked with @pytest.mark.slow last. See
# https://stackoverflow.com/questions/61533694/run-slow-pytest-commands-at-the-end-of-the-test-suite
def by_slow_marker(item):
    return bool(item.get_closest_marker("slow"))


def pytest_collection_modifyitems(items):
    items.sort(key=by_slow_marker)


class _FileState:
    """One initialized h5py.File (and VersionedHDF5File) per executing thread.

    pytest-run-parallel resolves fixtures once in the main thread and then invokes the
    same test function in N threads with those very same fixture values. Sharing a
    single h5py.File would make every thread stage the same version/dataset names.
    h5py's ``phil`` lock serializes the HDF5 calls, but it cannot disambiguate the
    names, so give each thread its own file instead. This mirrors h5py's ``make_name``
    approach without touching thousands of literal names in the test suite.
    """

    def __init__(self, tmp_path: Path, version_name: str | list[str] | None) -> None:
        self.tmp_path = tmp_path
        self.version_name = version_name
        self._local = threading.local()
        # list.append is atomic in CPython, including free-threaded builds. These
        # lists are only mutated by short, non-HDF5 operations, so they never take
        # phil while holding any other lock.
        self._files: list[h5py.File] = []
        self._vfiles: list[VersionedHDF5File] = []

    def _new_path(self) -> Path:
        n = getattr(self._local, "counter", 0)
        self._local.counter = n + 1
        return self.tmp_path / f"{make_name('file')}-{n}.h5"

    def open(self, *, version_name: str | list[str] | None = None) -> h5py.File:
        """Create a new initialized versioned file at a thread-unique path."""
        if version_name is None:
            version_name = self.version_name
        f = h5py.File(self._new_path(), "w")
        self._files.append(f)
        initialize(f)
        if version_name:
            if isinstance(version_name, str):
                version_name = [version_name]
            for name in version_name:
                f["_version_data/versions"].create_group(name)
        return f

    def file(self) -> h5py.File:
        f = getattr(self._local, "file", None)
        if f is None:
            f = self._local.file = self.open()
        return f

    def vfile(self) -> VersionedHDF5File:
        v = getattr(self._local, "vfile", None)
        if v is None:
            v = self._local.vfile = VersionedHDF5File(self.file())
            self._vfiles.append(v)
        return v

    def close_all(self) -> None:
        for v in self._vfiles:
            v.close()
        for f in self._files:
            _close_quietly(f)


class _Proxy:
    """Base class for the per-thread fixture proxies."""

    def __init__(self, state: _FileState) -> None:
        object.__setattr__(self, "_state", state)

    def __getattr__(self, name):
        return getattr(self._resolve(), name)

    def __setattr__(self, name, value):
        setattr(self._resolve(), name, value)

    def __delattr__(self, name):
        delattr(self._resolve(), name)

    def __repr__(self):
        return repr(self._resolve())

    def __getitem__(self, key):
        return self._resolve()[key]

    def __setitem__(self, key, value):
        self._resolve()[key] = value

    def __delitem__(self, key):
        del self._resolve()[key]

    def __contains__(self, key):
        return key in self._resolve()

    def __iter__(self):
        return iter(self._resolve())

    def __len__(self):
        return len(self._resolve())


class _FileProxy(_Proxy):
    # Library code (and h5py) occasionally checks isinstance(f, h5py.File).
    @property  # type: ignore[misc]
    def __class__(self):
        return h5py.File

    def _resolve(self) -> h5py.File:
        return self._state.file()


class _VFileProxy(_Proxy):
    # replay.delete_versions/modify_metadata switch on isinstance(f,
    # VersionedHDF5File) to unwrap f.f. Proxying must preserve that check.
    @property  # type: ignore[misc]
    def __class__(self):
        return VersionedHDF5File

    def _resolve(self) -> VersionedHDF5File:
        return self._state.vfile()


def _close_quietly(f: h5py.File) -> None:
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
def file_state(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[_FileState]:
    m = request.node.get_closest_marker("setup_args")
    kwargs = m.kwargs if m is not None else {}
    state = _FileState(tmp_path, kwargs.get("version_name", None))
    yield state
    state.close_all()


@pytest.fixture
def h5file(file_state: _FileState) -> _FileProxy:
    """An h5py.File-like object resolving to a distinct file in each thread."""
    return _FileProxy(file_state)


@pytest.fixture
def vfile(file_state: _FileState) -> _VFileProxy:
    """A VersionedHDF5File-like object resolving to a distinct file per thread."""
    return _VFileProxy(file_state)


@pytest.fixture
def setup_vfile(file_state: _FileState) -> Callable[..., h5py.File]:
    """Fixture which provides a function that creates an hdf5 file, optionally
    with groups. The filename is unique per thread and per call.
    """
    return file_state.open


def assert_slab_offsets(version, name, expect):
    """Assert that the StagedChangesArray.slab_offsets matches the expectations.
    This is useful to test chunk reuse.
    """
    ds = version[name]
    assert_array_equal(ds.staged_changes.slab_offsets, expect)
