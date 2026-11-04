"""Regression tests for free-threaded builds and the single coarse-grained lock."""

from __future__ import annotations

import ast
import datetime
import gc
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.testing import assert_equal

from versioned_hdf5 import VersionedHDF5File, wrappers
from versioned_hdf5._lock import phil


def test_gil_disabled_when_required():
    """The free-threaded pixi environments must actually run with the GIL off."""
    if os.environ.get("VERSIONED_HDF5_REQUIRE_GIL_DISABLED") != "1":
        pytest.skip("not a free-threaded pixi environment")
    assert hasattr(sys, "_is_gil_enabled"), "not a free-threaded interpreter"
    assert not sys._is_gil_enabled(), "the GIL was re-enabled during this test"


@pytest.mark.thread_unsafe(
    reason="spawns a helper thread to prove stage_version does not hold phil"
)
def test_stage_version_does_not_hold_phil(tmp_path, make_name: Callable[..., str]):
    """stage_version() must not hold the global lock across user code.

    The helper thread probes phil without blocking and then performs direct h5py I/O
    and finalization. If phil were held for the whole context, the nonblocking probe
    would fail and the h5py I/O would block. Timings are only guards against a hung
    thread; correctness is asserted from the probe result and thread completion.
    """
    ready = threading.Event()
    acquired = []
    errors = []
    probe_path = tmp_path / f"probe-{make_name('lock')}.h5"

    def probe():
        got = phil.acquire(blocking=False)
        acquired.append(got)
        if got:
            phil.release()
        ready.set()
        try:
            f = h5py.File(probe_path, "w")
            try:
                f.create_dataset("data", data=np.arange(3))
            finally:
                f.close()
            del f
            gc.collect()
        except BaseException as e:  # pragma: no cover - reported by the main thread
            errors.append(e)

    with h5py.File(tmp_path / f"{make_name('lock')}.h5", "w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("v0", "") as g:
            t = threading.Thread(target=probe)
            t.start()
            assert ready.wait(timeout=10), "helper thread did not reach the phil probe"
            t.join(timeout=10)
            assert not t.is_alive(), (
                "helper thread blocked while stage_version held phil"
            )
            assert acquired == [True], "phil was held across stage_version user code"
            assert not errors, errors
            g.create_dataset("data", data=np.arange(3), chunks=(2,))
        assert_equal(vf["v0"]["data"][:], np.arange(3))


@pytest.mark.thread_unsafe(
    reason="spawns threads that deliberately share one HDF5 file"
)
def test_concurrent_stage_versions_shared_file(tmp_path, make_name: Callable[..., str]):
    """Several threads may stage and commit versions on one VersionedHDF5File.

    The barrier is inside stage_version(), so all threads must be able to open the
    context at once. A whole-context global lock would break the barrier. Names are
    thread-unique; setup, mutations and commits are individually phil-protected.
    """
    n_threads = 4
    expected = np.arange(3)
    names: list[tuple[str, str]] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(n_threads)

    def worker(vf: VersionedHDF5File):
        version = make_name("v")
        dataset = make_name("data")
        try:
            with vf.stage_version(version, "") as g:
                barrier.wait(timeout=10)
                g.create_dataset(dataset, data=expected, chunks=(2,))
            names.append((version, dataset))
        except BaseException as e:  # pragma: no cover - reported by the main thread
            errors.append(e)

    with h5py.File(tmp_path / f"{make_name('shared')}.h5", "w") as f:
        vf = VersionedHDF5File(f)
        threads = [
            threading.Thread(target=worker, args=(vf,)) for _ in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors
        assert len(names) == n_threads
        assert set(vf) == {version for version, _ in names}
        for version, dataset in names:
            assert_equal(vf[version][dataset][:], expected)


def test_stage_version_failure_preserves_newer_current_and_parents(
    tmp_path, make_name: Callable[..., str]
):
    """A failed commit must not roll back another version's current pointer.

    Opening another stage context while ``bad`` is staged is a deterministic
    interleaving: ``good`` and ``child`` become current before ``bad`` fails on
    timestamp validation. ``child`` deliberately points at ``bad`` so rollback
    must reparent it rather than leave a dangling DAG parent.
    """
    bad_timestamp = datetime.datetime(2020, 6, 29, 22, 12, 56)

    def as_text(value):
        return value.decode() if isinstance(value, bytes) else value

    def stage_good():
        with vf.stage_version("good", "") as good:
            good.create_dataset("data", data=np.arange(4), chunks=(2,))

    def stage_child():
        with vf.stage_version("child", prev_version="bad") as child:
            child.create_dataset("data", data=np.arange(4), chunks=(2,))

    def interleave():
        with vf.stage_version("bad", timestamp=bad_timestamp) as bad:
            bad.create_dataset("data", data=np.arange(4), chunks=(2,))
            stage_good()
            stage_child()
            assert vf.current_version == "child"

    with h5py.File(tmp_path / f"{make_name('stage-failure')}.h5", "w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("r0", "") as g:
            g.create_dataset("data", data=np.arange(4), chunks=(2,))

        with pytest.raises(ValueError, match="UTC"):
            interleave()

        versions = set(vf)
        assert vf.current_version == "child"
        assert versions == {"r0", "good", "child"}
        assert as_text(vf["child"].attrs["prev_version"]) == "r0"
        for version in versions:
            parent = as_text(vf[version].attrs["prev_version"])
            assert parent == "__first_version__" or parent in versions
        assert_equal(vf["good"]["data"][:], np.arange(4))
        assert_equal(vf["child"]["data"][:], np.arange(4))


def test_no_with_phil_decorated_generators():
    """h5py's with_phil releases the lock as soon as a generator object is returned."""

    def body_has_yield(node: ast.AST) -> bool:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                return True
            if isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
            ):
                continue
            if body_has_yield(child):
                return True
        return False

    root = Path(wrappers.__file__).parent
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any("with_phil" in ast.unparse(d) for d in node.decorator_list):
                assert not body_has_yield(node), (
                    f"{path}:{node.lineno}: {node.name} is a generator decorated "
                    "with @with_phil"
                )
