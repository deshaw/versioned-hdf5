import time
import warnings

import h5py
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from versioned_hdf5 import VersionedHDF5File
from versioned_hdf5.h5py_compat import HAS_NPYSTRINGS
from versioned_hdf5.hashtable import Hashtable
from versioned_hdf5.wrappers import (
    DatasetWrapper,
    InMemoryArrayDataset,
    InMemoryDataset,
    InMemorySparseDataset,
)

from .conftest import assert_slab_offsets

pytestmark = pytest.mark.skipif(
    not HAS_NPYSTRINGS,
    reason="NpyStrings require h5py >=3.14 and NumPy >=2.0",
)


def assert_object_array_equal(actual, expect):
    """Assert that actual is an object array of either bytes or str.
    The exact contents (bytes or str) are allowed to diverge between h5py and
    versioned_hdf5.
    """
    actual = np.asarray(actual)
    expect = np.asarray(expect, dtype=object)
    assert actual.dtype.kind == "O"
    assert actual.shape == expect.shape

    for a, b in zip(actual.flat, expect.flat, strict=True):
        if isinstance(a, bytes):
            a = a.decode("utf-8")
        assert a == b, (actual, expect)


def assert_swaps_counter(ds, expect):
    """Test BufferMixin._swaps_counter. Do nothing for plain h5py.Dataset."""
    if isinstance(ds, DatasetWrapper):
        ds = ds.dataset
    if isinstance(ds, (InMemoryDataset, InMemoryArrayDataset, InMemorySparseDataset)):
        assert ds._swaps_counter == expect
    else:
        assert type(ds) is h5py.Dataset  # not a subclass


@pytest.fixture(
    params=[
        "h5py.Dataset",
        "InMemoryArrayDataset",
        "InMemorySparseDataset",
        "InMemoryDataset",
    ]
)
def any_dataset(h5file, request):
    """Run the test four times, yielding all possible types of writeable datasets."""
    if request.param == "h5py.Dataset":
        ds = h5file.create_dataset("x", data=["foo", ""], dtype="T")
        yield ds
    else:
        vfile = VersionedHDF5File(h5file)

        if request.param == "InMemoryArrayDataset":
            with vfile.stage_version("a0") as v:
                ds = v.create_dataset("x", data=["foo", ""], dtype=h5py.string_dtype())
                assert isinstance(ds.dataset, InMemoryArrayDataset)
                yield ds

        elif request.param == "InMemorySparseDataset":
            with vfile.stage_version("s0") as v:
                ds = v.create_dataset(
                    "x", shape=(2,), dtype=h5py.string_dtype(), fillvalue=b""
                )
                ds[0] = "foo"
                assert isinstance(ds, InMemorySparseDataset)
                yield ds

        elif request.param == "InMemoryDataset":
            with vfile.stage_version("d0") as v:
                v.create_dataset("x", data=["foo", ""], dtype=h5py.string_dtype())
            with vfile.stage_version("d1") as v:
                ds = v["x"]
                assert isinstance(ds.dataset, InMemoryDataset)
                yield ds

        else:  # pragma: no cover
            raise AssertionError("unreachable")

        ds = vfile[None]["x"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        assert_object_array_equal(ds, ["bar", ""])


def test_match_behaviour(any_dataset):
    """Test that versioned_hdf5 behaves like h5py when it comes to NpyStrings UX."""
    ds = any_dataset
    assert ds.dtype.kind == "O"
    assert ds.fillvalue == b""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        assert_object_array_equal(ds, ["foo", ""])

        ds[0] = np.array("bar", dtype="T")
        assert ds.dtype.kind == "O"
        assert_object_array_equal(ds[:], ["bar", ""])  # __getitem__
        assert_object_array_equal(np.array(ds), ["bar", ""])  # __array__

        view = ds.astype("T")
        assert view.dtype.kind == "T"
        assert_array_equal(view, np.array(["bar", ""], dtype="T"), strict=True)
        with pytest.raises((TypeError, ValueError)):
            view[0] = "baz"


def test_setitem_bare_strings(any_dataset):
    """Setting a bare string into a dataset doesn't swap the buffer dtype."""
    ds = any_dataset
    assert_swaps_counter(ds, 0)
    ds[0] = "bar"  # Doesn't swap
    assert_swaps_counter(ds, 0)
    ds[0] = np.asarray("bar", dtype="O")  # Doesn't swap
    assert_swaps_counter(ds, 0)
    ds[0] = np.asarray("bar", dtype="T")  # Swap from object dtype to StringDType
    assert_swaps_counter(ds, 1)
    ds[0] = np.asarray("bar", dtype="T")  # Doesn't swap
    assert_swaps_counter(ds, 1)
    ds[0] = "bar"  # Doesn't swap
    assert_swaps_counter(ds, 1)


def test_commit_doesnt_swap_buffer(vfile):
    """Committing a dataset with staged changes doesn't swap the buffer dtype."""
    with vfile.stage_version("v0") as v:
        x = v.create_dataset("x", data=["foo"], dtype="T")
        y = v.create_dataset("y", data=["foo"], dtype=h5py.string_dtype())
        assert isinstance(x.dataset, InMemoryArrayDataset)
        assert isinstance(y.dataset, InMemoryArrayDataset)
        assert x.dtype.kind == "O"
        assert y.dtype.kind == "O"
        assert x._buffer.dtype.kind == "T"
        assert y._buffer.dtype.kind == "O"

    assert_swaps_counter(x, 0)
    assert_swaps_counter(y, 0)

    with vfile.stage_version("v1") as v:
        x = v["x"]
        y = v["y"]
        # swap x's buffer while there are no staged slabs; counter does not increment.
        x[0] = np.asarray("bar", dtype="T")
        y[0] = "bar"
        assert isinstance(x.dataset, InMemoryDataset)
        assert isinstance(y.dataset, InMemoryDataset)
        assert x.dtype.kind == "O"
        assert y.dtype.kind == "O"
        assert x.dataset._buffer.dtype.kind == "T"
        assert y.dataset._buffer.dtype.kind == "O"

    assert_swaps_counter(x, 0)
    assert_swaps_counter(y, 0)

    with vfile.stage_version("v2") as v:
        x = v.create_dataset("x_sparse", shape=(1,), dtype="T")  # note: inconsequential
        y = v.create_dataset("y_sparse", shape=(1,), dtype=h5py.string_dtype())
        # swap x's buffer while there are no staged slabs; counter does not increment.
        x[0] = np.asarray("bar", dtype="T")
        y[0] = "foo"
        assert isinstance(x, InMemorySparseDataset)
        assert isinstance(y, InMemorySparseDataset)
        assert x.dtype.kind == "O"
        assert y.dtype.kind == "O"
        assert x._buffer.dtype.kind == "T"
        assert y._buffer.dtype.kind == "O"

    assert_swaps_counter(x, 0)
    assert_swaps_counter(y, 0)


def test_convert_to_array_dataset(vfile):
    """When DatasetWrapper hot-swaps a InMemoryDataset with a InMemoryArrayDataset
    after a full-array update, don't revert to object dtype."""
    with vfile.stage_version("v0") as v:
        x = v.create_dataset("x", data=["foo", "bar"], dtype="T")
        y = v.create_dataset("y", data=["foo", "bar"], dtype="T")
        z = v.create_dataset("z", data=["foo", "bar"], dtype="T")
    with vfile.stage_version("v1") as v:
        x = v["x"]
        y = v["y"]
        z = v["z"]
        # swap buffers
        for ds in (x, y, z):
            ds[0] = np.asarray("foo", dtype="T")
            assert isinstance(ds.dataset, InMemoryDataset)
            assert ds.dtype.kind == "O"
            assert ds.dataset._buffer.dtype.kind == "T"
            assert_swaps_counter(ds, 0)

        x[:] = np.asarray(["baz", "qux"], dtype="T")
        y[:] = np.asarray(["baz", "qux"], dtype=h5py.string_dtype())
        z[:] = ["baz", "qux"]

        for ds in (x, y, z):
            assert isinstance(x.dataset, InMemoryArrayDataset)
            assert ds.dtype.kind == "O"
            assert_swaps_counter(ds, 0)

        assert x.dataset._buffer.dtype.kind == "T"
        assert y.dataset._buffer.dtype.kind == "O"
        assert z.dataset._buffer.dtype.kind == "T"


@pytest.mark.slow
def test_tight_iteration_big_O_performance(h5file):
    """In h5py, it is OK to create and destroy astype("T") views in rapid succession,
    interleaved with __setitem__ calls of data with dtype="T":

        for i in range(0, ds.shape[0], 2):
            ds[i + 1, :] = f(ds.astype("T")[i, :])  # f() accepts and returns dtype="T"

    Test that, in versioned_hdf5, performance is O() to the updated elements, not to the
    size of the whole dataset - in other words, we're not converting everything.
    """
    NSMALL = 1000
    NLARGE = 100_000
    CHUNKS = (100, 1)

    def create_dataset(v, name, size, sparse=False):
        if sparse:
            ds = v.create_dataset(name, shape=(size, 1), chunks=CHUNKS, dtype="T")
            # InMemorySparseDataset cannot know if we're later going to use object
            # dtype or StringDType, so by default it uses object dtype.
            ds[:] = np.asarray("x", dtype="T")
        else:
            ds = v.create_dataset(
                name, data=np.full((size, 1), "x", dtype="T"), chunks=CHUNKS
            )
        return ds

    def benchmark(small, large):
        # Don't use wall time. The hypervisor on CI hosts can occasionally
        # steal the CPU for multiple seconds away from the VM.
        t0 = time.thread_time()
        for i in range(0, NSMALL, 2):
            small[i + 1, :] = np.strings.capitalize(small.astype("T")[i, :])
        t1 = time.thread_time()
        # large is much larger than small; however we benchmark the iteration on the
        # same number of eleemnts
        for i in range(0, NSMALL, 2):
            large[i + 1, :] = np.strings.capitalize(large.astype("T")[i, :])
        t2 = time.thread_time()

        # Generously allow 5x difference in performance to ensure stability.
        # If there are conversions of the whole dataset, we're going to get 100x.
        np.testing.assert_allclose(t2 - t1, t1 - t0, rtol=5, atol=0)

    small = create_dataset(h5file, "small", NSMALL)
    large = create_dataset(h5file, "large", NLARGE)
    benchmark(small, large)

    vfile = VersionedHDF5File(h5file)
    with vfile.stage_version("v0") as v:
        small = create_dataset(v, "small", NSMALL)
        large = create_dataset(v, "large", NLARGE)
        assert isinstance(small.dataset, InMemoryArrayDataset)
        assert isinstance(large.dataset, InMemoryArrayDataset)
        benchmark(small, large)
        assert_swaps_counter(large, 0)

    with vfile.stage_version("v1") as v:
        small = v["small"]
        large = v["large"]
        assert isinstance(small.dataset, InMemoryDataset)
        assert isinstance(large.dataset, InMemoryDataset)
        benchmark(small, large)
        assert_swaps_counter(large, 0)

    with vfile.stage_version("v2") as v:
        small = create_dataset(v, "small_sp", NSMALL, sparse=True)
        large = create_dataset(v, "large_sp", NLARGE, sparse=True)
        assert isinstance(small, InMemorySparseDataset)
        assert isinstance(large, InMemorySparseDataset)
        benchmark(small, large)
        assert_swaps_counter(large, 0)


def test_multiple_swaps_warning(vfile):
    """Test that flipping the buffer multiple times between object and StringDType
    raises a warning.
    """
    match = "Performing multiple internal conversions"

    with vfile.stage_version("v0") as v:
        ds = v.create_dataset("x", data=["foo"], dtype="T")
        assert isinstance(ds.dataset, InMemoryArrayDataset)
        assert ds._buffer.dtype.kind == "T"
        assert_swaps_counter(ds, 0)

        ds[0] = np.asarray("bar", dtype=h5py.string_dtype())
        assert_swaps_counter(ds, 1)

        with pytest.warns(UserWarning, match=match):
            ds[0] = np.asarray("bar", dtype="T")
        assert_swaps_counter(ds, 2)

        _ = ds.astype("T")
        assert_swaps_counter(ds, 2)

        with pytest.warns(UserWarning, match=match):
            _ = ds[:]  # convert to object dtype
        assert_swaps_counter(ds, 3)

        _ = ds[:]
        assert_swaps_counter(ds, 3)

        _ = np.asarray(ds)
        assert_swaps_counter(ds, 3)

        _ = np.asarray(ds, dtype=h5py.string_dtype())
        assert_swaps_counter(ds, 3)

        with pytest.warns(UserWarning, match=match):
            _ = ds.astype("T")
        assert_swaps_counter(ds, 4)

        _ = np.asarray(ds, dtype="T")
        assert_swaps_counter(ds, 4)

        with pytest.warns(UserWarning, match=match):
            _ = np.asarray(ds)
        assert_swaps_counter(ds, 5)


class BadHashtable(Hashtable):
    def hash(self, data):  # noqa: ARG002
        raise AssertionError("monkeypatch OK")


class NpyStringsHashtable(Hashtable):
    def hash(self, data):
        assert data.dtype.kind == "T", data
        return super().hash(data)


def test_hashtable_monkeypatch(vfile, monkeypatch):
    """Test monkey-patching of the hash table"""
    monkeypatch.setattr("versioned_hdf5.backend.Hashtable", BadHashtable)
    with (
        pytest.raises(AssertionError, match="monkeypatch OK"),
        vfile.stage_version("v0") as v,
    ):
        v.create_dataset("bad", data=["foo"], dtype="T")


def test_hash_native(vfile, monkeypatch):
    """Test that chunks reach the hash() function without being converted to object
    dtype. This is for future-proofing to enable using a cython-based hash algorithm.
    """
    monkeypatch.setattr("versioned_hdf5.backend.Hashtable", NpyStringsHashtable)

    # Hash from InMemoryArrayDataset
    with vfile.stage_version("v0") as v:
        ds = v.create_dataset(
            "dense", data=["ab", "cd", "ef", "gh"], chunks=(2,), dtype="T"
        )
        assert isinstance(ds.dataset, InMemoryArrayDataset)
        assert ds._buffer.dtype.kind == "T"

    # Hash from InMemoryDataset
    with vfile.stage_version("v1") as v:
        ds = v["dense"]
        # One hash differs; chunk order is swapped
        # Avoid having the whole dataset replaced with InMemoryArrayDataset
        ds[:3] = np.asarray(["Ef", "gh", "ab"], dtype="T")
        ds[3] = "cd"
        assert isinstance(ds.dataset, InMemoryDataset)
        assert ds.dataset._buffer.dtype.kind == "T"

    # Hash from InMemorySparseDataset
    with vfile.stage_version("v2") as v:
        ds = v.create_dataset("sparse", shape=(4,), chunks=(2,), dtype="T")
        ds[:] = np.asarray(["ab", "cd", "ef", "gh"], dtype="T")
        assert isinstance(ds, InMemorySparseDataset)
        assert ds._buffer.dtype.kind == "T"

    with vfile.stage_version("v3") as v:
        # Chunk 0 from v0 has been reused as chunk 1 in v1
        assert_slab_offsets(v, "dense", [4, 0])
        assert_slab_offsets(v, "sparse", [0, 2])

        ds = v["dense"]
        # Use a full-dataset update to replace InMemoryDataset with InMemoryArrayDataset
        ds[:] = np.asarray(["ab", "cd", "ij", "kl"], dtype="T")
        assert isinstance(ds.dataset, InMemoryArrayDataset)
        assert ds.dataset._buffer.dtype.kind == "T"

    with vfile.stage_version("v4") as v:
        assert_slab_offsets(v, "dense", [0, 6])


def test_hash_compat(vfile):
    """Test that NpyStrings and object strings produce identical hashes."""
    with vfile.stage_version("v0") as v:
        v.create_dataset(
            "a", data=["x", "y", "z"], chunks=(1,), dtype=h5py.string_dtype()
        )
        v.create_dataset("b", data=["x", "y", "z"], chunks=(1,), dtype="T")
        v.create_dataset(
            "c", data=["x", "y", "z"], chunks=(1,), dtype=h5py.string_dtype()
        )
        v.create_dataset("d", data=["x", "y", "z"], chunks=(1,), dtype="T")

    with vfile.stage_version("v1") as v:
        # chunk 0 is updated unchanged
        # chunk 1 is updated to a different value
        # chunk 2 is not updated

        # Same dtype as v0
        v["a"][:2] = np.asarray(["x", "w"], dtype=h5py.string_dtype())
        v["b"][:2] = np.asarray(["x", "w"], dtype="T")
        # dtype changed object <-> StringDType
        v["c"][:2] = np.asarray(["x", "w"], dtype="T")
        v["d"][:2] = np.asarray(["x", "w"], dtype=h5py.string_dtype())

    with vfile.stage_version("v2") as v:
        assert_slab_offsets(v, "a", [0, 3, 2])
        assert_slab_offsets(v, "b", [0, 3, 2])
        assert_slab_offsets(v, "c", [0, 3, 2])
        assert_slab_offsets(v, "d", [0, 3, 2])


def test_hash_arena_strings(vfile):
    """Test that we're not calling hashlib.hash.digest() on NpyStrings,
    which could cause hash collisions:
    https://github.com/numpy/numpy/issues/29226
    """
    data = np.asarray(
        [
            "This string is too long to be stored inline by NpyStrings",
            "This string has the same length but differs in content!  ",
            "Yet another very long string of different content        ",
        ],
        dtype="T",
    )

    with vfile.stage_version("v0") as v:
        v.create_dataset("x", data=data[:2], chunks=(1,), dtype="T")
    with vfile.stage_version("v1") as v:
        assert_slab_offsets(v, "x", [0, 1])
        v["x"][0] = data[1]
        v["x"][1] = data[2]
    with vfile.stage_version("v2") as v:
        assert_slab_offsets(v, "x", [1, 2])


def test_datasetwrapper_setitem(vfile):
    """DatasetWrapper.__setitem__() hot-swaps a InMemoryDataset for a
    InMemoryArrayDataset when the whole dataset is updated.
    Test that this doesn't flip the buffer dtype.
    """
    with vfile.stage_version("v0") as v:
        ds = v.create_dataset("x", data=["foo", "bar"], dtype="T")

    with vfile.stage_version("v1") as v:
        ds = v["x"]
        # swap buffers
        _ = ds.astype("T")
        assert isinstance(ds.dataset, InMemoryDataset)
        assert ds.dataset._buffer.dtype.kind == "T"
        assert ds.dtype.kind == "O"

        ds[:] = ["baz", "qux"]
        assert isinstance(ds.dataset, InMemoryArrayDataset)
        assert ds.dataset._buffer.dtype.kind == "T"
        assert ds.dtype.kind == "O"


def test_datasetwrapper_resize(vfile):
    """DatasetWrapper.resize() hot-swaps a InMemoryArrayDataset for a
    InMemorySparseDataset when enlarging.
    Test that this doesn't flip the buffer dtype.
    """
    with vfile.stage_version("v0") as v:
        ds = v.create_dataset(
            "x",
            data=["foo", "bar"],
            dtype="T",
            chunks=(2,),
            maxshape=(None,),
        )
        _ = ds.astype("T")
        assert isinstance(ds.dataset, InMemoryArrayDataset)
        assert ds.dataset._buffer.dtype.kind == "T"
        assert ds.dtype.kind == "O"

        ds.resize((3,))
        assert isinstance(ds.dataset, InMemorySparseDataset)
        assert ds.dataset._buffer.dtype.kind == "T"
        assert ds.dtype.kind == "O"
