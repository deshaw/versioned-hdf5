import datetime
import itertools
import logging
import os
import pathlib
import shutil

import h5py
import numpy as np
import pytest
from h5py._hl.filters import guess_chunk
from numpy.testing import assert_equal

from versioned_hdf5 import VersionedHDF5File
from versioned_hdf5.backend import DATA_VERSION, DEFAULT_CHUNK_SIZE
from versioned_hdf5.replay import delete_versions
from versioned_hdf5.versions import TIMESTAMP_FMT, all_versions
from versioned_hdf5.wrappers import (
    AxisError,
    DatasetWrapper,
    InMemoryArrayDataset,
    InMemoryDataset,
    InMemoryGroup,
    InMemorySparseDataset,
)

pytestmark = pytest.mark.api


TEST_DATA = pathlib.Path(__file__).parent.parent / "test_data"


def test_stage_version(vfile):
    """Test that versions can be staged and are the expected shape."""
    test_data = np.concatenate(
        (
            np.ones((2 * DEFAULT_CHUNK_SIZE,)),
            2 * np.ones((DEFAULT_CHUNK_SIZE,)),
            3 * np.ones((DEFAULT_CHUNK_SIZE,)),
        )
    )

    # Chunk size is intelligently selected based on first dataset written
    chunk_size = guess_chunk(test_data.shape, None, test_data.dtype.itemsize)[0]
    with vfile.stage_version("version1", "") as group:
        group["test_data"] = test_data

    version1 = vfile["version1"]
    assert version1.attrs["prev_version"] == "__first_version__"
    assert_equal(version1["test_data"], test_data)

    ds = vfile.f["/_version_data/test_data/raw_data"]

    # The dataset has 3 different arrays of size chunk_size, so should have
    # shape 3*chunk_size
    assert ds.shape == (3 * chunk_size,)
    assert_equal(ds[0 : 1 * chunk_size], 1.0)
    assert_equal(ds[1 * chunk_size : 2 * chunk_size], 2.0)
    assert_equal(ds[2 * chunk_size : 3 * chunk_size], 3.0)

    with vfile.stage_version("version2", "version1") as group:
        group["test_data"][0] = 0.0

    version2 = vfile["version2"]
    assert version2.attrs["prev_version"] == "version1"
    test_data[0] = 0.0
    assert_equal(version2["test_data"], test_data)

    assert ds.shape == (4 * chunk_size,)
    assert_equal(ds[0 : 1 * chunk_size], 1.0)
    assert_equal(ds[1 * chunk_size : 2 * chunk_size], 2.0)
    assert_equal(ds[2 * chunk_size : 3 * chunk_size], 3.0)
    assert_equal(ds[3 * chunk_size], 0.0)
    assert_equal(ds[3 * chunk_size + 1 : 4 * chunk_size], 1.0)


def test_stage_version_chunk_size(vfile):
    chunk_size = 2**10

    test_data = np.concatenate(
        (
            np.ones((2 * chunk_size,)),
            2 * np.ones((chunk_size,)),
            3 * np.ones((chunk_size,)),
        )
    )

    with vfile.stage_version("version1", "") as group:
        group.create_dataset("test_data", data=test_data, chunks=(chunk_size,))

    with pytest.raises(ValueError), vfile.stage_version("version_bad") as group:
        group.create_dataset("test_data", data=test_data, chunks=(2**9,))

    version1 = vfile["version1"]
    assert version1.attrs["prev_version"] == "__first_version__"
    assert_equal(version1["test_data"], test_data)

    ds = vfile.f["/_version_data/test_data/raw_data"]

    assert ds.shape == (3 * chunk_size,)
    assert_equal(ds[0 : 1 * chunk_size], 1.0)
    assert_equal(ds[1 * chunk_size : 2 * chunk_size], 2.0)
    assert_equal(ds[2 * chunk_size : 3 * chunk_size], 3.0)

    with vfile.stage_version("version2", "version1") as group:
        group["test_data"][0] = 0.0

    version2 = vfile["version2"]
    assert version2.attrs["prev_version"] == "version1"
    test_data[0] = 0.0
    assert_equal(version2["test_data"], test_data)

    assert ds.shape == (4 * chunk_size,)
    assert_equal(ds[0 : 1 * chunk_size], 1.0)
    assert_equal(ds[1 * chunk_size : 2 * chunk_size], 2.0)
    assert_equal(ds[2 * chunk_size : 3 * chunk_size], 3.0)
    assert_equal(ds[3 * chunk_size], 0.0)
    assert_equal(ds[3 * chunk_size + 1 : 4 * chunk_size], 1.0)


def test_stage_version_compression(vfile):
    test_data = np.concatenate(
        (
            np.ones((2 * DEFAULT_CHUNK_SIZE,)),
            2 * np.ones((DEFAULT_CHUNK_SIZE,)),
            3 * np.ones((DEFAULT_CHUNK_SIZE,)),
        )
    )

    with vfile.stage_version("version1", "") as group:
        group.create_dataset(
            "test_data", data=test_data, compression="gzip", compression_opts=3
        )

    with pytest.raises(ValueError), vfile.stage_version("version_bad") as group:
        group.create_dataset("test_data", data=test_data, compression="lzf")

    with pytest.raises(ValueError), vfile.stage_version("version_bad") as group:
        group.create_dataset(
            "test_data", data=test_data, compression="gzip", compression_opts=4
        )

    version1 = vfile["version1"]
    assert version1.attrs["prev_version"] == "__first_version__"
    assert_equal(version1["test_data"], test_data)

    ds = vfile.f["/_version_data/test_data/raw_data"]
    assert ds.compression == "gzip"
    assert ds.compression_opts == 3

    with vfile.stage_version("version2", "version1") as group:
        group["test_data"][0] = 0.0

    version2 = vfile["version2"]
    assert version2.attrs["prev_version"] == "version1"
    test_data[0] = 0.0
    assert_equal(version2["test_data"], test_data)

    assert ds.compression == "gzip"
    assert ds.compression_opts == 3


def test_version_int_slicing(vfile):
    test_data = np.concatenate(
        (
            np.ones((2 * DEFAULT_CHUNK_SIZE,)),
            2 * np.ones((DEFAULT_CHUNK_SIZE,)),
            3 * np.ones((DEFAULT_CHUNK_SIZE,)),
        )
    )

    with vfile.stage_version("version1", "") as group:
        group["test_data"] = test_data

    with vfile.stage_version("version2", "version1") as group:
        group["test_data"][0] = 2.0

    with vfile.stage_version("version3", "version2") as group:
        group["test_data"][0] = 3.0

    with vfile.stage_version("version2_1", "version1", make_current=False) as group:
        group["test_data"][0] = 2.0

    assert vfile[0]["test_data"][0] == 3.0

    with pytest.raises(KeyError):
        vfile["bad"]

    with pytest.raises(IndexError):
        vfile[1]

    assert vfile[-1]["test_data"][0] == 2.0
    assert vfile[-2]["test_data"][0] == 1.0, vfile[-2]
    with pytest.raises(IndexError):
        vfile[-3]

    vfile.current_version = "version2"

    assert vfile[0]["test_data"][0] == 2.0
    assert vfile[-1]["test_data"][0] == 1.0
    with pytest.raises(IndexError):
        vfile[-2]

    vfile.current_version = "version2_1"

    assert vfile[0]["test_data"][0] == 2.0
    assert vfile[-1]["test_data"][0] == 1.0
    with pytest.raises(IndexError):
        vfile[-2]

    vfile.current_version = "version1"

    assert vfile[0]["test_data"][0] == 1.0
    with pytest.raises(IndexError):
        vfile[-1]


def test_version_name_slicing(vfile):
    test_data = np.concatenate(
        (
            np.ones((2 * DEFAULT_CHUNK_SIZE,)),
            2 * np.ones((DEFAULT_CHUNK_SIZE,)),
            3 * np.ones((DEFAULT_CHUNK_SIZE,)),
        )
    )

    with vfile.stage_version("version1", "") as group:
        group["test_data"] = test_data

    with vfile.stage_version("version2", "version1") as group:
        group["test_data"][0] = 2.0

    with vfile.stage_version("version3", "version2") as group:
        group["test_data"][0] = 3.0

    with vfile.stage_version("version2_1", "version1", make_current=False) as group:
        group["test_data"][0] = 2.0

    assert vfile[0]["test_data"][0] == 3.0

    with pytest.raises(IndexError):
        vfile[1]

    assert vfile[-1]["test_data"][0] == 2.0
    assert vfile[-2]["test_data"][0] == 1.0, vfile[-2]

    with pytest.raises(ValueError):
        vfile["/_version_data"]


def test_iter_versions(vfile):
    test_data = np.concatenate(
        (
            np.ones((2 * DEFAULT_CHUNK_SIZE,)),
            2 * np.ones((DEFAULT_CHUNK_SIZE,)),
            3 * np.ones((DEFAULT_CHUNK_SIZE,)),
        )
    )

    with vfile.stage_version("version1", "") as group:
        group["test_data"] = test_data

    with vfile.stage_version("version2", "version1") as group:
        group["test_data"][0] = 2.0

    assert set(vfile) == {"version1", "version2"}

    # __contains__ is implemented from __iter__ automatically
    assert "version1" in vfile
    assert "version2" in vfile
    assert "version3" not in vfile


def test_create_dataset(vfile):
    test_data = np.concatenate(
        (
            np.ones((2 * DEFAULT_CHUNK_SIZE,)),
            2 * np.ones((DEFAULT_CHUNK_SIZE,)),
            3 * np.ones((DEFAULT_CHUNK_SIZE,)),
        )
    )

    with vfile.stage_version("version1", "") as group:
        group.create_dataset("test_data", data=test_data)

    version1 = vfile["version1"]
    assert version1.attrs["prev_version"] == "__first_version__"
    assert_equal(version1["test_data"], test_data)

    with vfile.stage_version("version2") as group:
        group.create_dataset("test_data2", data=test_data)

    ds = vfile.f["/_version_data/test_data/raw_data"]
    assert ds.shape == (3 * DEFAULT_CHUNK_SIZE,)
    assert_equal(ds[0 : 1 * DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(ds[1 * DEFAULT_CHUNK_SIZE : 2 * DEFAULT_CHUNK_SIZE], 2.0)
    assert_equal(ds[2 * DEFAULT_CHUNK_SIZE : 3 * DEFAULT_CHUNK_SIZE], 3.0)

    ds = vfile.f["/_version_data/test_data2/raw_data"]
    assert ds.shape == (3 * DEFAULT_CHUNK_SIZE,)
    assert_equal(ds[0 : 1 * DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(ds[1 * DEFAULT_CHUNK_SIZE : 2 * DEFAULT_CHUNK_SIZE], 2.0)
    assert_equal(ds[2 * DEFAULT_CHUNK_SIZE : 3 * DEFAULT_CHUNK_SIZE], 3.0)

    assert list(vfile.f["/_version_data/versions/__first_version__"]) == []
    assert (
        list(vfile.f["/_version_data/versions/version1"])
        == list(vfile["version1"])
        == ["test_data"]
    )
    assert (
        list(vfile.f["/_version_data/versions/version2"])
        == list(vfile["version2"])
        == ["test_data", "test_data2"]
    )


def test_create_dataset_warns_for_ignored_kwargs(vfile):
    """create_dataset() in versioned_hdf5 has a lot less kwargs
    than the same function in h5py. Test that the extra args are
    accepted, but ignored with a warning.
    """
    match = "parameter is currently ignored for versioned datasets"
    with vfile.stage_version("v1", "") as group:
        with pytest.warns(UserWarning, match=match):
            group.create_dataset("a", data=[1, 2], track_order=True)
        with pytest.warns(UserWarning, match=match):
            group.create_dataset("b", data=[1, 2], track_order=False)
        # None is quietly ignored
        group.create_dataset("c", data=[1, 2], track_order=None)

        # Quietly ignore maxshape if it's a tuple of Nones
        with pytest.warns(UserWarning, match=match):
            group.create_dataset("d", data=[1, 2], maxshape=(3,))
        group.create_dataset("e", data=[1, 2], maxshape=(None,))


def test_changes_dataset(vfile):
    # Testcase similar to those on generate_data.py
    test_data = np.ones((2 * DEFAULT_CHUNK_SIZE,))

    name = "testname"

    with vfile.stage_version("version1", "") as group:
        group.create_dataset(f"{name}/key", data=test_data)
        group.create_dataset(f"{name}/val", data=test_data)

    version1 = vfile["version1"]
    assert version1.attrs["prev_version"] == "__first_version__"
    assert_equal(version1[f"{name}/key"], test_data)
    assert_equal(version1[f"{name}/val"], test_data)

    with vfile.stage_version("version2") as group:
        key_ds = group[f"{name}/key"]
        val_ds = group[f"{name}/val"]
        val_ds[0] = -1
        key_ds[0] = 0

    key = vfile["version2"][f"{name}/key"]
    assert key.shape == (2 * DEFAULT_CHUNK_SIZE,)
    assert_equal(key[0], 0)
    assert_equal(key[1 : 2 * DEFAULT_CHUNK_SIZE], 1.0)

    val = vfile["version2"][f"{name}/val"]
    assert val.shape == (2 * DEFAULT_CHUNK_SIZE,)
    assert_equal(val[0], -1.0)
    assert_equal(val[1 : 2 * DEFAULT_CHUNK_SIZE], 1.0)

    assert list(vfile.f["_version_data/versions/__first_version__"]) == []
    assert (
        list(vfile.f["_version_data/versions/version1"])
        == list(vfile["version1"])
        == [name]
    )
    assert (
        list(vfile.f["_version_data/versions/version2"])
        == list(vfile["version2"])
        == [name]
    )


def test_small_dataset(vfile):
    # Test creating a dataset that is smaller than the chunk size
    data = np.ones((100,))

    with vfile.stage_version("version1") as group:
        group.create_dataset("test", data=data, chunks=(2**14,))

    assert_equal(vfile["version1"]["test"], data)


def test_unmodified(vfile):
    test_data = np.concatenate(
        (
            np.ones((2 * DEFAULT_CHUNK_SIZE,)),
            2 * np.ones((DEFAULT_CHUNK_SIZE,)),
            3 * np.ones((DEFAULT_CHUNK_SIZE,)),
        )
    )

    with vfile.stage_version("version1") as group:
        group.create_dataset("test_data", data=test_data)
        group.create_dataset("test_data2", data=test_data)

    assert set(vfile.f["_version_data/versions/version1"]) == {
        "test_data",
        "test_data2",
    }
    assert set(vfile["version1"]) == {"test_data", "test_data2"}
    assert_equal(vfile["version1"]["test_data"], test_data)
    assert_equal(vfile["version1"]["test_data2"], test_data)
    assert vfile["version1"].datasets().keys() == {"test_data", "test_data2"}

    with vfile.stage_version("version2") as group:
        group["test_data2"][0] = 0.0

    assert set(vfile.f["_version_data/versions/version2"]) == {
        "test_data",
        "test_data2",
    }
    assert set(vfile["version2"]) == {"test_data", "test_data2"}
    assert_equal(vfile["version2"]["test_data"], test_data)
    assert_equal(vfile["version2"]["test_data2"][0], 0.0)
    assert_equal(vfile["version2"]["test_data2"][1:], test_data[1:])


def test_delete_version(vfile):
    test_data = np.concatenate(
        (
            np.ones((2 * DEFAULT_CHUNK_SIZE,)),
            2 * np.ones((DEFAULT_CHUNK_SIZE,)),
            3 * np.ones((DEFAULT_CHUNK_SIZE,)),
        )
    )

    with vfile.stage_version("version1") as group:
        group.create_dataset("test_data", data=test_data)
        group.create_dataset("test_data2", data=test_data)

    with vfile.stage_version("version2") as group:
        del group["test_data2"]

    assert set(vfile.f["_version_data/versions/version2"]) == {"test_data"}
    assert set(vfile["version2"]) == {"test_data"}
    assert_equal(vfile["version2"]["test_data"], test_data)
    assert vfile["version2"].datasets().keys() == {"test_data"}


def test_resize(vfile):
    no_offset_data = np.ones((2 * DEFAULT_CHUNK_SIZE,))

    offset_data = np.concatenate((np.ones((DEFAULT_CHUNK_SIZE,)), np.ones((2,))))

    with vfile.stage_version("version1") as group:
        group.create_dataset("no_offset", data=no_offset_data)
        group.create_dataset("offset", data=offset_data)

    group = vfile["version1"]
    assert group["no_offset"].shape == (2 * DEFAULT_CHUNK_SIZE,)
    assert group["offset"].shape == (DEFAULT_CHUNK_SIZE + 2,)
    assert_equal(group["no_offset"][: 2 * DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(group["offset"][: DEFAULT_CHUNK_SIZE + 2], 1.0)

    # Resize larger, chunk multiple
    with vfile.stage_version("larger_chunk_multiple") as group:
        group["no_offset"].resize((3 * DEFAULT_CHUNK_SIZE,))
        group["offset"].resize((3 * DEFAULT_CHUNK_SIZE,))

    group = vfile["larger_chunk_multiple"]
    assert group["no_offset"].shape == (3 * DEFAULT_CHUNK_SIZE,)
    assert group["offset"].shape == (3 * DEFAULT_CHUNK_SIZE,)
    assert_equal(group["no_offset"][: 2 * DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(group["no_offset"][2 * DEFAULT_CHUNK_SIZE :], 0.0)
    assert_equal(group["offset"][: DEFAULT_CHUNK_SIZE + 2], 1.0)
    assert_equal(group["offset"][DEFAULT_CHUNK_SIZE + 2 :], 0.0)

    # Resize larger, non-chunk multiple
    with vfile.stage_version("larger_chunk_non_multiple", "version1") as group:
        group["no_offset"].resize((3 * DEFAULT_CHUNK_SIZE + 2,))
        group["offset"].resize((3 * DEFAULT_CHUNK_SIZE + 2,))

    group = vfile["larger_chunk_non_multiple"]
    assert group["no_offset"].shape == (3 * DEFAULT_CHUNK_SIZE + 2,)
    assert group["offset"].shape == (3 * DEFAULT_CHUNK_SIZE + 2,)
    assert_equal(group["no_offset"][: 2 * DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(group["no_offset"][2 * DEFAULT_CHUNK_SIZE :], 0.0)
    assert_equal(group["offset"][: DEFAULT_CHUNK_SIZE + 2], 1.0)
    assert_equal(group["offset"][DEFAULT_CHUNK_SIZE + 2 :], 0.0)

    # Resize smaller, chunk multiple
    with vfile.stage_version("smaller_chunk_multiple", "version1") as group:
        group["no_offset"].resize((DEFAULT_CHUNK_SIZE,))
        group["offset"].resize((DEFAULT_CHUNK_SIZE,))

    group = vfile["smaller_chunk_multiple"]
    assert group["no_offset"].shape == (DEFAULT_CHUNK_SIZE,)
    assert group["offset"].shape == (DEFAULT_CHUNK_SIZE,)
    assert_equal(group["no_offset"][:], 1.0)
    assert_equal(group["offset"][:], 1.0)

    # Resize smaller, chunk non-multiple
    with vfile.stage_version("smaller_chunk_non_multiple", "version1") as group:
        group["no_offset"].resize((DEFAULT_CHUNK_SIZE + 2,))
        group["offset"].resize((DEFAULT_CHUNK_SIZE + 2,))

    group = vfile["smaller_chunk_non_multiple"]
    assert group["no_offset"].shape == (DEFAULT_CHUNK_SIZE + 2,)
    assert group["offset"].shape == (DEFAULT_CHUNK_SIZE + 2,)
    assert_equal(group["no_offset"][:], 1.0)
    assert_equal(group["offset"][:], 1.0)

    # Resize after creation
    with vfile.stage_version("version2", "version1") as group:
        # Cover the case where some data is already read in
        group["offset"][-1] = 2.0

        group["no_offset"].resize((3 * DEFAULT_CHUNK_SIZE + 2,))
        group["offset"].resize((3 * DEFAULT_CHUNK_SIZE + 2,))

        assert group["no_offset"].shape == (3 * DEFAULT_CHUNK_SIZE + 2,)
        assert group["offset"].shape == (3 * DEFAULT_CHUNK_SIZE + 2,)
        assert_equal(group["no_offset"][: 2 * DEFAULT_CHUNK_SIZE], 1.0)
        assert_equal(group["no_offset"][2 * DEFAULT_CHUNK_SIZE :], 0.0)
        assert_equal(group["offset"][: DEFAULT_CHUNK_SIZE + 1], 1.0)
        assert_equal(group["offset"][DEFAULT_CHUNK_SIZE + 1], 2.0)
        assert_equal(group["offset"][DEFAULT_CHUNK_SIZE + 2 :], 0.0)

        group["no_offset"].resize((3 * DEFAULT_CHUNK_SIZE,))
        group["offset"].resize((3 * DEFAULT_CHUNK_SIZE,))

        assert group["no_offset"].shape == (3 * DEFAULT_CHUNK_SIZE,)
        assert group["offset"].shape == (3 * DEFAULT_CHUNK_SIZE,)
        assert_equal(group["no_offset"][: 2 * DEFAULT_CHUNK_SIZE], 1.0)
        assert_equal(group["no_offset"][2 * DEFAULT_CHUNK_SIZE :], 0.0)
        assert_equal(group["offset"][: DEFAULT_CHUNK_SIZE + 1], 1.0)
        assert_equal(group["offset"][DEFAULT_CHUNK_SIZE + 1], 2.0)
        assert_equal(group["offset"][DEFAULT_CHUNK_SIZE + 2 :], 0.0)

        group["no_offset"].resize((DEFAULT_CHUNK_SIZE + 2,))
        group["offset"].resize((DEFAULT_CHUNK_SIZE + 2,))

        assert group["no_offset"].shape == (DEFAULT_CHUNK_SIZE + 2,)
        assert group["offset"].shape == (DEFAULT_CHUNK_SIZE + 2,)
        assert_equal(group["no_offset"][:], 1.0)
        assert_equal(group["offset"][: DEFAULT_CHUNK_SIZE + 1], 1.0)
        assert_equal(group["offset"][DEFAULT_CHUNK_SIZE + 1], 2.0)

        group["no_offset"].resize((DEFAULT_CHUNK_SIZE,))
        group["offset"].resize((DEFAULT_CHUNK_SIZE,))

        assert group["no_offset"].shape == (DEFAULT_CHUNK_SIZE,)
        assert group["offset"].shape == (DEFAULT_CHUNK_SIZE,)
        assert_equal(group["no_offset"][:], 1.0)
        assert_equal(group["offset"][:], 1.0)

    group = vfile["version2"]
    assert group["no_offset"].shape == (DEFAULT_CHUNK_SIZE,)
    assert group["offset"].shape == (DEFAULT_CHUNK_SIZE,)
    assert_equal(group["no_offset"][:], 1.0)
    assert_equal(group["offset"][:], 1.0)

    # Resize smaller than a chunk
    small_data = np.array([1, 2, 3])

    with vfile.stage_version("version1_small", "") as group:
        group.create_dataset("small", data=small_data)

    with vfile.stage_version("version2_small", "version1_small") as group:
        group["small"].resize((5,))
        assert_equal(group["small"], np.array([1, 2, 3, 0, 0]))
        group["small"][3:] = np.array([4, 5])
        assert_equal(group["small"], np.array([1, 2, 3, 4, 5]))

    group = vfile["version1_small"]
    assert_equal(group["small"], np.array([1, 2, 3]))
    group = vfile["version2_small"]
    assert_equal(group["small"], np.array([1, 2, 3, 4, 5]))

    # Resize after calling create_dataset, larger
    with vfile.stage_version("resize_after_create_larger", "") as group:
        group.create_dataset("data", data=offset_data)
        group["data"].resize((DEFAULT_CHUNK_SIZE + 4,))

        assert group["data"].shape == (DEFAULT_CHUNK_SIZE + 4,)
        assert_equal(group["data"][: DEFAULT_CHUNK_SIZE + 2], 1.0)
        assert_equal(group["data"][DEFAULT_CHUNK_SIZE + 2 :], 0.0)

    group = vfile["resize_after_create_larger"]
    assert group["data"].shape == (DEFAULT_CHUNK_SIZE + 4,)
    assert_equal(group["data"][: DEFAULT_CHUNK_SIZE + 2], 1.0)
    assert_equal(group["data"][DEFAULT_CHUNK_SIZE + 2 :], 0.0)

    # Resize after calling create_dataset, smaller
    with vfile.stage_version("resize_after_create_smaller", "") as group:
        group.create_dataset("data", data=offset_data)
        group["data"].resize((DEFAULT_CHUNK_SIZE - 4,))

        assert group["data"].shape == (DEFAULT_CHUNK_SIZE - 4,)
        assert_equal(group["data"][:], 1.0)

    group = vfile["resize_after_create_smaller"]
    assert group["data"].shape == (DEFAULT_CHUNK_SIZE - 4,)
    assert_equal(group["data"][:], 1.0)


def test_resize_unaligned(vfile):
    ds_name = "test_resize_unaligned"
    with vfile.stage_version("0") as group:
        group.create_dataset(ds_name, data=np.arange(1000))

    for i in range(1, 10):
        with vfile.stage_version(str(i)) as group:
            len_ = len(group[ds_name])
            assert_equal(group[ds_name][:], np.arange(i * 1000))
            group[ds_name].resize((len_ + 1000,))
            group[ds_name][-1000:] = np.arange(len_, len_ + 1000)
            assert_equal(group[ds_name][:], np.arange((i + 1) * 1000))


@pytest.mark.slow
def test_resize_multiple_dimensions(vfile):
    # Test semantics against raw HDF5
    shapes = range(5, 25, 5)  # 5, 10, 15, 20
    chunks = (10, 10, 10)
    for i, (oldshape, newshape) in enumerate(
        itertools.combinations_with_replacement(itertools.product(shapes, repeat=3), 2)
    ):
        data = np.arange(np.prod(oldshape)).reshape(oldshape)
        # Get the ground truth from h5py
        vfile.f.create_dataset(
            f"data{i}",
            data=data,
            fillvalue=-1,
            chunks=chunks,
            maxshape=(None, None, None),
        )
        vfile.f[f"data{i}"].resize(newshape)
        new_data = vfile.f[f"data{i}"][()]

        # resize after creation
        with vfile.stage_version(f"version1_{i}") as group:
            group.create_dataset(
                f"dataset1_{i}", data=data, chunks=chunks, fillvalue=-1
            )
            group[f"dataset1_{i}"].resize(newshape)
            assert group[f"dataset1_{i}"].shape == newshape
            assert_equal(group[f"dataset1_{i}"][()], new_data)

        version1 = vfile[f"version1_{i}"]
        assert version1[f"dataset1_{i}"].shape == newshape
        assert_equal(version1[f"dataset1_{i}"][()], new_data)

        # resize in a new version
        with vfile.stage_version(f"version2_1_{i}", "") as group:
            group.create_dataset(
                f"dataset2_{i}", data=data, chunks=chunks, fillvalue=-1
            )
        with vfile.stage_version(f"version2_2_{i}", f"version2_1_{i}") as group:
            group[f"dataset2_{i}"].resize(newshape)
            assert group[f"dataset2_{i}"].shape == newshape
            assert_equal(
                group[f"dataset2_{i}"][()], new_data, str((oldshape, newshape))
            )

        version2_2 = vfile[f"version2_2_{i}"]
        assert version2_2[f"dataset2_{i}"].shape == newshape
        assert_equal(version2_2[f"dataset2_{i}"][()], new_data)

        # resize after some data is read in
        with vfile.stage_version(f"version3_1_{i}", "") as group:
            group.create_dataset(
                f"dataset3_{i}", data=data, chunks=chunks, fillvalue=-1
            )
        with vfile.stage_version(f"version3_2_{i}", f"version3_1_{i}") as group:
            # read in first and last chunks
            group[f"dataset3_{i}"][0, 0, 0]
            group[f"dataset3_{i}"][-1, -1, -1]
            group[f"dataset3_{i}"].resize(newshape)
            assert group[f"dataset3_{i}"].shape == newshape
            assert_equal(group[f"dataset3_{i}"][()], new_data)

        version3_2 = vfile[f"version3_2_{i}"]
        assert version3_2[f"dataset3_{i}"].shape == newshape
        assert_equal(version3_2[f"dataset3_{i}"][()], new_data)


def test_resize_int_types_shrink(tmp_path):
    # Test that resizing with numpy int types works
    with h5py.File(tmp_path / "data.h5", "w") as f:
        vfile = VersionedHDF5File(f)
        with vfile.stage_version("r0") as sv:
            sv.create_dataset(
                "values", data=np.arange(17), maxshape=(None,), chunks=(3,)
            )
    # Test shrinking:
    with h5py.File(tmp_path / "data.h5", "r+") as f:
        vfile = VersionedHDF5File(f)
        with vfile.stage_version("r1") as sv:
            old_size = sv["values"].size
            new_size = old_size - 4
            sv["values"].resize((new_size,))


def test_resize_int_types_grow(tmp_path):
    filename = tmp_path / "data.h5"
    # Test that resizing with numpy int types works
    with h5py.File(filename, "w") as f:
        vfile = VersionedHDF5File(f)
        with vfile.stage_version("r0") as sv:
            sv.create_dataset(
                "values", data=np.arange(17), maxshape=(None,), chunks=(3,)
            )
    # Test growing repeatedly:
    with h5py.File(filename, "r+") as f:
        vfile = VersionedHDF5File(f)
        with vfile.stage_version("r1") as sv:
            old_size = sv["values"].size
            new_size = old_size + 7
            sv["values"].resize((new_size,))
            sv["values"][old_size:new_size] = np.arange(new_size - old_size)
            old_size = sv["values"].size
            new_size = old_size + 7
            sv["values"].resize((new_size,))
            sv["values"][old_size:new_size] = np.arange(new_size - old_size)


def test_resize_sparse(vfile):
    with vfile.stage_version("version1") as group:
        ds = group.create_dataset("data", shape=(5, 5), chunks=(3, 3))
        ds[0, 0] = 1
        ds[4, 4] = 2

        ds.resize((6, 6))
        assert ds.shape == (6, 6)
        expect = np.zeros((6, 6))
        expect[0, 0] = 1
        expect[4, 4] = 2
        assert_equal(ds[:], expect)

        ds.resize((4, 3))
        assert ds.shape == (4, 3)
        expect = expect[:4, :3]
        assert_equal(ds[:], expect)

    ds = vfile["version1"]["data"]
    assert ds.shape == (4, 3)
    assert_equal(ds[:], expect)


def test_resize_axis(vfile):
    # test axis= parameter of resize
    with vfile.stage_version("v0") as sv:
        ds = sv.create_dataset("x", shape=(5, 5), chunks=(3, 3), maxshape=(None, None))
        ds.resize(6, axis=0)
        assert ds.shape == (6, 5)
        ds.resize(4, axis=1)
        assert ds.shape == (6, 4)
        ds.resize(7, axis=-2)
        assert ds.shape == (7, 4)
        ds.resize(np.int32(8), axis=0)
        assert ds.shape == (8, 4)
        assert type(ds.shape[0]) is int
        ds.resize(9.0, axis=0)
        assert ds.shape == (9, 4)
        assert type(ds.shape[0]) is int

        with pytest.raises(AxisError):
            ds.resize(8, axis=2)
        with pytest.raises(AxisError):
            ds.resize(8, axis=-3)
        with pytest.raises(TypeError, match="must be an integer"):
            ds.resize((2, 2), axis=0)


@pytest.mark.parametrize(
    "size",
    [
        (6,),
        [6],
        6,
        (np.int8(6),),
        np.int8(6),
        np.asarray([6]),
        np.asarray(6),
    ],
)
def test_resize_weird_size(vfile, size):
    with vfile.stage_version("v0") as sv:
        # InMemorySparseDataset
        ds = sv.create_dataset("x", shape=(5,), chunks=(3,), maxshape=(None,))
        ds.resize(size)
        assert ds.shape == (6,)
        assert type(ds.shape[0]) is int

        # InMemoryArrayDataset
        ds = sv.create_dataset("y", data=np.arange(5), chunks=(3,), maxshape=(None,))
        ds.resize(size)
        assert ds.shape == (6,)
        assert type(ds.shape[0]) is int

    with vfile.stage_version("v1") as sv:
        # InMemoryDataset
        ds = sv["x"]
        ds.resize(size)
        assert ds.shape == (6,)
        assert type(ds.shape[0]) is int


def test_getitem(vfile):
    data = np.arange(2 * DEFAULT_CHUNK_SIZE)

    with vfile.stage_version("version1") as group:
        group.create_dataset("test_data", data=data)

        test_data = group["test_data"]
        assert test_data.shape == (2 * DEFAULT_CHUNK_SIZE,)
        assert_equal(test_data[0], 0)
        assert test_data[0].dtype == np.int64
        assert_equal(test_data[:], data)
        assert_equal(
            test_data[: DEFAULT_CHUNK_SIZE + 1], data[: DEFAULT_CHUNK_SIZE + 1]
        )

    with vfile.stage_version("version2") as group:
        test_data = group["test_data"]
        assert test_data.shape == (2 * DEFAULT_CHUNK_SIZE,)
        assert_equal(test_data[0], 0)
        assert test_data[0].dtype == np.int64
        assert_equal(test_data[:], data)
        assert_equal(
            test_data[: DEFAULT_CHUNK_SIZE + 1], data[: DEFAULT_CHUNK_SIZE + 1]
        )


def test_timestamp_auto(vfile):
    data = np.ones((2 * DEFAULT_CHUNK_SIZE,))

    with vfile.stage_version("version1") as group:
        group.create_dataset("test_data", data=data)

    assert isinstance(vfile["version1"].attrs["timestamp"], str)


def test_timestamp_manual(vfile):
    data1 = np.ones(2 * DEFAULT_CHUNK_SIZE)
    data2 = np.ones(3 * DEFAULT_CHUNK_SIZE)

    ts1 = datetime.datetime(2020, 6, 29, 20, 12, 56, tzinfo=datetime.timezone.utc)
    ts2 = datetime.datetime(2020, 6, 29, 22, 12, 56)
    with vfile.stage_version("version1", timestamp=ts1) as group:
        group["test_data_1"] = data1

    assert vfile["version1"].attrs["timestamp"] == ts1.strftime(TIMESTAMP_FMT)

    with (
        pytest.raises(ValueError),
        vfile.stage_version("version2", timestamp=ts2) as group,
    ):
        group["test_data_2"] = data2

    with (
        pytest.raises(TypeError),
        vfile.stage_version("version3", timestamp="2020-6-29") as group,
    ):
        group["test_data_3"] = data1


def test_timestamp_manual_datetime64(vfile):
    data = np.ones((2 * DEFAULT_CHUNK_SIZE,))

    # Also tests that it works correctly for 0 fractional part (issue #190).
    ts = datetime.datetime(2020, 6, 29, 20, 12, 56, tzinfo=datetime.timezone.utc)
    npts = np.datetime64(ts.replace(tzinfo=None))

    with vfile.stage_version("version1", timestamp=npts) as group:
        group["test_data"] = data

    v1 = vfile["version1"]

    assert v1.attrs["timestamp"] == ts.strftime(TIMESTAMP_FMT)

    assert vfile[npts] == v1
    assert vfile[ts] == v1
    assert vfile.get_version_by_timestamp(npts, exact=True) == v1
    assert vfile.get_version_by_timestamp(ts, exact=True) == v1


def test_getitem_by_timestamp(vfile):
    data = np.arange(2 * DEFAULT_CHUNK_SIZE)

    with vfile.stage_version("version1") as group:
        group.create_dataset("test_data", data=data)

    v1 = vfile["version1"]
    ts1 = datetime.datetime.strptime(v1.attrs["timestamp"], TIMESTAMP_FMT)
    assert vfile[ts1] == v1
    assert vfile.get_version_by_timestamp(ts1) == v1
    assert vfile.get_version_by_timestamp(ts1, exact=True) == v1

    dt1 = np.datetime64(ts1.replace(tzinfo=None))
    assert vfile[dt1] == v1
    assert vfile.get_version_by_timestamp(dt1) == v1
    assert vfile.get_version_by_timestamp(dt1, exact=True) == v1

    minute = datetime.timedelta(minutes=1)
    second = datetime.timedelta(seconds=1)

    ts2 = ts1 + minute
    dt2 = np.datetime64(ts2.replace(tzinfo=None))

    with vfile.stage_version("version2", timestamp=ts2) as group:
        group["test_data"][0] += 1

    v2 = vfile["version2"]
    assert vfile[ts2] == v2
    assert vfile.get_version_by_timestamp(ts2) == v2
    assert vfile.get_version_by_timestamp(ts2, exact=True) == v2

    assert vfile[dt2] == v2
    assert vfile.get_version_by_timestamp(dt2) == v2
    assert vfile.get_version_by_timestamp(dt2, exact=True) == v2

    ts2_1 = ts2 + second
    dt2_1 = np.datetime64(ts2_1.replace(tzinfo=None))

    assert vfile[ts2_1] == v2
    assert vfile.get_version_by_timestamp(ts2_1) == v2
    with pytest.raises(KeyError):
        vfile.get_version_by_timestamp(ts2_1, exact=True)

    assert vfile[dt2_1] == v2
    assert vfile.get_version_by_timestamp(dt2_1) == v2
    with pytest.raises(KeyError):
        vfile.get_version_by_timestamp(dt2_1, exact=True)

    ts1_1 = ts1 + second
    dt1_1 = np.datetime64(ts1_1.replace(tzinfo=None))

    assert vfile[ts1_1] == v1
    assert vfile.get_version_by_timestamp(ts1_1) == v1
    with pytest.raises(KeyError):
        vfile.get_version_by_timestamp(ts1_1, exact=True)

    assert vfile[dt1_1] == v1
    assert vfile.get_version_by_timestamp(dt1_1) == v1
    with pytest.raises(KeyError):
        vfile.get_version_by_timestamp(dt1_1, exact=True)

    ts0 = ts1 - second
    dt0 = np.datetime64(ts0.replace(tzinfo=None))

    with pytest.raises(KeyError):
        vfile[ts0]
    with pytest.raises(KeyError):
        vfile.get_version_by_timestamp(ts0)
    with pytest.raises(KeyError):
        vfile.get_version_by_timestamp(ts0, exact=True)
    with pytest.raises(KeyError):
        vfile[dt0]
    with pytest.raises(KeyError):
        vfile.get_version_by_timestamp(dt0)
    with pytest.raises(KeyError):
        vfile.get_version_by_timestamp(dt0, exact=True)


def test_nonroot(vfile):
    g = vfile.f.create_group("subgroup")
    file = VersionedHDF5File(g)

    test_data = np.concatenate(
        (
            np.ones((2 * DEFAULT_CHUNK_SIZE,)),
            2 * np.ones((DEFAULT_CHUNK_SIZE,)),
            3 * np.ones((DEFAULT_CHUNK_SIZE,)),
        )
    )

    chunk_size = guess_chunk(test_data.shape, None, test_data.dtype.itemsize)[0]
    with file.stage_version("version1", "") as group:
        group["test_data"] = test_data

    version1 = file["version1"]
    assert version1.attrs["prev_version"] == "__first_version__"
    assert_equal(version1["test_data"], test_data)

    ds = vfile.f["/subgroup/_version_data/test_data/raw_data"]

    assert ds.shape == (3 * chunk_size,)
    assert_equal(ds[0 : 1 * chunk_size], 1.0)
    assert_equal(ds[1 * chunk_size : 2 * chunk_size], 2.0)
    assert_equal(ds[2 * chunk_size : 3 * chunk_size], 3.0)


def test_attrs(vfile):
    data = np.arange(2 * DEFAULT_CHUNK_SIZE)

    with vfile.stage_version("version1") as group:
        group.create_dataset("test_data", data=data)

        test_data = group["test_data"]
        assert "test_attr" not in test_data.attrs
        test_data.attrs["test_attr"] = 0

    assert (
        vfile["version1"]["test_data"].attrs["test_attr"]
        == vfile.f["_version_data"]["versions"]["version1"]["test_data"].attrs[
            "test_attr"
        ]
        == 0
    )

    with vfile.stage_version("version2") as group:
        test_data = group["test_data"]
        assert test_data.attrs["test_attr"] == 0
        test_data.attrs["test_attr"] = 1

    assert (
        vfile["version1"]["test_data"].attrs["test_attr"]
        == vfile.f["_version_data"]["versions"]["version1"]["test_data"].attrs[
            "test_attr"
        ]
        == 0
    )

    assert (
        vfile["version2"]["test_data"].attrs["test_attr"]
        == vfile.f["_version_data"]["versions"]["version2"]["test_data"].attrs[
            "test_attr"
        ]
        == 1
    )


def test_auto_delete(vfile):
    try:
        with vfile.stage_version("version1") as group:
            raise RuntimeError
    except RuntimeError:
        pass
    else:
        raise AssertionError("did not raise")

    # Make sure the version got deleted so that we can make it again
    data = np.arange(2 * DEFAULT_CHUNK_SIZE)

    with vfile.stage_version("version1") as group:
        group.create_dataset("test_data", data=data)

    assert_equal(vfile["version1"]["test_data"], data)


def test_delitem(vfile):
    data = np.arange(2 * DEFAULT_CHUNK_SIZE)

    with vfile.stage_version("version1") as group:
        group.create_dataset("test_data", data=data)

    with vfile.stage_version("version2") as group:
        group.create_dataset("test_data2", data=data)

    del vfile["version2"]

    assert list(vfile) == ["version1"]
    assert vfile.current_version == "version1"

    with pytest.raises(KeyError):
        del vfile["version2"]

    del vfile["version1"]

    assert list(vfile) == []
    assert vfile.current_version == "__first_version__"


def test_groups(vfile):
    data = np.ones(2 * DEFAULT_CHUNK_SIZE)

    with vfile.stage_version("version1") as group:
        group.create_group("group1")
        group.create_dataset("group1/test_data", data=data)
        assert_equal(group["group1"]["test_data"], data)
        assert_equal(group["group1/test_data"], data)

    version = vfile["version1"]
    assert_equal(version["group1"]["test_data"], data)
    assert_equal(version["group1/test_data"], data)

    with vfile.stage_version("version2", "") as group:
        group.create_dataset("group1/test_data", data=data)
        assert_equal(group["group1"]["test_data"], data)
        assert_equal(group["group1/test_data"], data)

    version = vfile["version2"]
    assert_equal(version["group1"]["test_data"], data)
    assert_equal(version["group1/test_data"], data)

    with vfile.stage_version("version3", "version1") as group:
        group["group1"]["test_data"][0] = 0
        group["group1/test_data"][1] = 0

        assert_equal(group["group1"]["test_data"][:2], 0)
        assert_equal(group["group1"]["test_data"][2:], 1)

        assert_equal(group["group1/test_data"][:2], 0)
        assert_equal(group["group1/test_data"][2:], 1)

    version = vfile["version3"]
    assert_equal(version["group1"]["test_data"][:2], 0)
    assert_equal(version["group1"]["test_data"][2:], 1)

    assert_equal(version["group1/test_data"][:2], 0)
    assert_equal(version["group1/test_data"][2:], 1)

    assert list(version) == ["group1"]
    assert list(version["group1"]) == ["test_data"]

    with vfile.stage_version("version4", "version3") as group:
        group.create_dataset("group2/test_data", data=2 * data)

        assert_equal(group["group1"]["test_data"][:2], 0)
        assert_equal(group["group1"]["test_data"][2:], 1)
        assert_equal(group["group2"]["test_data"][:], 2)

        assert_equal(group["group1/test_data"][:2], 0)
        assert_equal(group["group1/test_data"][2:], 1)
        assert_equal(group["group2/test_data"][:], 2)

    version = vfile["version4"]
    assert_equal(version["group1"]["test_data"][:2], 0)
    assert_equal(version["group1"]["test_data"][2:], 1)
    assert_equal(group["group2"]["test_data"][:], 2)

    assert_equal(version["group1/test_data"][:2], 0)
    assert_equal(version["group1/test_data"][2:], 1)
    assert_equal(group["group2/test_data"][:], 2)

    assert list(version) == ["group1", "group2"]
    assert list(version["group1"]) == ["test_data"]
    assert list(version["group2"]) == ["test_data"]

    with vfile.stage_version("version5", "") as group:
        group.create_dataset("group1/group2/test_data", data=data)
        assert_equal(group["group1"]["group2"]["test_data"], data)
        assert_equal(group["group1/group2"]["test_data"], data)
        assert_equal(group["group1"]["group2/test_data"], data)
        assert_equal(group["group1/group2/test_data"], data)

    version = vfile["version5"]
    assert_equal(version["group1"]["group2"]["test_data"], data)
    assert_equal(version["group1/group2"]["test_data"], data)
    assert_equal(version["group1"]["group2/test_data"], data)
    assert_equal(version["group1/group2/test_data"], data)

    with vfile.stage_version("version6", "") as group:
        group.create_dataset("group1/test_data1", data=data)
        group.create_dataset("group1/group2/test_data2", data=2 * data)
        group.create_dataset("group1/group2/group3/test_data3", data=3 * data)
        group.create_dataset("group1/group2/test_data4", data=4 * data)

        assert_equal(group["group1"]["test_data1"], data)
        assert_equal(group["group1/test_data1"], data)

        assert_equal(group["group1"]["group2"]["test_data2"], 2 * data)
        assert_equal(group["group1/group2"]["test_data2"], 2 * data)
        assert_equal(group["group1"]["group2/test_data2"], 2 * data)
        assert_equal(group["group1/group2/test_data2"], 2 * data)

        assert_equal(group["group1"]["group2"]["group3"]["test_data3"], 3 * data)
        assert_equal(group["group1/group2"]["group3"]["test_data3"], 3 * data)
        assert_equal(group["group1/group2"]["group3/test_data3"], 3 * data)
        assert_equal(group["group1"]["group2/group3/test_data3"], 3 * data)
        assert_equal(group["group1/group2/group3/test_data3"], 3 * data)

        assert_equal(group["group1"]["group2"]["test_data4"], 4 * data)
        assert_equal(group["group1/group2"]["test_data4"], 4 * data)
        assert_equal(group["group1"]["group2/test_data4"], 4 * data)
        assert_equal(group["group1/group2/test_data4"], 4 * data)

        assert list(group) == ["group1"]
        assert set(group["group1"]) == {"group2", "test_data1"}
        assert (
            set(group["group1"]["group2"])
            == set(group["group1/group2"])
            == {"group3", "test_data2", "test_data4"}
        )
        assert (
            list(group["group1"]["group2"]["group3"])
            == list(group["group1/group2/group3"])
            == ["test_data3"]
        )

    version = vfile["version6"]
    assert_equal(version["group1"]["test_data1"], data)
    assert_equal(version["group1/test_data1"], data)

    assert_equal(version["group1"]["group2"]["test_data2"], 2 * data)
    assert_equal(version["group1/group2"]["test_data2"], 2 * data)
    assert_equal(version["group1"]["group2/test_data2"], 2 * data)
    assert_equal(version["group1/group2/test_data2"], 2 * data)

    assert_equal(version["group1"]["group2"]["group3"]["test_data3"], 3 * data)
    assert_equal(version["group1/group2"]["group3"]["test_data3"], 3 * data)
    assert_equal(version["group1/group2"]["group3/test_data3"], 3 * data)
    assert_equal(version["group1"]["group2/group3/test_data3"], 3 * data)
    assert_equal(version["group1/group2/group3/test_data3"], 3 * data)

    assert_equal(version["group1"]["group2"]["test_data4"], 4 * data)
    assert_equal(version["group1/group2"]["test_data4"], 4 * data)
    assert_equal(version["group1"]["group2/test_data4"], 4 * data)
    assert_equal(version["group1/group2/test_data4"], 4 * data)

    assert list(version) == ["group1"]
    assert set(version["group1"]) == {"group2", "test_data1"}
    assert (
        set(version["group1"]["group2"])
        == set(version["group1/group2"])
        == {"group3", "test_data2", "test_data4"}
    )
    assert (
        list(version["group1"]["group2"]["group3"])
        == list(version["group1/group2/group3"])
        == ["test_data3"]
    )

    with vfile.stage_version("version-bad", "") as group:
        with pytest.raises(ValueError):
            group.create_dataset("/group1/test_data", data=data)
        with pytest.raises(ValueError):
            group.create_group("/group1")


def test_group_contains(vfile):
    data = np.ones(2 * DEFAULT_CHUNK_SIZE)

    with vfile.stage_version("version1") as group:
        group.create_dataset("group1/group2/test_data", data=data)
        assert "group1" in group
        assert "group2" in group["group1"]
        assert "test_data" in group["group1/group2"]
        assert "test_data" not in group
        assert "test_data" not in group["group1"]
        assert "group1/group2" in group
        assert "group1/group3" not in group
        assert "group1/group2/test_data" in group
        assert "group1/group3/test_data" not in group
        assert "group1/group3/test_data2" not in group

    with vfile.stage_version("version2") as group:
        group.create_dataset("group1/group3/test_data2", data=data)
        assert "group1" in group
        assert "group2" in group["group1"]
        assert "group3" in group["group1"]
        assert "test_data" in group["group1/group2"]
        assert "test_data" not in group
        assert "test_data" not in group["group1"]
        assert "test_data2" in group["group1/group3"]
        assert "test_data2" not in group["group1/group2"]
        assert "group1/group2" in group
        assert "group1/group3" in group
        assert "group1/group2/test_data" in group
        assert "group1/group3/test_data" not in group
        assert "group1/group3/test_data2" in group

    version1 = vfile["version1"]
    version2 = vfile["version2"]
    assert "group1" in version1
    assert "group1/" in version1
    assert "group1" in version2
    assert "group1/" in version2
    assert "group2" in version1["group1"]
    assert "group2/" in version1["group1"]
    assert "group2" in version2["group1"]
    assert "group2/" in version2["group1"]
    assert "group3" not in version1["group1"]
    assert "group3/" not in version1["group1"]
    assert "group3" in version2["group1"]
    assert "group3/" in version2["group1"]
    assert "group1/group2" in version1
    assert "group1/group2/" in version1
    assert "group1/group2" in version2
    assert "group1/group2/" in version2
    assert "group1/group3" not in version1
    assert "group1/group3/" not in version1
    assert "group1/group3" in version2
    assert "group1/group3/" in version2
    assert "group1/group2/test_data" in version1
    assert "group1/group2/test_data/" in version1
    assert "group1/group2/test_data" in version2
    assert "group1/group2/test_data/" in version2
    assert "group1/group3/test_data" not in version1
    assert "group1/group3/test_data/" not in version1
    assert "group1/group3/test_data" not in version2
    assert "group1/group3/test_data/" not in version2
    assert "group1/group3/test_data2" not in version1
    assert "group1/group3/test_data2/" not in version1
    assert "group1/group3/test_data2" in version2
    assert "group1/group3/test_data2/" in version2
    assert "test_data" in version1["group1/group2"]
    assert "test_data" in version2["group1/group2"]
    assert "test_data" not in version1
    assert "test_data" not in version2
    assert "test_data" not in version1["group1"]
    assert "test_data" not in version2["group1"]
    assert "test_data2" in version2["group1/group3"]
    assert "test_data2" not in version1["group1/group2"]
    assert "test_data2" not in version2["group1/group2"]

    assert "/_version_data/versions/version1/" in version1
    assert "/_version_data/versions/version1" in version1
    assert "/_version_data/versions/version1/" not in version2
    assert "/_version_data/versions/version1" not in version2
    assert "/_version_data/versions/version1/group1" in version1
    assert "/_version_data/versions/version1/group1" not in version2
    assert "/_version_data/versions/version1/group1/group2" in version1
    assert "/_version_data/versions/version1/group1/group2" not in version2


def test_moved_file(setup_vfile):
    h5file = setup_vfile()
    path = pathlib.Path(h5file.filename)

    # See issue #28. Make sure the virtual datasets do not hard-code the filename.
    file = VersionedHDF5File(h5file)
    data = np.ones(2 * DEFAULT_CHUNK_SIZE)
    with file.stage_version("version1") as group:
        group["dataset"] = data
    file.close()

    with h5py.File(h5file.filename, "r") as f:
        file = VersionedHDF5File(f)
        assert_equal(file["version1"]["dataset"][:], data)
        file.close()

    new_path = path.parent / "test2.hdf5"
    os.rename(path, new_path)

    with h5py.File(new_path, "r") as f:
        file = VersionedHDF5File(f)
        assert_equal(file["version1"]["dataset"][:], data)
        file.close()


def test_list_assign(vfile):
    data = [1, 2, 3]

    with vfile.stage_version("version1") as group:
        group["dataset"] = data

        assert_equal(group["dataset"][:], data)

    assert_equal(vfile["version1"]["dataset"][:], data)


def test_nested_group(vfile):
    # Issue #66
    data1 = np.array([1, 1])
    data2 = np.array([2, 2])

    with vfile.stage_version("1") as sv:
        sv.create_dataset("bar/baz", data=data1)
        assert_equal(sv["bar/baz"][:], data1)

    assert_equal(sv["bar/baz"][:], data1)

    with vfile.stage_version("2") as sv:
        sv.create_dataset("bar/bon/1/data/axes/date", data=data2)
        assert_equal(sv["bar/baz"][:], data1)
        assert_equal(sv["bar/bon/1/data/axes/date"][:], data2)

    version1 = vfile["1"]
    version2 = vfile["2"]
    assert_equal(version1["bar/baz"][:], data1)
    assert_equal(version2["bar/baz"][:], data1)
    assert "bar/bon/1/data/axes/date" not in version1
    assert_equal(version2["bar/bon/1/data/axes/date"][:], data2)


def test_fillvalue(vfile):
    # Based on test_resize(), but only the resize largers that use the fill
    # value
    fillvalue = 5.0

    no_offset_data = np.ones((2 * DEFAULT_CHUNK_SIZE,))

    offset_data = np.concatenate((np.ones((DEFAULT_CHUNK_SIZE,)), np.ones((2,))))

    with vfile.stage_version("version1") as group:
        group.create_dataset("no_offset", data=no_offset_data, fillvalue=fillvalue)
        group.create_dataset("offset", data=offset_data, fillvalue=fillvalue)

    group = vfile["version1"]
    assert group["no_offset"].shape == (2 * DEFAULT_CHUNK_SIZE,)
    assert group["offset"].shape == (DEFAULT_CHUNK_SIZE + 2,)
    assert_equal(group["no_offset"][: 2 * DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(group["offset"][: DEFAULT_CHUNK_SIZE + 2], 1.0)

    # Resize larger, chunk multiple
    with vfile.stage_version("larger_chunk_multiple") as group:
        group["no_offset"].resize((3 * DEFAULT_CHUNK_SIZE,))
        group["offset"].resize((3 * DEFAULT_CHUNK_SIZE,))

    group = vfile["larger_chunk_multiple"]
    assert group["no_offset"].shape == (3 * DEFAULT_CHUNK_SIZE,)
    assert group["offset"].shape == (3 * DEFAULT_CHUNK_SIZE,)
    assert_equal(group["no_offset"][: 2 * DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(group["no_offset"][2 * DEFAULT_CHUNK_SIZE :], fillvalue)
    assert_equal(group["offset"][: DEFAULT_CHUNK_SIZE + 2], 1.0)
    assert_equal(group["offset"][DEFAULT_CHUNK_SIZE + 2 :], fillvalue)

    # Resize larger, non-chunk multiple
    with vfile.stage_version("larger_chunk_non_multiple", "version1") as group:
        group["no_offset"].resize((3 * DEFAULT_CHUNK_SIZE + 2,))
        group["offset"].resize((3 * DEFAULT_CHUNK_SIZE + 2,))

    group = vfile["larger_chunk_non_multiple"]
    assert group["no_offset"].shape == (3 * DEFAULT_CHUNK_SIZE + 2,)
    assert group["offset"].shape == (3 * DEFAULT_CHUNK_SIZE + 2,)
    assert_equal(group["no_offset"][: 2 * DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(group["no_offset"][2 * DEFAULT_CHUNK_SIZE :], fillvalue)
    assert_equal(group["offset"][: DEFAULT_CHUNK_SIZE + 2], 1.0)
    assert_equal(group["offset"][DEFAULT_CHUNK_SIZE + 2 :], fillvalue)

    # Resize after creation
    with vfile.stage_version("version2", "version1") as group:
        # Cover the case where some data is already read in
        group["offset"][-1] = 2.0

        group["no_offset"].resize((3 * DEFAULT_CHUNK_SIZE + 2,))
        group["offset"].resize((3 * DEFAULT_CHUNK_SIZE + 2,))

        assert group["no_offset"].shape == (3 * DEFAULT_CHUNK_SIZE + 2,)
        assert group["offset"].shape == (3 * DEFAULT_CHUNK_SIZE + 2,)
        assert_equal(group["no_offset"][: 2 * DEFAULT_CHUNK_SIZE], 1.0)
        assert_equal(group["no_offset"][2 * DEFAULT_CHUNK_SIZE :], fillvalue)
        assert_equal(group["offset"][: DEFAULT_CHUNK_SIZE + 1], 1.0)
        assert_equal(group["offset"][DEFAULT_CHUNK_SIZE + 1], 2.0)
        assert_equal(group["offset"][DEFAULT_CHUNK_SIZE + 2 :], fillvalue)

        group["no_offset"].resize((3 * DEFAULT_CHUNK_SIZE,))
        group["offset"].resize((3 * DEFAULT_CHUNK_SIZE,))

        assert group["no_offset"].shape == (3 * DEFAULT_CHUNK_SIZE,)
        assert group["offset"].shape == (3 * DEFAULT_CHUNK_SIZE,)
        assert_equal(group["no_offset"][: 2 * DEFAULT_CHUNK_SIZE], 1.0)
        assert_equal(group["no_offset"][2 * DEFAULT_CHUNK_SIZE :], fillvalue)
        assert_equal(group["offset"][: DEFAULT_CHUNK_SIZE + 1], 1.0)
        assert_equal(group["offset"][DEFAULT_CHUNK_SIZE + 1], 2.0)
        assert_equal(group["offset"][DEFAULT_CHUNK_SIZE + 2 :], fillvalue)

    group = vfile["version2"]
    assert group["no_offset"].shape == (3 * DEFAULT_CHUNK_SIZE,)
    assert group["offset"].shape == (3 * DEFAULT_CHUNK_SIZE,)
    assert_equal(group["no_offset"][: 2 * DEFAULT_CHUNK_SIZE], 1.0)
    assert_equal(group["no_offset"][2 * DEFAULT_CHUNK_SIZE :], fillvalue)
    assert_equal(group["offset"][: DEFAULT_CHUNK_SIZE + 1], 1.0)
    assert_equal(group["offset"][DEFAULT_CHUNK_SIZE + 1], 2.0)
    assert_equal(group["offset"][DEFAULT_CHUNK_SIZE + 2 :], fillvalue)

    # Resize after calling create_dataset, larger
    with vfile.stage_version("resize_after_create_larger", "") as group:
        group.create_dataset("data", data=offset_data, fillvalue=fillvalue)
        group["data"].resize((DEFAULT_CHUNK_SIZE + 4,))

        assert group["data"].shape == (DEFAULT_CHUNK_SIZE + 4,)
        assert_equal(group["data"][: DEFAULT_CHUNK_SIZE + 2], 1.0)
        assert_equal(group["data"][DEFAULT_CHUNK_SIZE + 2 :], fillvalue)


def test_multidimsional(vfile):
    data = np.ones((2 * DEFAULT_CHUNK_SIZE, 5))

    with vfile.stage_version("version1") as g:
        g.create_dataset("test_data", data=data, chunks=(DEFAULT_CHUNK_SIZE, 2))
        assert_equal(g["test_data"][()], data)

    version1 = vfile["version1"]
    assert_equal(version1["test_data"][()], data)

    data2 = data.copy()
    data2[0, 1] = 2

    with vfile.stage_version("version2") as g:
        g["test_data"][0, 1] = 2
        assert g["test_data"][0, 1] == 2
        assert_equal(g["test_data"][()], data2)

    version2 = vfile["version2"]
    assert version2["test_data"][0, 1] == 2
    assert_equal(version2["test_data"][()], data2)

    data3 = data.copy()
    data3[0:1] = 3

    with vfile.stage_version("version3", "version1") as g:
        g["test_data"][0:1] = 3
        assert_equal(g["test_data"][0:1], 3)
        assert_equal(g["test_data"][()], data3)

    version3 = vfile["version3"]
    assert_equal(version3["test_data"][0:1], 3)
    assert_equal(version3["test_data"][()], data3)


def test_group_chunks_compression(vfile):
    # Chunks and compression are similar, so test them both at the same time.
    data = np.ones((2 * DEFAULT_CHUNK_SIZE, 5))

    with vfile.stage_version("version1") as g:
        g2 = g.create_group("group")
        g2.create_dataset(
            "test_data",
            data=data,
            chunks=(DEFAULT_CHUNK_SIZE, 2),
            compression="gzip",
            compression_opts=3,
        )
        assert_equal(g2["test_data"][()], data)
        assert_equal(g["group/test_data"][()], data)
        assert_equal(g["group"]["test_data"][()], data)

    version1 = vfile["version1"]
    assert_equal(version1["group"]["test_data"][()], data)
    assert_equal(version1["group/test_data"][()], data)

    raw_data = vfile.f["/_version_data/group/test_data/raw_data"]
    assert raw_data.compression == "gzip"
    assert raw_data.compression_opts == 3


def test_closes(vfile):
    data = np.ones(DEFAULT_CHUNK_SIZE)

    with vfile.stage_version("version1") as g:
        g.create_dataset("test_data", data=data)
    assert vfile._closed is False
    assert vfile.closed is False

    version_data = vfile._version_data
    versions = vfile._versions

    h5pyfile = vfile.f
    vfile.close()

    assert vfile._closed is True
    assert vfile.closed is True
    assert not hasattr(vfile, "f")
    assert not hasattr(vfile, "_version_data")
    assert not hasattr(vfile, "_versions")
    assert repr(vfile) == "<Closed VersionedHDF5File>"

    reopened_file = VersionedHDF5File(h5pyfile)
    assert list(reopened_file["version1"]) == ["test_data"]
    assert_equal(reopened_file["version1"]["test_data"][()], data)

    assert reopened_file._version_data == version_data
    assert reopened_file._versions == versions

    # Close the underlying file
    h5pyfile.close()
    assert vfile.closed is True
    with pytest.raises(ValueError):
        vfile["version1"]
    with pytest.raises(ValueError):
        vfile["version2"]
    assert repr(vfile) == "<Closed VersionedHDF5File>"


@pytest.mark.parametrize(
    ("data1", "data2"),
    [
        (b"baz", b"foo"),
        (np.asarray("baz", dtype="S"), np.asarray("foo", dtype="S")),
        (
            np.asarray("baz", dtype=h5py.string_dtype()),
            np.asarray("foo", dtype=h5py.string_dtype()),
        ),
        (1.5, 2.3),
        (1, 0),
        (np.int16(1), np.int16(2)),
    ],
)
def test_scalar_dataset(vfile, data1, data2):
    dtype = np.asarray(data1).dtype
    with vfile.stage_version("v1") as group:
        group["scalar_ds"] = data1

    v1_ds = vfile["v1"]["scalar_ds"]
    assert v1_ds[()] == data1
    assert v1_ds.shape == ()
    assert v1_ds.dtype == dtype
    assert v1_ds._buffer.dtype == dtype

    with vfile.stage_version("v2") as group:
        group["scalar_ds"] = data2

    v2_ds = vfile["v2"]["scalar_ds"]
    assert v2_ds[()] == data2
    assert v2_ds.shape == ()
    assert v2_ds.dtype == dtype
    assert v2_ds._buffer.dtype == dtype


def test_store_binary_as_void(vfile):
    with vfile.stage_version("version1") as sv:
        sv["test_store_binary_data"] = [np.void(b"1111")]

    version1 = vfile["version1"]
    assert_equal(version1["test_store_binary_data"][0], np.void(b"1111"))

    with vfile.stage_version("version2") as sv:
        sv["test_store_binary_data"][:] = [np.void(b"1234567890")]

    version2 = vfile["version2"]
    assert_equal(version2["test_store_binary_data"][0], np.void(b"1234"))


def test_check_committed(vfile):
    data = np.ones((DEFAULT_CHUNK_SIZE,))

    with vfile.stage_version("version1") as g:
        g.create_dataset("test_data", data=data)

    with pytest.raises(ValueError, match="committed"):
        g["data"] = data

    with pytest.raises(ValueError, match="committed"):
        g.create_dataset("data", data=data)

    with pytest.raises(ValueError, match="committed"):
        g.create_group("subgruop")

    with pytest.raises(ValueError, match="committed"):
        del g["test_data"]

    # Incorrectly uses g from the previous version (InMemoryArrayDataset)
    with (  # noqa: PT012
        pytest.raises(ValueError, match="committed"),
        vfile.stage_version("version2"),
    ):
        assert isinstance(g["test_data"].dataset, InMemoryArrayDataset)
        g["test_data"][0] = 1

    with (  # noqa: PT012
        pytest.raises(ValueError, match="committed"),
        vfile.stage_version("version2"),
    ):
        assert isinstance(g["test_data"].dataset, InMemoryArrayDataset)
        g["test_data"].resize((100,))

    with vfile.stage_version("version2") as g2:
        pass

    # Incorrectly uses g from the previous version (InMemoryDataset)
    with (  # noqa: PT012
        pytest.raises(ValueError, match="committed"),
        vfile.stage_version("version3"),
    ):
        assert isinstance(g2["test_data"], DatasetWrapper)
        assert isinstance(g2["test_data"].dataset, InMemoryDataset)
        g2["test_data"][0] = 1

    with (  # noqa: PT012
        pytest.raises(ValueError, match="committed"),
        vfile.stage_version("version3"),
    ):
        assert isinstance(g2["test_data"], DatasetWrapper)
        assert isinstance(g2["test_data"].dataset, InMemoryDataset)
        g2["test_data"].resize((100,))

    assert repr(g) == '<Committed InMemoryGroup "/_version_data/versions/version1">'


def test_set_chunks_nested(vfile):
    with vfile.stage_version("0") as sv:
        data_group = sv.create_group("data")
        data_group.create_dataset("bar", data=np.arange(4))

    with vfile.stage_version("1") as sv:
        data_group = sv["data"]
        data_group.create_dataset("props/1/bar", data=np.arange(0, 4, 2))


def test_InMemoryArrayDataset_chunks(vfile):
    with vfile.stage_version("0") as sv:
        data_group = sv.create_group("data")
        data_group.create_dataset(
            "g/bar",
            data=np.arange(4),
            chunks=(100,),
            compression="gzip",
            compression_opts=3,
        )
        assert isinstance(data_group["g/bar"].dataset, InMemoryArrayDataset)
        assert data_group["g/bar"].chunks == (100,)
        assert data_group["g/bar"].compression == "gzip"
        assert data_group["g/bar"].compression_opts == 3


@pytest.mark.parametrize(
    "dt",
    [
        h5py.string_dtype("utf-8"),
        h5py.string_dtype("ascii"),
        h5py.string_dtype("utf-8", length=20),
        h5py.string_dtype("ascii", length=20),
    ],
)
def test_string_dtypes(setup_vfile, dt):
    # Make sure the fillvalue logic works correctly for custom h5py string
    # dtypes.
    data = np.full(10, b"hello world", dtype=dt)

    with setup_vfile() as f:
        file = VersionedHDF5File(f)
        with file.stage_version("0") as sv:
            sv.create_dataset("name", shape=(10,), dtype=dt, data=data)
            assert isinstance(sv["name"].dataset, InMemoryArrayDataset)
            sv["name"].resize((11,))

        assert file["0"]["name"].dtype == dt
        assert_equal(file["0"]["name"][:10], data)
        assert file["0"]["name"][10] == b"", dt.metadata

        with file.stage_version("1") as sv:
            assert isinstance(sv["name"], DatasetWrapper)
            assert isinstance(sv["name"].dataset, InMemoryDataset)
            sv["name"].resize((12,))

        assert file["1"]["name"].dtype == dt
        assert_equal(file["1"]["name"][:10], data, str(dt.metadata))
        assert file["1"]["name"][10] == b"", dt.metadata
        assert file["1"]["name"][11] == b"", dt.metadata

        # Make sure we are matching the pure h5py behavior
        f.create_dataset(
            "name", shape=(10,), dtype=dt, data=data, chunks=(10,), maxshape=(None,)
        )
        f["name"].resize((11,))
        assert f["name"].dtype == dt
        assert_equal(f["name"][:10], data)
        assert f["name"][10] == b"", dt.metadata


def test_empty(vfile):
    with vfile.stage_version("version1") as g:
        g["data"] = np.arange(10)
        g.create_dataset("data2", data=np.empty((1, 0, 2)), chunks=(5, 5, 5))
        assert_equal(g["data2"][()], np.empty((1, 0, 2)))
    assert_equal(vfile["version1"]["data2"][()], np.empty((1, 0, 2)))

    with vfile.stage_version("version2") as g:
        g["data"].resize((0,))
        assert_equal(g["data"][()], np.empty((0,)))

    assert_equal(vfile["version2"]["data"][()], np.empty((0,)))
    assert_equal(vfile["version2"]["data2"][()], np.empty((1, 0, 2)))


def test_read_only(setup_vfile):
    vfile = setup_vfile()
    filename = vfile.filename
    with vfile as f:
        file = VersionedHDF5File(f)
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        with file.stage_version("version1", timestamp=timestamp) as g:
            g["data"] = [0, 1, 2]

        with pytest.raises(ValueError):
            g["data"][0] = 1
        with pytest.raises(ValueError):
            g["data2"] = [1, 2, 3]

        with pytest.raises(ValueError):
            file["version1"]["data"][0] = 1
        with pytest.raises(ValueError):
            file["version1"]["data2"] = [1, 2, 3]

        with pytest.raises(ValueError):
            file[timestamp]["data"][0] = 1
        with pytest.raises(ValueError):
            file[timestamp]["data2"] = [1, 2, 3]

    with h5py.File(filename, "r+") as f:
        file = VersionedHDF5File(f)

        with pytest.raises(ValueError):
            file["version1"]["data"][0] = 1
        with pytest.raises(ValueError):
            file["version1"]["data2"] = [1, 2, 3]

        with pytest.raises(ValueError):
            file[timestamp]["data"][0] = 1
        with pytest.raises(ValueError):
            file[timestamp]["data2"] = [1, 2, 3]


def test_delete_datasets(vfile):
    data1 = np.arange(10)
    data2 = np.zeros(20, dtype=int)
    with vfile.stage_version("version1") as g:
        g["data"] = data1
        g.create_group("group1/group2")
        g["group1"]["group2"]["data1"] = data1

    with vfile.stage_version("del_data") as g:
        del g["data"]

    with vfile.stage_version("del_data1", "version1") as g:
        del g["group1/group2/data1"]

    with vfile.stage_version("del_group2", "version1") as g:
        del g["group1/group2"]

    with vfile.stage_version("del_group1", "version1") as g:
        del g["group1/"]

    with vfile.stage_version("version2", "del_data") as g:
        g["data"] = np.zeros(20, dtype=int)

    with vfile.stage_version("version3", "del_data1") as g:
        g["group1/group2/data1"] = data2

    with vfile.stage_version("version4", "del_group2") as g:
        g.create_group("group1/group2")
        g["group1/group2/data1"] = data2

    with vfile.stage_version("version5", "del_group1") as g:
        g.create_group("group1/group2")
        g["group1/group2/data1"] = data2

    assert set(vfile["version1"]) == {"group1", "data"}
    assert list(vfile["version1"]["group1"]) == ["group2"]
    assert list(vfile["version1"]["group1"]["group2"]) == ["data1"]
    assert_equal(vfile["version1"]["data"][:], data1)
    assert_equal(vfile["version1"]["group1/group2/data1"][:], data1)

    assert list(vfile["del_data"]) == ["group1"]
    assert list(vfile["del_data"]["group1"]) == ["group2"]
    assert list(vfile["del_data"]["group1"]["group2"]) == ["data1"]
    assert_equal(vfile["del_data"]["group1/group2/data1"][:], data1)

    assert set(vfile["del_data1"]) == {"group1", "data"}
    assert list(vfile["del_data1"]["group1"]) == ["group2"]
    assert list(vfile["del_data1"]["group1"]["group2"]) == []
    assert_equal(vfile["del_data1"]["data"][:], data1)

    assert set(vfile["del_group2"]) == {"group1", "data"}
    assert list(vfile["del_group2"]["group1"]) == []
    assert_equal(vfile["del_group2"]["data"][:], data1)

    assert list(vfile["del_group1"]) == ["data"]
    assert_equal(vfile["del_group1"]["data"][:], data1)

    assert set(vfile["version2"]) == {"group1", "data"}
    assert list(vfile["version2"]["group1"]) == ["group2"]
    assert list(vfile["version2"]["group1"]["group2"]) == ["data1"]
    assert_equal(vfile["version2"]["data"][:], data2)
    assert_equal(vfile["version2"]["group1/group2/data1"][:], data1)

    assert set(vfile["version3"]) == {"group1", "data"}
    assert list(vfile["version3"]["group1"]) == ["group2"]
    assert list(vfile["version3"]["group1"]["group2"]) == ["data1"]
    assert_equal(vfile["version3"]["data"][:], data1)
    assert_equal(vfile["version3"]["group1/group2/data1"][:], data2)

    assert set(vfile["version4"]) == {"group1", "data"}
    assert list(vfile["version4"]["group1"]) == ["group2"]
    assert list(vfile["version4"]["group1"]["group2"]) == ["data1"]
    assert_equal(vfile["version4"]["data"][:], data1)
    assert_equal(vfile["version4"]["group1/group2/data1"][:], data2)

    assert set(vfile["version5"]) == {"group1", "data"}
    assert list(vfile["version5"]["group1"]) == ["group2"]
    assert list(vfile["version5"]["group1"]["group2"]) == ["data1"]
    assert_equal(vfile["version5"]["data"][:], data1)
    assert_equal(vfile["version5"]["group1/group2/data1"][:], data2)


def test_auto_create_group(vfile):
    with vfile.stage_version("version1") as g:
        g["a/b/c"] = [0, 1, 2]
        assert_equal(g["a"]["b"]["c"][:], [0, 1, 2])

    assert_equal(vfile["version1"]["a"]["b"]["c"][:], [0, 1, 2])


def test_scalar(setup_vfile):
    file = setup_vfile()
    filename = file.filename
    with file as f:
        vfile = VersionedHDF5File(f)
        with vfile.stage_version("version1") as g:
            dtype = h5py.special_dtype(vlen=bytes)
            g.create_dataset("bar", data=np.array(["aaa"], dtype="O"), dtype=dtype)

    with h5py.File(filename, "r+") as f:
        vfile = VersionedHDF5File(f)
        assert isinstance(vfile["version1"]["bar"], DatasetWrapper)
        assert isinstance(vfile["version1"]["bar"].dataset, InMemoryDataset)
        # Should return a scalar, not a shape () array
        assert isinstance(vfile["version1"]["bar"][0], bytes)

    with h5py.File(filename, "r") as f:
        vfile = VersionedHDF5File(f)
        assert isinstance(vfile["version1"]["bar"], h5py.Dataset)
        # Should return a scalar, not a shape () array
        assert isinstance(vfile["version1"]["bar"][0], bytes)


def test_sparse(vfile):
    with vfile.stage_version("version1") as g:
        g.create_dataset(
            "test_data",
            shape=(10_000, 10_000),
            dtype=np.dtype("int64"),
            data=None,
            chunks=(100, 100),
            fillvalue=1,
        )
        assert isinstance(g["test_data"], InMemorySparseDataset)
        assert g["test_data"][0, 0] == 1
        assert g["test_data"][0, 1] == 1
        assert g["test_data"][200, 1] == 1

        g["test_data"][0, 0] = 2
        assert g["test_data"][0, 0] == 2
        assert g["test_data"][0, 1] == 1
        assert g["test_data"][200, 1] == 1

    with vfile.stage_version("version2") as g:
        assert isinstance(g["test_data"], DatasetWrapper)
        assert isinstance(g["test_data"].dataset, InMemoryDataset)
        assert g["test_data"][0, 0] == 2
        assert g["test_data"][0, 1] == 1
        assert g["test_data"][200, 1] == 1

        g["test_data"][200, 1] = 3

        assert g["test_data"][0, 0] == 2
        assert g["test_data"][0, 1] == 1
        assert g["test_data"][200, 1] == 3

    assert vfile["version1"]["test_data"][0, 0] == 2
    assert vfile["version1"]["test_data"][0, 1] == 1
    assert vfile["version1"]["test_data"][200, 1] == 1

    assert vfile["version2"]["test_data"][0, 0] == 2
    assert vfile["version2"]["test_data"][0, 1] == 1
    assert vfile["version2"]["test_data"][200, 1] == 3


def test_sparse_empty(vfile):
    with vfile.stage_version("version1") as g:
        g.create_dataset(
            "test_data",
            shape=(10_000, 10_000),
            dtype=np.dtype("int64"),
            data=None,
            chunks=(100, 100),
            fillvalue=1,
        )
        # Don't read or write any data from the sparse dataset

    assert vfile["version1"]["test_data"][0, 0] == 1
    assert vfile["version1"]["test_data"][0, 1] == 1
    assert vfile["version1"]["test_data"][200, 1] == 1

    with vfile.stage_version("version2") as g:
        assert isinstance(g["test_data"], DatasetWrapper)
        assert isinstance(g["test_data"].dataset, InMemoryDataset)
        assert g["test_data"][0, 0] == 1
        assert g["test_data"][0, 1] == 1
        assert g["test_data"][200, 1] == 1

        g["test_data"][0, 0] = 2
        g["test_data"][200, 1] = 2

        assert g["test_data"][0, 0] == 2
        assert g["test_data"][0, 1] == 1
        assert g["test_data"][200, 1] == 2

    assert vfile["version1"]["test_data"][0, 0] == 1
    assert vfile["version1"]["test_data"][0, 1] == 1
    assert vfile["version1"]["test_data"][200, 1] == 1

    assert vfile["version2"]["test_data"][0, 0] == 2
    assert vfile["version2"]["test_data"][0, 1] == 1
    assert vfile["version2"]["test_data"][200, 1] == 2


def test_sparse_large(vfile):
    # This is currently inefficient in terms of time, but test that it isn't
    # inefficient in terms of memory.
    with vfile.stage_version("version1") as g:
        # test_data would be 100GB if stored entirely in memory. We use a huge
        # chunk size to avoid taking too long with the current code that loops
        # over all chunk indices.
        g.create_dataset(
            "test_data",
            shape=(100_000_000_000,),
            data=None,
            chunks=(10_000_000,),
            fillvalue=0.0,
        )
        assert isinstance(g["test_data"], InMemorySparseDataset)
        assert g["test_data"][0] == 0
        assert g["test_data"][1] == 0
        assert g["test_data"][20_000_000] == 0

        g["test_data"][0] = 1
        assert g["test_data"][0] == 1
        assert g["test_data"][1] == 0
        assert g["test_data"][20_000_000] == 0

    with vfile.stage_version("version2") as g:
        assert isinstance(g["test_data"], DatasetWrapper)
        assert isinstance(g["test_data"].dataset, InMemoryDataset)
        assert g["test_data"][0] == 1
        assert g["test_data"][1] == 0
        assert g["test_data"][20_000_000] == 0

        g["test_data"][20_000_000] = 2

        assert g["test_data"][0] == 1
        assert g["test_data"][1] == 0
        assert g["test_data"][20_000_000] == 2

    assert vfile["version1"]["test_data"][0] == 1
    assert vfile["version1"]["test_data"][1] == 0
    assert vfile["version1"]["test_data"][20_000_000] == 0

    assert vfile["version2"]["test_data"][0] == 1
    assert vfile["version2"]["test_data"][1] == 0
    assert vfile["version2"]["test_data"][20_000_000] == 2


def test_no_recursive_version_group_access(vfile):
    timestamp1 = datetime.datetime.now(datetime.timezone.utc)
    with vfile.stage_version("version1", timestamp=timestamp1) as g:
        g.create_dataset("test", data=[1, 2, 3])

    timestamp2 = datetime.datetime.now(datetime.timezone.utc)
    minute = datetime.timedelta(minutes=1)
    with vfile.stage_version("version2", timestamp=timestamp2) as g:
        vfile["version1"]  # Doesn't raise
        with pytest.raises(ValueError):
            vfile["version2"]

        vfile[timestamp1]  # Doesn't raise
        # Without +minute, it will pick the previous version, as the
        # uncommitted group only has a placeholder timestamp, which will be
        # after timestamp2. Since this isn't supposed to work in the first
        # place, this isn't a big deal.
        with pytest.raises(ValueError):
            vfile[timestamp2 + minute]


def test_empty_dataset_str_dtype(vfile):
    # Issue #161. Make sure the dtype is maintained correctly for empty
    # datasets with custom string dtypes.
    with vfile.stage_version("version1") as g:
        g.create_dataset(
            "bar", data=np.array(["a", "b", "c"], dtype="S5"), dtype=np.dtype("S5")
        )
        g["bar"].resize((0,))
    with vfile.stage_version("version2") as g:
        g["bar"].resize((3,))
        g["bar"][:] = np.array(["a", "b", "c"], dtype="S5")


def test_datasetwrapper(vfile):
    with vfile.stage_version("r0") as sv:
        sv.create_dataset("bar", data=[1, 2, 3], chunks=(2,))
        sv["bar"].attrs["key"] = 0
        assert isinstance(sv["bar"].dataset, InMemoryArrayDataset)
        assert dict(sv["bar"].attrs) == {"key": 0}
        assert sv["bar"].chunks == (2,)

    with vfile.stage_version("r1") as sv:
        assert isinstance(sv["bar"], DatasetWrapper)
        assert isinstance(sv["bar"].dataset, InMemoryDataset)
        assert sv["bar"].attrs["key"] == 0
        sv["bar"].attrs["key"] = 1
        assert sv["bar"].attrs["key"] == 1
        assert sv["bar"].chunks == (2,)

        sv["bar"][:] = [4, 5, 6]
        assert isinstance(sv["bar"], DatasetWrapper)
        assert isinstance(sv["bar"].dataset, InMemoryArrayDataset)
        assert sv["bar"].attrs["key"] == 1
        assert sv["bar"].chunks == (2,)


def test_mask_reading(tmp_path):
    # Reading a virtual dataset with a mask does not work in HDF5, so make
    # sure it still works for versioned datasets.
    filename = tmp_path / "file.hdf5"
    mask = np.array([True, True, False], dtype="bool")

    with h5py.File(filename, "w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("r0") as sv:
            sv.create_dataset("bar", data=[1, 2, 3], chunks=(2,))
            b = sv["bar"][mask]
            assert_equal(b, [1, 2])

        b = vf["r0"]["bar"][mask]
        assert_equal(b, [1, 2])

    with h5py.File(filename, "r+") as f:
        vf = VersionedHDF5File(f)
        sv = vf["r0"]
        b = sv["bar"][mask]
        assert_equal(b, [1, 2])


def test_mask_reading_read_only(tmp_path):
    # Reading a virtual dataset with a mask does not work in HDF5, so make
    # sure it still works for versioned datasets.
    filename = tmp_path / "file.hdf5"
    mask = np.array([True, True, False], dtype="bool")

    with h5py.File(filename, "w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("r0") as sv:
            sv.create_dataset("bar", data=[1, 2, 3], chunks=(2,))
            b = sv["bar"][mask]
            assert_equal(b, [1, 2])

        b = vf["r0"]["bar"][mask]
        assert_equal(b, [1, 2])

    with h5py.File(filename, "r") as f:
        vf = VersionedHDF5File(f)
        sv = vf["r0"]
        b = sv["bar"][mask]
        assert_equal(b, [1, 2])


def test_read_only_no_wrappers(setup_vfile):
    file = setup_vfile()
    filename = file.filename
    # Read-only files should not use the wrapper classes
    with file as f:
        vfile = VersionedHDF5File(f)
        with vfile.stage_version("version1") as g:
            g.create_dataset("bar", data=np.array([0, 1, 2]))

    with h5py.File(filename, "r+") as f:
        vfile = VersionedHDF5File(f)
        assert isinstance(vfile["version1"], InMemoryGroup)
        assert isinstance(vfile["version1"]["bar"], DatasetWrapper)
        assert isinstance(vfile["version1"]["bar"].dataset, InMemoryDataset)

    with h5py.File(filename, "r") as f:
        vfile = VersionedHDF5File(f)
        assert isinstance(vfile["version1"], h5py.Group)
        assert isinstance(vfile["version1"]["bar"], h5py.Dataset)


def test_stage_version_log_stats(tmp_path, caplog):
    """Test that stage_version logs stats after writing data."""
    caplog.set_level(logging.DEBUG)
    filename = tmp_path / "file.hdf5"

    with h5py.File(filename, "w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("r0") as sv:
            bar_shape_r0 = (2, 15220, 2)
            bar_chunks_r0 = (300, 100, 2)
            baz_shape_r0 = (1, 10, 2)
            baz_chunks_r0 = (600, 2, 4)

            sv.create_dataset(
                "bar",
                bar_shape_r0,
                chunks=bar_chunks_r0,
                data=np.full((2, 15220, 2), 0),
            )
            sv.create_dataset(
                "baz", baz_shape_r0, chunks=baz_chunks_r0, data=np.full((1, 10, 2), 0)
            )

        assert caplog.records
        assert (
            "bar: New chunks written: 2; Number of chunks reused: 151"
            in caplog.records[-2].getMessage()
        )
        assert (
            "baz: New chunks written: 1; Number of chunks reused: 4"
            in caplog.records[-1].getMessage()
        )

        with vf.stage_version("r1") as sv:
            bar_shape_r1 = (3, 15222, 2)
            baz_shape_r1 = (1, 40, 2)

            bar = sv["bar"]
            bar.resize(bar_shape_r1)
            baz = sv["baz"]
            baz.resize(baz_shape_r1)
            baz[:, -10:, :] = np.full((1, 10, 2), 3)

        assert (
            "bar: New chunks written: 2; Number of chunks reused: 151"
            in caplog.records[-2].getMessage()
        )
        assert (
            "baz: New chunks written: 1; Number of chunks reused: 4"
            in caplog.records[-1].getMessage()
        )


def test_data_version_identifier_valid(tmp_path, caplog):
    """Test that a file with valid data version id opens without a log message."""
    caplog.set_level(logging.INFO)
    filename = tmp_path / "file.h5"
    with h5py.File(filename, "w") as f:
        VersionedHDF5File(f)
        assert f["_version_data"]["versions"].attrs["data_version"] == DATA_VERSION

    with h5py.File(filename, "r") as f:
        VersionedHDF5File(f)
        assert f["_version_data"]["versions"].attrs["data_version"] == DATA_VERSION

    assert len(caplog.records) == 0


def test_data_version_identifier_missing(tmp_path, caplog):
    """Test that a file with no data version identifier logs a message when opened."""
    caplog.set_level(logging.INFO)
    filename = tmp_path / "file.h5"
    with h5py.File(filename, "w") as f:
        vf = VersionedHDF5File(f)
        assert f["_version_data"]["versions"].attrs["data_version"] == DATA_VERSION

        # create a dtype='O' array
        with vf.stage_version("v0") as sv:
            sv.create_dataset(
                "values",
                data=np.array(["hello", "world"]),
                dtype=h5py.string_dtype(encoding="ascii"),
            )

        # Directly remove the data version identifier; this is
        # equivalent to v1.
        del f["_version_data/versions"].attrs["data_version"]

    with h5py.File(filename, "r") as f:
        VersionedHDF5File(f)

    assert len(caplog.records) == 1


def test_rebuild_hashtable(tmp_path, caplog):
    """Verify rebuilding the hashtable for a single object dtype and single version.

    1. An info log message is issued to the user about DATA_VERSION mismatch
    2. Check that the hash table has been modified for the data
    3. Check that the hashes produced are stable

    The test data contains a single version of a single dataset. The dataset is an
    object dtype array of strings.
    """
    caplog.set_level(logging.INFO)

    bad_file = TEST_DATA / "object_dtype_bad_hashtable_data.h5"
    filename = tmp_path / "file.h5"
    shutil.copy(bad_file, filename)

    with h5py.File(filename, mode="r+") as f:
        assert f["_version_data"]["versions"].attrs.get("data_version") is None
        original_hashes = f["_version_data/data_with_bad_hashtable/hash_table"][:]

        VersionedHDF5File(f)

        assert "data_version" not in f["_version_data"]["versions"].attrs
        assert_equal(
            original_hashes, f["_version_data/data_with_bad_hashtable/hash_table"][:]
        )

    # Info log message to the user is issued
    assert len(caplog.records) == 1
    caplog.clear()

    # now actually rebuild the hashes
    with h5py.File(filename, mode="r+") as f:
        assert f["_version_data"]["versions"].attrs.get("data_version") is None
        original_hashes = f["_version_data/data_with_bad_hashtable/hash_table"][:]

        vf = VersionedHDF5File(f)
        vf.rebuild_object_dtype_hashtables()

        assert f["_version_data"]["versions"].attrs["data_version"] == DATA_VERSION
        new_hashes = f["_version_data/data_with_bad_hashtable/hash_table"][:]

    # Info log message to the user is issued warning about DATA_VERSION mismatch;
    # another log message issued when rebuilding hash table
    assert len(caplog.records) == 2
    assert not np.all(original_hashes[0][0] == new_hashes[0][0])

    # Ensure new hash is stable
    expected = np.array(
        [
            252,
            36,
            175,
            92,
            195,
            20,
            142,
            4,
            239,
            58,
            202,
            209,
            147,
            120,
            100,
            137,
            156,
            220,
            173,
            52,
            45,
            1,
            230,
            255,
            252,
            205,
            149,
            145,
            65,
            175,
            239,
            159,
        ],
        dtype=np.uint8,
    )
    assert np.all(new_hashes[0][0] == expected)

    # The slice into the raw data for the hash entry should only be 4 elements, not
    # the default chunk size
    assert np.all(new_hashes[0][1] == np.array([0, 4]))


def test_rebuild_hashtable_multiple_datasets(tmp_path, caplog):
    """Verify rebuilding the hashtable for multiple datasets across multiple versions.

    1. An info log message is issued to the user about DATA_VERSION mismatch
    2. Check that the hash table has been modified for the data
    3. Check that the hashes produced are stable

    The test data contains 3 datasets:
        1. An object dtype array of strings
        2. An object dtype array of some other strings
        3. An array of ints

    The first two have bad hash tables, but the third doesn't. Here we rebuild all of
    the hash tables and check whether they're what we expect.
    """
    caplog.set_level(logging.INFO)

    bad_file = TEST_DATA / "object_dtype_bad_hashtable_data2.h5"
    filename = tmp_path / "file.h5"
    shutil.copy(bad_file, filename)

    with h5py.File(filename, mode="r+") as f:
        assert f["_version_data"]["versions"].attrs.get("data_version") is None
        original_hashes_arr1 = f["_version_data/data_with_bad_hashtable/hash_table"][:]
        original_hashes_arr2 = f["_version_data/data_with_bad_hashtable2/hash_table"][:]
        original_hashes_arr3 = f["_version_data/linspace/hash_table"][:]

        VersionedHDF5File(f)

        assert "data_version" not in f["_version_data"]["versions"].attrs
        assert_equal(
            original_hashes_arr1,
            f["_version_data/data_with_bad_hashtable/hash_table"][:],
        )
        assert_equal(
            original_hashes_arr2,
            f["_version_data/data_with_bad_hashtable2/hash_table"][:],
        )
        assert_equal(original_hashes_arr3, f["_version_data/linspace/hash_table"][:])

    # Info log message to the user is issued
    assert len(caplog.records) == 1
    caplog.clear()

    with h5py.File(filename, mode="r+") as f:
        assert f["_version_data"]["versions"].attrs.get("data_version") is None
        original_hashes_arr1 = f["_version_data/data_with_bad_hashtable/hash_table"][:]
        original_hashes_arr2 = f["_version_data/data_with_bad_hashtable2/hash_table"][:]
        original_hashes_arr3 = f["_version_data/linspace/hash_table"][:]

        vf = VersionedHDF5File(f)
        vf.rebuild_object_dtype_hashtables()

        assert f["_version_data"]["versions"].attrs["data_version"] == DATA_VERSION
        new_hashes_arr1 = f["_version_data/data_with_bad_hashtable/hash_table"][:]
        new_hashes_arr2 = f["_version_data/data_with_bad_hashtable2/hash_table"][:]
        new_hashes_arr3 = f["_version_data/linspace/hash_table"][:]

    original_hashes = [
        original_hashes_arr1,
        original_hashes_arr2,
        original_hashes_arr3,
    ]
    new_hashes = [
        new_hashes_arr1,
        new_hashes_arr2,
        new_hashes_arr3,
    ]

    # Info log message to the user is issued warning about DATA_VERSION mismatch;
    # another log message issued when rebuilding hash table
    assert len(caplog.records) == 2

    for original_hash_arr, new_hash_arr in zip(
        original_hashes, new_hashes, strict=True
    ):
        # When data is written originally, if all chunks are unique the number of
        # entries in the hash table should be the number of writes that were made. The
        # new hash table should follow the same rule.
        assert len(original_hash_arr) == len(new_hash_arr)

        # The slices into the raw data should span the number of elements in the
        # chunk. In this case, each of these arrays fits in a single chunk. Here, we
        # check the slices into the raw data to make sure they are identical for the
        # rebuilt hash table.
        for arr_hash, new_hash in zip(original_hash_arr, new_hash_arr, strict=True):
            assert np.all(arr_hash[1] == new_hash[1])

    for arr_hash, new_hash in zip(original_hashes_arr1, new_hashes_arr1, strict=True):
        assert not np.all(arr_hash[0] == new_hash[0])

    for arr_hash, new_hash in zip(original_hashes_arr2, new_hashes_arr2, strict=True):
        assert not np.all(arr_hash[0] == new_hash[0])

    # This is an integer array, it should have been correctly hashed originally.
    # Check that the new hashes match.
    for arr_hash, new_hash in zip(original_hashes_arr3, new_hashes_arr3, strict=True):
        assert np.all(arr_hash[0] == new_hash[0])


def test_rebuild_hashtable_nested_dataset(tmp_path, caplog):
    """Test rebuilding the hash tables of a nested dataset."""
    caplog.set_level(logging.INFO)

    bad_file = TEST_DATA / "nested_data_old_data_version.h5"
    filename = tmp_path / "file.h5"
    shutil.copy(bad_file, filename)

    with h5py.File(filename, mode="r+") as f:
        VersionedHDF5File(f)

        # Check that the hash table exists in the nested dataset
        assert "hash_table" in f["_version_data/data/values"]

    # Info log message to the user is issued
    assert len(caplog.records) == 1
    caplog.clear()

    with h5py.File(filename, mode="r+") as f:
        vf = VersionedHDF5File(f)
        vf.rebuild_object_dtype_hashtables()

    # Info log message to the user is issued warning about DATA_VERSION mismatch;
    # another log message issued when rebuilding hash table
    assert len(caplog.records) == 2


def test_rebuild_hashtable_multiple_nested_dataset(tmp_path, caplog):
    """Test rebuilding the hash tables of a nested dataset."""
    caplog.set_level(logging.INFO)

    bad_file = TEST_DATA / "multiple_nested_data_old_data_version.h5"
    filename = tmp_path / "file.h5"
    shutil.copy(bad_file, filename)

    with h5py.File(filename, mode="r+") as f:
        old_hashtable = f["_version_data/foo/bar/baz/foo/bar/baz/values/hash_table"][:]
        VersionedHDF5File(f)

        # Check that the hash table exists in the nested dataset
        assert "hash_table" in f["_version_data/foo/bar/baz/foo/bar/baz/values"]

    # Info log message to the user is issued
    assert len(caplog.records) == 1
    caplog.clear()

    with h5py.File(filename, mode="r+") as f:
        vf = VersionedHDF5File(f)
        vf.rebuild_object_dtype_hashtables()

        new_hashtable = f["_version_data/foo/bar/baz/foo/bar/baz/values/hash_table"][:]

        # Check that the bytes in the hash table are different
        assert not np.array_equal(new_hashtable[0][0], old_hashtable[0][0])

    # Info log message to the user is issued warning about DATA_VERSION mismatch;
    # another log message issued when rebuilding hash table
    assert len(caplog.records) == 2


def test_rebuild_hashtable_chunk_reuse(tmp_path, caplog):
    """Test that the correct chunks are used after rebuilding the tables."""
    caplog.set_level(logging.INFO)

    bad_file = TEST_DATA / "object_dtype_bad_hashtable_chunk_reuse.h5"
    filename = tmp_path / "file.h5"
    shutil.copy(bad_file, filename)

    with h5py.File(filename, mode="r+") as f:
        old_hashtable = f["_version_data/values/hash_table"][:]
        VersionedHDF5File(f)

        # Check that the hash table exists in the nested dataset
        assert "hash_table" in f["_version_data/values"]

    # Info log message to the user is issued
    assert len(caplog.records) == 1
    caplog.clear()

    with h5py.File(filename, mode="r+") as f:
        vf = VersionedHDF5File(f)
        vf.rebuild_object_dtype_hashtables()

        new_hashtable = f["_version_data/values/hash_table"][:]

        # Check that the bytes in the hash table are different
        assert not np.array_equal(new_hashtable[0][0], old_hashtable[0][0])

    # Info log message to the user is issued warning about DATA_VERSION mismatch;
    # another log message issued when rebuilding hash table
    assert len(caplog.records) == 2

    with h5py.File(filename, mode="r") as f:
        vf = VersionedHDF5File(f)
        cv = vf[vf.current_version]
        assert np.array_equal(
            cv["values"][:], [bytes(str(j), "utf-8") for j in range(10)]
        )

    # add versions, check that correct chunks are reused
    for i in range(1, 11):
        with h5py.File(filename, mode="r+") as f:
            vf = VersionedHDF5File(f)
            with vf.stage_version(f"reuse.{i}") as sv:
                sv["values"] = np.arange(i).astype(str).astype("O")

        with h5py.File(filename, mode="r") as f:
            # no new chunks should have been added
            assert f["_version_data/values/raw_data"].shape == (
                f["_version_data/values/raw_data"].chunks[0] * 11,
            )
            # data should be correct
            vf = VersionedHDF5File(f)
            cv = vf[vf.current_version]
            assert np.array_equal(
                cv["values"][:], [bytes(str(j), "utf-8") for j in range(i)]
            )


def test_rebuild_hashtable_chunk_reuse_unicode(tmp_path, caplog):
    """Test that the correct chunks are used after rebuilding the tables for
    non-ascii data.
    """
    caplog.set_level(logging.INFO)

    bad_file = TEST_DATA / "object_dtype_bad_hashtable_chunk_reuse_unicode.h5"
    filename = tmp_path / "file.h5"
    shutil.copy(bad_file, filename)

    with h5py.File(filename, mode="r+") as f:
        old_hashtable = f["_version_data/values/hash_table"][:]
        VersionedHDF5File(f)

        # Check that the hash table exists in the nested dataset
        assert "hash_table" in f["_version_data/values"]

    # Info log message to the user is issued
    assert len(caplog.records) == 1
    caplog.clear()

    with h5py.File(filename, mode="r+") as f:
        vf = VersionedHDF5File(f)
        vf.rebuild_object_dtype_hashtables()

        new_hashtable = f["_version_data/values/hash_table"][:]

        # Check that the bytes in the hash table are different
        assert not np.array_equal(new_hashtable[0][0], old_hashtable[0][0])

    # Info log message to the user is issued warning about DATA_VERSION mismatch;
    # another log message issued when rebuilding hash table
    assert len(caplog.records) == 2

    numbers = [
        "れい",
        "いち",
        "に",
        "さん",
        "し",
        "ご",
        "ろく",
        "しち",
        "はち",
        "きゅう",
        "じゅう",
    ]

    with h5py.File(filename, mode="r") as f:
        vf = VersionedHDF5File(f)
        cv = vf[vf.current_version]
        assert np.array_equal(
            cv["values"][:], [bytes(numbers[j], "utf-8") for j in range(11)]
        )

    # add versions, check that correct chunks are reused
    for i in range(0, 11):
        with h5py.File(filename, mode="r+") as f:
            vf = VersionedHDF5File(f)
            with vf.stage_version(f"reuse.{i}") as sv:
                sv["values"] = np.array(numbers[: i + 1], dtype="O")

        with h5py.File(filename, mode="r") as f:
            # no new chunks should have been added
            assert f["_version_data/values/raw_data"].shape == (
                f["_version_data/values/raw_data"].chunks[0] * 11,
            )
            # data should be correct
            vf = VersionedHDF5File(f)
            cv = vf[vf.current_version]
            assert np.array_equal(
                cv["values"][:], [bytes(numbers[j], "utf-8") for j in range(i + 1)]
            )


def test_rebuild_hashtable_chunk_reuse_multi_dim(tmp_path, caplog):
    """Test that the correct chunks are used after rebuilding the tables for a
    multi-dimensional array.
    """
    caplog.set_level(logging.INFO)

    bad_file = TEST_DATA / "object_dtype_bad_hashtable_chunk_reuse_multi_dim.h5"
    filename = tmp_path / "file.h5"
    shutil.copy(bad_file, filename)

    with h5py.File(filename, mode="r+") as f:
        old_hashtable = f["_version_data/values/hash_table"][:]
        VersionedHDF5File(f)

        # Check that the hash table exists in the nested dataset
        assert "hash_table" in f["_version_data/values"]

    # Info log message to the user is issued
    assert len(caplog.records) == 1
    caplog.clear()

    with h5py.File(filename, mode="r+") as f:
        vf = VersionedHDF5File(f)
        vf.rebuild_object_dtype_hashtables()

        new_hashtable = f["_version_data/values/hash_table"][:]

        # Check that the bytes in the hash table are different
        assert not np.array_equal(new_hashtable[0][0], old_hashtable[0][0])

    # Info log message to the user is issued warning about DATA_VERSION mismatch;
    # another log message issued when rebuilding hash table
    assert len(caplog.records) == 2

    with h5py.File(filename, mode="r") as f:
        vf = VersionedHDF5File(f)
        cv = vf[vf.current_version]
        assert np.array_equal(
            cv["values"][:],
            np.array(
                [
                    [
                        (chr(ord("a") + ((10 + j + k) % 10)) * 3).encode("utf-8")
                        for j in range(4)
                    ]
                    for k in range(4)
                ],
                dtype="O",
            ),
        )

    # add versions, check that correct chunks are reused
    for i in range(1, 11):
        values_i = np.array(
            [
                [chr(ord("a") + ((i + j + k) % 10)) * 3 for j in range(4)]
                for k in range(4)
            ],
            dtype="O",
        )
        with h5py.File(filename, mode="r+") as f:
            vf = VersionedHDF5File(f)
            with vf.stage_version(f"reuse.{i}") as sv:
                sv["values"] = values_i

        with h5py.File(filename, mode="r") as f:
            # no new chunks should have been added
            raw_data_shape = f["_version_data/values/raw_data"].shape
            raw_data_chunks = f["_version_data/values/raw_data"].chunks
            assert raw_data_shape == (raw_data_chunks[0] * 44, raw_data_chunks[1])
            # data should be correct
            vf = VersionedHDF5File(f)
            cv = vf[vf.current_version]
            assert np.array_equal(
                cv["values"][:],
                np.array([[v.encode("utf-8") for v in a] for a in values_i], dtype="O"),
            )


def test_get_diff(tmp_path):
    """Check that the diff betwen two versions returns the expected chunks."""
    test_data = np.concatenate(
        (
            np.ones((2 * DEFAULT_CHUNK_SIZE,)),
            2 * np.ones((DEFAULT_CHUNK_SIZE,)),
            3 * np.ones((DEFAULT_CHUNK_SIZE,)),
        )
    )

    filename = tmp_path / "data.h5"

    with h5py.File(filename, "w") as f:
        vfile = VersionedHDF5File(f)

        with vfile.stage_version("v1") as g:
            g["test_data"] = test_data

        with vfile.stage_version("v2") as g:
            g["test_data"][0] = 0.0

    with h5py.File(filename, "r") as f:
        vfile = VersionedHDF5File(f)
        diff = vfile.get_diff("test_data", "v1", "v2")

    print(diff)


def test_get_diff_same_version(tmp_path):
    """Check that the diff between a version and itself is nothing."""
    test_data = np.concatenate(
        (
            np.ones((2 * DEFAULT_CHUNK_SIZE,)),
            2 * np.ones((DEFAULT_CHUNK_SIZE,)),
            3 * np.ones((DEFAULT_CHUNK_SIZE,)),
        )
    )

    filename = tmp_path / "data.h5"

    with h5py.File(filename, "w") as f:
        vfile = VersionedHDF5File(f)

        with vfile.stage_version("v1") as g:
            g["test_data"] = test_data

    with h5py.File(filename, "r") as f:
        vfile = VersionedHDF5File(f)
        diff = vfile.get_diff("test_data", "v1", "v1")

    assert diff == {}


def test_versions_property(vfile):
    """Test that VersionedHDF5File.versions returns the same as
    all_versions(vfile).
    """

    for i in range(100):
        with vfile.stage_version(f"r{i}") as sv:
            sv["values"] = np.arange(i, 100)
            assert set(all_versions(vfile.f)) == set(vfile.versions)

    # keep only every 10th version
    versions_to_delete = []
    versions = sorted(
        [(v, vfile._versions[v].attrs["timestamp"]) for v in vfile._versions],
        key=lambda t: t[1],
    )
    for i, v in enumerate(versions):
        if i % 10 != 0:
            versions_to_delete.append(v[0])

    # Delete some versions and check for the correct versions again
    delete_versions(vfile, versions_to_delete)
    assert set(all_versions(vfile.f)) == set(vfile.versions)


def test_make_empty_dataset(tmp_path):
    """Check that creating a dataset before making it empty can be done successfully.

    This test would pass unless the file gets closed/reopened for each operation,
    which is why we do that here; unsure about why that is, but it must be related to
    flushing reads/writes.

    See https://github.com/deshaw/versioned-hdf5/issues/314 for context.
    """
    filename = tmp_path / "tmp.h5"
    with h5py.File(filename, "w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("r0") as sv:
            sv.create_dataset("values", data=np.array([1, 2, 3]))

    with h5py.File(filename, "r+") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("r1") as sv:
            sv["values"].resize((0,))

    with h5py.File(filename, "r+") as f:
        delete_versions(f, ["r0"])

    with h5py.File(filename, "r+") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("r2") as sv:
            sv["values"].resize((0,))

    with h5py.File(filename, "r+") as f:
        vf = VersionedHDF5File(f)
        cv = vf[vf.current_version]
        assert_equal(cv["values"][:], np.array([]))


def test_make_empty_multidimensional_dataset(tmp_path):
    """Check that creating a multidimensional dataset before making it empty can be
    done successfully.

    See https://github.com/deshaw/versioned-hdf5/issues/430 for context.
    """
    filename = tmp_path / "tmp.h5"
    with h5py.File(filename, "w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("r0") as sv:
            sv.create_dataset(
                "values", data=np.array([[1, 2, 3], [4, 5, 6]]), chunks=(100, 100)
            )

    with h5py.File(filename, "r+") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("r1") as sv:
            sv["values"].resize((0, 0))

    with h5py.File(filename, "r+") as f:
        delete_versions(f, ["r0"])

    with h5py.File(filename, "r+") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("r2") as sv:
            sv["values"].resize((0, 0))

    with h5py.File(filename, "r+") as f:
        vf = VersionedHDF5File(f)
        cv = vf[vf.current_version]
        assert_equal(cv["values"][:], np.zeros((0, 0), dtype="int64"))


def test_insert_in_middle_multi_dim(tmp_path):
    """
    Test we correctly handle inserting into a multi-dimensional Dataset
    and shift the existing entries back.
    """
    rs = np.random.RandomState(0)
    dims = 3
    filename = tmp_path / "tmp.h5"

    with h5py.File(filename, "w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("v0") as sv:
            for i in range(dims):
                sv.create_dataset(
                    f"axis{i}",
                    dtype=np.dtype("int64"),
                    shape=(0,),
                    chunks=(10000,),
                    maxshape=(None,),
                )
            sv.create_dataset(
                "value",
                dtype=np.dtype("int64"),
                shape=tuple([0 for _ in range(dims)]),
                chunks=tuple([20 for _ in range(dims)]),
                maxshape=tuple([None for _ in range(dims)]),
                fillvalue=0,
            )
            sv.create_dataset(
                "mask",
                dtype=np.dtype("int8"),
                shape=tuple([0 for _ in range(dims)]),
                chunks=tuple([20 for _ in range(dims)]),
                maxshape=tuple([None for _ in range(dims)]),
                fillvalue=2,
            )
    for i in range(1, 101):
        with h5py.File(filename, "r+") as f:
            vf = VersionedHDF5File(f)
            with vf.stage_version(f"v{i}") as sv:
                new_axes = tuple(np.unique(rs.randint(30, size=5)) for _ in range(dims))
                new_value = np.full(tuple(len(ax) for ax in new_axes), i)
                new_mask = np.full(tuple(len(ax) for ax in new_axes), 0)

                # figure out how existing axes map to new axes
                new_indices = []
                existing_indices = []
                new_shape = []
                for i in range(dims):
                    axis_ds = sv[f"axis{i}"]
                    all_axis, indices = np.unique(
                        np.concatenate([axis_ds[:], new_axes[i]]), return_inverse=True
                    )
                    existing_indices.append(tuple(indices[: len(axis_ds)]))
                    new_indices.append(tuple(indices[len(axis_ds) :]))
                    axis_ds.resize((len(all_axis),))
                    axis_ds[:] = all_axis
                    new_shape.append(len(all_axis))

                new_indices = tuple(new_indices)
                existing_indices = tuple(existing_indices)
                new_shape = tuple(new_shape)

                # merge value
                value_ds = sv["value"]
                all_data = np.full(new_shape, value_ds.fillvalue)
                existing_data = value_ds[:]
                all_data[np.ix_(*existing_indices)] = existing_data
                all_data[np.ix_(*new_indices)] = new_value
                value_ds.resize(new_shape)
                value_ds[:] = all_data

                # merge mask
                mask_ds = sv["mask"]
                all_mask = np.full(new_shape, mask_ds.fillvalue)
                existing_mask = mask_ds[:]
                all_mask[np.ix_(*existing_indices)] = existing_mask
                all_mask[np.ix_(*new_indices)] = new_mask
                mask_ds.resize(new_shape)
                mask_ds[:] = all_mask

        with h5py.File(filename, "r") as f:
            vf = VersionedHDF5File(f)
            cv = vf[vf.current_version]
            assert_equal(cv["value"], all_data)
            assert_equal(cv["mask"], all_mask)


def test_verify_string_chunk_reuse_bytes_one_dimensional(tmp_path):
    """Test that string chunk reuse works as intended."""
    filename = tmp_path / "tmp.h5"

    with h5py.File(filename, mode="w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("r0") as group:
            group.create_dataset(
                "values",
                data=np.array(["a", "b", "c", "a", "b", "c"], dtype="O"),
                dtype=h5py.string_dtype(length=None),
                maxshape=(None,),
                chunks=(3,),
            )

        # The underlying dataset stores strings as bytes
        assert_equal(
            f["_version_data/values/raw_data"][:].astype(object),
            np.array([b"a", b"b", b"c"]).astype(object),
        )


def test_other_compression_bad_value(vfile):
    """Test that invalid compression types do not validate."""
    with pytest.raises(ValueError, match="invalid"), vfile.stage_version("r0") as sv:
        sv.create_dataset(
            "values",
            data=np.arange(10),
            compression=-1,
            compression_opts=(0, 0, 0, 0, 7, 1, 2),
        )


@pytest.mark.parametrize("raw", [True, False])
def test_blosc_compression_validates(h5file, raw):
    """Test that third-party compression filters from hdf5plugin or pytables
    such as Blosc validate correctly.
    """
    hdf5plugin = pytest.importorskip("hdf5plugin")

    if raw:
        kwargs = {  # Note: this also works with pytables
            "compression": 32001,
            "compression_opts": (0, 0, 0, 0, 7, 1, 2),
        }
    else:
        kwargs = {
            "compression": hdf5plugin.Blosc(
                cname="lz4hc", clevel=7, shuffle=hdf5plugin.Blosc.SHUFFLE
            ),
        }

    vf = VersionedHDF5File(h5file)
    with vf.stage_version("r0") as sv:
        sv.create_dataset("values", data=np.arange(10), **kwargs)

    assert h5file["_version_data/versions/r0/values"].compression is None
    raw_data = h5file["_version_data/values/raw_data"]
    assert raw_data.compression is None
    assert "32001" in raw_data._filters

    # First four numbers are reserved for blosc compression;
    # others are actual compression options
    assert raw_data._filters["32001"][4:] == (7, 1, 2)


@pytest.mark.parametrize("dtype", ["i2", int, float, np.uint8])
def test_create_dataset_dtype_arg(vfile, dtype):
    with vfile.stage_version("r0") as sv:
        dset = sv.create_dataset("x", shape=(2,), dtype=dtype)
        assert isinstance(dset.dtype, np.dtype)
        assert dset.dtype == np.dtype(dtype)

        dset = sv.create_dataset("y", data=[1, 2], dtype=dtype)
        assert isinstance(dset.dtype, np.dtype)
        assert dset.dtype == np.dtype(dtype)

    with vfile.stage_version("r1") as sv:
        dset = sv["x"]
        assert isinstance(dset.dtype, np.dtype)
        assert dset.dtype == np.dtype(dtype)

        dset = sv["y"]
        assert isinstance(dset.dtype, np.dtype)
        assert dset.dtype == np.dtype(dtype)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("fletcher32", True),
        ("scaleoffset", 5),
        ("shuffle", True),
    ],
)
@pytest.mark.parametrize("sparse", [False, True])
def test_create_dataset_other_filters(vfile, name, value, sparse):
    kwargs = {"shape": (2,)} if sparse else {"data": [1, 2]}
    kwargs[name] = value

    with vfile.stage_version("r0") as sv:
        x = sv.create_dataset("x", **kwargs)
        assert getattr(x, name) == value

    raw_data = vfile.f["_version_data"]["x"]["raw_data"]
    assert getattr(raw_data, name) == value

    with vfile.stage_version("r1") as sv:
        x = sv["x"]
        assert getattr(x, name) == value
