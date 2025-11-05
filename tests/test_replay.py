import shutil
import subprocess
from unittest import mock

import h5py
import hypothesis
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from ndindex import Slice

from versioned_hdf5 import VersionedHDF5File
from versioned_hdf5.hashtable import Hashtable
from versioned_hdf5.replay import (
    _get_parent,
    _recreate_hashtable,
    _recreate_raw_data,
    _recreate_virtual_dataset,
    delete_version,
    delete_versions,
    modify_metadata,
)


def setup_vfile(file):
    with file.stage_version("version1") as g:
        data = g.create_dataset(
            "test_data", data=None, fillvalue=1.0, shape=(10000,), chunks=(1000,)
        )
        data[0] = 0.0
        g.create_dataset("test_data2", data=[1, 2, 3], chunks=(1000,))
        group = g.create_group("group")
        group.create_dataset("test_data4", data=[1, 2, 3, 4], chunks=(1000,))

    with file.stage_version("version2") as g:
        g["test_data"][2000] = 2.0
        g.create_dataset("test_data3", data=[1, 2, 3, 4], chunks=(1000,))
        g["group"]["test_data4"][0] = 5


def check_data(file, test_data_fillvalue=1.0, version2=True, test_data4_fillvalue=0):
    assert set(file["version1"]) == {"test_data", "test_data2", "group"}
    assert file["version1"]["test_data"].shape == (10000,)
    assert file["version1"]["test_data"][0] == 0.0
    assert np.all(file["version1"]["test_data"][1:] == test_data_fillvalue)

    if version2:
        assert set(file["version2"]) == {
            "test_data",
            "test_data2",
            "test_data3",
            "group",
        }
        assert file["version2"]["test_data"].shape == (10000,)
        assert file["version2"]["test_data"][0] == 0.0
        assert np.all(file["version2"]["test_data"][1:2000] == test_data_fillvalue)
        assert file["version2"]["test_data"][2000] == 2.0
        assert np.all(file["version2"]["test_data"][2001:] == test_data_fillvalue)

    assert file["version1"]["test_data2"].shape == (3,)
    assert np.all(file["version1"]["test_data2"][:] == [1, 2, 3])

    if version2:
        assert file["version2"]["test_data2"].shape == (3,)
        assert np.all(file["version2"]["test_data2"][:] == [1, 2, 3])

    assert "test_data3" not in file["version1"]

    if version2:
        assert file["version2"]["test_data3"].shape == (4,)
        assert np.all(file["version2"]["test_data3"][:] == [1, 2, 3, 4])

    assert set(file["version1"]["group"]) == {"test_data4"}
    assert file["version1"]["group"]["test_data4"].shape == (4,)
    np.testing.assert_equal(file["version1"]["group"]["test_data4"][:4], [1, 2, 3, 4])
    assert np.all(file["version1"]["group"]["test_data4"][4:] == test_data4_fillvalue)

    if version2:
        assert set(file["version2"]["group"]) == {"test_data4"}
        assert file["version2"]["group"]["test_data4"].shape == (4,)
        np.testing.assert_equal(
            file["version2"]["group"]["test_data4"][:4], [5, 2, 3, 4]
        )
        assert np.all(
            file["version2"]["group"]["test_data4"][4:] == test_data4_fillvalue
        )


def test_modify_metadata_compression(vfile):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile["version1"]["test_data"].compression is None
    assert vfile["version2"]["test_data"].compression is None
    assert vfile["version1"]["test_data"].compression_opts is None
    assert vfile["version2"]["test_data"].compression_opts is None

    assert vfile["version1"]["test_data2"].compression is None
    assert vfile["version2"]["test_data2"].compression is None
    assert vfile["version1"]["test_data2"].compression_opts is None
    assert vfile["version2"]["test_data2"].compression_opts is None

    assert vfile["version2"]["test_data3"].compression is None
    assert vfile["version2"]["test_data3"].compression_opts is None

    assert vfile["version1"]["group"]["test_data4"].compression is None
    assert vfile["version2"]["group"]["test_data4"].compression is None
    assert vfile["version1"]["group"]["test_data4"].compression_opts is None
    assert vfile["version2"]["group"]["test_data4"].compression_opts is None

    assert f["_version_data"]["test_data"]["raw_data"].compression is None
    assert f["_version_data"]["test_data2"]["raw_data"].compression is None
    assert f["_version_data"]["test_data3"]["raw_data"].compression is None
    assert f["_version_data"]["group"]["test_data4"]["raw_data"].compression is None

    assert f["_version_data"]["test_data"]["raw_data"].compression_opts is None
    assert f["_version_data"]["test_data2"]["raw_data"].compression_opts is None
    assert f["_version_data"]["test_data3"]["raw_data"].compression_opts is None
    assert (
        f["_version_data"]["group"]["test_data4"]["raw_data"].compression_opts is None
    )

    modify_metadata(f, "test_data2", compression="gzip", compression_opts=3)
    check_data(vfile)

    assert vfile["version1"]["test_data"].compression is None
    assert vfile["version2"]["test_data"].compression is None
    assert vfile["version1"]["test_data"].compression_opts is None
    assert vfile["version2"]["test_data"].compression_opts is None

    assert vfile["version1"]["test_data2"].compression == "gzip"
    assert vfile["version2"]["test_data2"].compression == "gzip"
    assert vfile["version1"]["test_data2"].compression_opts == 3
    assert vfile["version2"]["test_data2"].compression_opts == 3

    assert vfile["version2"]["test_data3"].compression is None
    assert vfile["version2"]["test_data3"].compression_opts is None

    assert vfile["version1"]["group"]["test_data4"].compression is None
    assert vfile["version2"]["group"]["test_data4"].compression is None
    assert vfile["version1"]["group"]["test_data4"].compression_opts is None
    assert vfile["version2"]["group"]["test_data4"].compression_opts is None

    assert f["_version_data"]["test_data"]["raw_data"].compression is None
    assert f["_version_data"]["test_data2"]["raw_data"].compression == "gzip"
    assert f["_version_data"]["test_data3"]["raw_data"].compression is None
    assert f["_version_data"]["group"]["test_data4"]["raw_data"].compression is None

    assert f["_version_data"]["test_data"]["raw_data"].compression_opts is None
    assert f["_version_data"]["test_data2"]["raw_data"].compression_opts == 3
    assert f["_version_data"]["test_data3"]["raw_data"].compression_opts is None
    assert (
        f["_version_data"]["group"]["test_data4"]["raw_data"].compression_opts is None
    )

    # Make sure the tmp group group has been destroyed.
    assert set(f["_version_data"]) == {
        "test_data",
        "test_data2",
        "test_data3",
        "group",
        "versions",
    }
    assert set(f["_version_data"]["group"]) == {"test_data4"}


def test_modify_metadata_compression2(vfile):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile["version1"]["test_data"].compression is None
    assert vfile["version2"]["test_data"].compression is None
    assert vfile["version1"]["test_data"].compression_opts is None
    assert vfile["version2"]["test_data"].compression_opts is None

    assert vfile["version1"]["test_data2"].compression is None
    assert vfile["version2"]["test_data2"].compression is None
    assert vfile["version1"]["test_data2"].compression_opts is None
    assert vfile["version2"]["test_data2"].compression_opts is None

    assert vfile["version2"]["test_data3"].compression is None
    assert vfile["version2"]["test_data3"].compression_opts is None

    assert vfile["version1"]["group"]["test_data4"].compression is None
    assert vfile["version2"]["group"]["test_data4"].compression is None
    assert vfile["version1"]["group"]["test_data4"].compression_opts is None
    assert vfile["version2"]["group"]["test_data4"].compression_opts is None

    assert f["_version_data"]["test_data"]["raw_data"].compression is None
    assert f["_version_data"]["test_data2"]["raw_data"].compression is None
    assert f["_version_data"]["test_data3"]["raw_data"].compression is None
    assert f["_version_data"]["group"]["test_data4"]["raw_data"].compression is None

    assert f["_version_data"]["test_data"]["raw_data"].compression_opts is None
    assert f["_version_data"]["test_data2"]["raw_data"].compression_opts is None
    assert f["_version_data"]["test_data3"]["raw_data"].compression_opts is None
    assert (
        f["_version_data"]["group"]["test_data4"]["raw_data"].compression_opts is None
    )

    modify_metadata(f, "group/test_data4", compression="gzip", compression_opts=3)
    check_data(vfile)

    assert vfile["version1"]["test_data"].compression is None
    assert vfile["version2"]["test_data"].compression is None
    assert vfile["version1"]["test_data"].compression_opts is None
    assert vfile["version2"]["test_data"].compression_opts is None

    assert vfile["version1"]["test_data2"].compression is None
    assert vfile["version2"]["test_data2"].compression is None
    assert vfile["version1"]["test_data2"].compression_opts is None
    assert vfile["version2"]["test_data2"].compression_opts is None

    assert vfile["version2"]["test_data3"].compression is None
    assert vfile["version2"]["test_data3"].compression_opts is None

    assert vfile["version1"]["group"]["test_data4"].compression == "gzip"
    assert vfile["version2"]["group"]["test_data4"].compression == "gzip"
    assert vfile["version1"]["group"]["test_data4"].compression_opts == 3
    assert vfile["version2"]["group"]["test_data4"].compression_opts == 3

    assert f["_version_data"]["test_data"]["raw_data"].compression is None
    assert f["_version_data"]["test_data2"]["raw_data"].compression is None
    assert f["_version_data"]["test_data3"]["raw_data"].compression is None
    assert f["_version_data"]["group"]["test_data4"]["raw_data"].compression == "gzip"

    assert f["_version_data"]["test_data"]["raw_data"].compression_opts is None
    assert f["_version_data"]["test_data2"]["raw_data"].compression_opts is None
    assert f["_version_data"]["test_data3"]["raw_data"].compression_opts is None
    assert f["_version_data"]["group"]["test_data4"]["raw_data"].compression_opts == 3

    # Make sure the tmp group group has been destroyed.
    assert set(f["_version_data"]) == {
        "test_data",
        "test_data2",
        "test_data3",
        "group",
        "versions",
    }
    assert set(f["_version_data"]["group"]) == {"test_data4"}


def test_modify_metadata_chunks(vfile):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile["version1"]["test_data"].chunks == (1000,)
    assert vfile["version2"]["test_data"].chunks == (1000,)

    assert vfile["version1"]["test_data2"].chunks == (1000,)
    assert vfile["version2"]["test_data2"].chunks == (1000,)

    assert vfile["version2"]["test_data3"].chunks == (1000,)

    assert vfile["version1"]["group"]["test_data4"].chunks == (1000,)
    assert vfile["version2"]["group"]["test_data4"].chunks == (1000,)

    assert f["_version_data"]["test_data"]["raw_data"].chunks == (1000,)
    assert f["_version_data"]["test_data2"]["raw_data"].chunks == (1000,)
    assert f["_version_data"]["test_data3"]["raw_data"].chunks == (1000,)
    assert f["_version_data"]["group"]["test_data4"]["raw_data"].chunks == (1000,)

    modify_metadata(f, "test_data2", chunks=(500,))
    check_data(vfile)

    assert vfile["version1"]["test_data"].chunks == (1000,)
    assert vfile["version2"]["test_data"].chunks == (1000,)

    assert vfile["version1"]["test_data2"].chunks == (500,)
    assert vfile["version2"]["test_data2"].chunks == (500,)

    assert vfile["version2"]["test_data3"].chunks == (1000,)

    assert vfile["version1"]["group"]["test_data4"].chunks == (1000,)
    assert vfile["version2"]["group"]["test_data4"].chunks == (1000,)

    assert f["_version_data"]["test_data"]["raw_data"].chunks == (1000,)
    assert f["_version_data"]["test_data2"]["raw_data"].chunks == (500,)
    assert f["_version_data"]["test_data3"]["raw_data"].chunks == (1000,)
    assert f["_version_data"]["group"]["test_data4"]["raw_data"].chunks == (1000,)

    # Make sure the tmp group group has been destroyed.
    assert set(f["_version_data"]) == {
        "test_data",
        "test_data2",
        "test_data3",
        "group",
        "versions",
    }
    assert set(f["_version_data"]["group"]) == {"test_data4"}


def test_modify_metadata_chunk2(vfile):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile["version1"]["test_data"].chunks == (1000,)
    assert vfile["version2"]["test_data"].chunks == (1000,)

    assert vfile["version1"]["test_data2"].chunks == (1000,)
    assert vfile["version2"]["test_data2"].chunks == (1000,)

    assert vfile["version2"]["test_data3"].chunks == (1000,)

    assert vfile["version1"]["group"]["test_data4"].chunks == (1000,)
    assert vfile["version2"]["group"]["test_data4"].chunks == (1000,)

    assert f["_version_data"]["test_data"]["raw_data"].chunks == (1000,)
    assert f["_version_data"]["test_data2"]["raw_data"].chunks == (1000,)
    assert f["_version_data"]["test_data3"]["raw_data"].chunks == (1000,)
    assert f["_version_data"]["group"]["test_data4"]["raw_data"].chunks == (1000,)

    modify_metadata(f, "group/test_data4", chunks=(500,))
    check_data(vfile)

    assert vfile["version1"]["test_data"].chunks == (1000,)
    assert vfile["version2"]["test_data"].chunks == (1000,)

    assert vfile["version1"]["test_data2"].chunks == (1000,)
    assert vfile["version2"]["test_data2"].chunks == (1000,)

    assert vfile["version2"]["test_data3"].chunks == (1000,)

    assert vfile["version1"]["group"]["test_data4"].chunks == (500,)
    assert vfile["version2"]["group"]["test_data4"].chunks == (500,)

    assert f["_version_data"]["test_data"]["raw_data"].chunks == (1000,)
    assert f["_version_data"]["test_data2"]["raw_data"].chunks == (1000,)
    assert f["_version_data"]["test_data3"]["raw_data"].chunks == (1000,)
    assert f["_version_data"]["group"]["test_data4"]["raw_data"].chunks == (500,)

    # Make sure the tmp group group has been destroyed.
    assert set(f["_version_data"]) == {
        "test_data",
        "test_data2",
        "test_data3",
        "group",
        "versions",
    }
    assert set(f["_version_data"]["group"]) == {"test_data4"}


def test_modify_metadata_dtype(vfile):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile["version1"]["test_data"].dtype == np.float64
    assert vfile["version2"]["test_data"].dtype == np.float64

    assert vfile["version1"]["test_data2"].dtype == np.int64
    assert vfile["version2"]["test_data2"].dtype == np.int64

    assert vfile["version2"]["test_data3"].dtype == np.int64

    assert vfile["version1"]["group"]["test_data4"].dtype == np.int64
    assert vfile["version2"]["group"]["test_data4"].dtype == np.int64

    assert f["_version_data"]["test_data"]["raw_data"].dtype == np.float64
    assert f["_version_data"]["test_data2"]["raw_data"].dtype == np.int64
    assert f["_version_data"]["test_data3"]["raw_data"].dtype == np.int64
    assert f["_version_data"]["group"]["test_data4"]["raw_data"].dtype == np.int64

    modify_metadata(f, "test_data", dtype=np.float32)  # sparse dataset
    modify_metadata(f, "test_data2", dtype=np.float64)  # dense dataset
    check_data(vfile)

    assert vfile["version1"]["test_data"].dtype == np.float32
    assert vfile["version2"]["test_data"].dtype == np.float32

    assert vfile["version1"]["test_data2"].dtype == np.float64
    assert vfile["version2"]["test_data2"].dtype == np.float64

    assert vfile["version2"]["test_data3"].dtype == np.int64

    assert vfile["version1"]["group"]["test_data4"].dtype == np.int64
    assert vfile["version2"]["group"]["test_data4"].dtype == np.int64

    assert f["_version_data"]["test_data"]["raw_data"].dtype == np.float32
    assert f["_version_data"]["test_data2"]["raw_data"].dtype == np.float64
    assert f["_version_data"]["test_data3"]["raw_data"].dtype == np.int64
    assert f["_version_data"]["group"]["test_data4"]["raw_data"].dtype == np.int64

    # Make sure the tmp group group has been destroyed.
    assert set(f["_version_data"]) == {
        "test_data",
        "test_data2",
        "test_data3",
        "group",
        "versions",
    }
    assert set(f["_version_data"]["group"]) == {"test_data4"}


def test_modify_metadata_dtype2(vfile):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile["version1"]["test_data"].dtype == np.float64
    assert vfile["version2"]["test_data"].dtype == np.float64

    assert vfile["version1"]["test_data2"].dtype == np.int64
    assert vfile["version2"]["test_data2"].dtype == np.int64

    assert vfile["version2"]["test_data3"].dtype == np.int64

    assert vfile["version1"]["group"]["test_data4"].dtype == np.int64
    assert vfile["version2"]["group"]["test_data4"].dtype == np.int64

    assert f["_version_data"]["test_data"]["raw_data"].dtype == np.float64
    assert f["_version_data"]["test_data2"]["raw_data"].dtype == np.int64
    assert f["_version_data"]["test_data3"]["raw_data"].dtype == np.int64
    assert f["_version_data"]["group"]["test_data4"]["raw_data"].dtype == np.int64

    modify_metadata(f, "group/test_data4", dtype=np.float64)
    check_data(vfile)

    assert vfile["version1"]["test_data"].dtype == np.float64
    assert vfile["version2"]["test_data"].dtype == np.float64

    assert vfile["version1"]["test_data2"].dtype == np.int64
    assert vfile["version2"]["test_data2"].dtype == np.int64

    assert vfile["version2"]["test_data3"].dtype == np.int64

    assert vfile["version1"]["group"]["test_data4"].dtype == np.float64
    assert vfile["version2"]["group"]["test_data4"].dtype == np.float64

    assert f["_version_data"]["test_data"]["raw_data"].dtype == np.float64
    assert f["_version_data"]["test_data2"]["raw_data"].dtype == np.int64
    assert f["_version_data"]["test_data3"]["raw_data"].dtype == np.int64
    assert f["_version_data"]["group"]["test_data4"]["raw_data"].dtype == np.float64

    # Make sure the tmp group group has been destroyed.
    assert set(f["_version_data"]) == {
        "test_data",
        "test_data2",
        "test_data3",
        "group",
        "versions",
    }
    assert set(f["_version_data"]["group"]) == {"test_data4"}


@pytest.mark.parametrize("new_fillvalue", [0, 3])
def test_modify_metadata_fillvalue1(vfile, new_fillvalue):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile["version1"]["test_data"].fillvalue == 1.0
    assert vfile["version2"]["test_data"].fillvalue == 1.0

    assert vfile["version1"]["test_data2"].fillvalue == 0
    assert vfile["version2"]["test_data2"].fillvalue == 0

    assert vfile["version2"]["test_data3"].fillvalue == 0

    assert vfile["version1"]["group"]["test_data4"].fillvalue == 0
    assert vfile["version2"]["group"]["test_data4"].fillvalue == 0

    assert f["_version_data"]["test_data"]["raw_data"].fillvalue == 1.0
    assert f["_version_data"]["test_data2"]["raw_data"].fillvalue == 0
    assert f["_version_data"]["test_data3"]["raw_data"].fillvalue == 0
    assert f["_version_data"]["group"]["test_data4"]["raw_data"].fillvalue == 0

    modify_metadata(f, "test_data", fillvalue=new_fillvalue)
    check_data(vfile, test_data_fillvalue=new_fillvalue)

    assert vfile["version1"]["test_data"].fillvalue == new_fillvalue
    assert vfile["version2"]["test_data"].fillvalue == new_fillvalue

    assert vfile["version1"]["test_data2"].fillvalue == 0
    assert vfile["version2"]["test_data2"].fillvalue == 0

    assert vfile["version2"]["test_data3"].fillvalue == 0

    assert vfile["version1"]["group"]["test_data4"].fillvalue == 0
    assert vfile["version2"]["group"]["test_data4"].fillvalue == 0

    assert f["_version_data"]["test_data"]["raw_data"].fillvalue == new_fillvalue
    assert f["_version_data"]["test_data2"]["raw_data"].fillvalue == 0
    assert f["_version_data"]["test_data3"]["raw_data"].fillvalue == 0
    assert f["_version_data"]["group"]["test_data4"]["raw_data"].fillvalue == 0

    # Make sure the tmp group group has been destroyed.
    assert set(f["_version_data"]) == {
        "test_data",
        "test_data2",
        "test_data3",
        "group",
        "versions",
    }
    assert set(f["_version_data"]["group"]) == {"test_data4"}


def test_modify_metadata_fillvalue2(vfile):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile["version1"]["test_data"].fillvalue == 1.0
    assert vfile["version2"]["test_data"].fillvalue == 1.0

    assert vfile["version1"]["test_data2"].fillvalue == 0
    assert vfile["version2"]["test_data2"].fillvalue == 0

    assert vfile["version2"]["test_data3"].fillvalue == 0

    assert vfile["version1"]["group"]["test_data4"].fillvalue == 0
    assert vfile["version2"]["group"]["test_data4"].fillvalue == 0

    assert f["_version_data"]["test_data"]["raw_data"].fillvalue == 1.0
    assert f["_version_data"]["test_data2"]["raw_data"].fillvalue == 0
    assert f["_version_data"]["test_data3"]["raw_data"].fillvalue == 0
    assert f["_version_data"]["group"]["test_data4"]["raw_data"].fillvalue == 0

    modify_metadata(f, "test_data2", fillvalue=3)
    check_data(vfile)

    assert vfile["version1"]["test_data"].fillvalue == 1.0
    assert vfile["version2"]["test_data"].fillvalue == 1.0

    assert vfile["version1"]["test_data2"].fillvalue == 3
    assert vfile["version2"]["test_data2"].fillvalue == 3

    assert vfile["version2"]["test_data3"].fillvalue == 0

    assert vfile["version1"]["group"]["test_data4"].fillvalue == 0
    assert vfile["version2"]["group"]["test_data4"].fillvalue == 0

    assert f["_version_data"]["test_data"]["raw_data"].fillvalue == 1.0
    assert f["_version_data"]["test_data2"]["raw_data"].fillvalue == 3
    assert f["_version_data"]["test_data3"]["raw_data"].fillvalue == 0
    assert f["_version_data"]["group"]["test_data4"]["raw_data"].fillvalue == 0

    # Make sure the tmp group group has been destroyed.
    assert set(f["_version_data"]) == {
        "test_data",
        "test_data2",
        "test_data3",
        "group",
        "versions",
    }
    assert set(f["_version_data"]["group"]) == {"test_data4"}


def test_modify_metadata_fillvalue3(vfile):
    setup_vfile(vfile)

    f = vfile.f

    assert vfile["version1"]["test_data"].fillvalue == 1.0
    assert vfile["version2"]["test_data"].fillvalue == 1.0

    assert vfile["version1"]["test_data2"].fillvalue == 0
    assert vfile["version2"]["test_data2"].fillvalue == 0

    assert vfile["version2"]["test_data3"].fillvalue == 0

    assert vfile["version1"]["group"]["test_data4"].fillvalue == 0
    assert vfile["version2"]["group"]["test_data4"].fillvalue == 0

    assert f["_version_data"]["test_data"]["raw_data"].fillvalue == 1.0
    assert f["_version_data"]["test_data2"]["raw_data"].fillvalue == 0
    assert f["_version_data"]["test_data3"]["raw_data"].fillvalue == 0
    assert f["_version_data"]["group"]["test_data4"]["raw_data"].fillvalue == 0

    modify_metadata(f, "group/test_data4", fillvalue=2)
    check_data(vfile)

    assert vfile["version1"]["test_data"].fillvalue == 1.0
    assert vfile["version2"]["test_data"].fillvalue == 1.0

    assert vfile["version1"]["test_data2"].fillvalue == 0
    assert vfile["version2"]["test_data2"].fillvalue == 0

    assert vfile["version2"]["test_data3"].fillvalue == 0

    assert vfile["version1"]["group"]["test_data4"].fillvalue == 2
    assert vfile["version2"]["group"]["test_data4"].fillvalue == 2

    assert f["_version_data"]["test_data"]["raw_data"].fillvalue == 1.0
    assert f["_version_data"]["test_data2"]["raw_data"].fillvalue == 0
    assert f["_version_data"]["test_data3"]["raw_data"].fillvalue == 0
    assert f["_version_data"]["group"]["test_data4"]["raw_data"].fillvalue == 2

    # Make sure the tmp group group has been destroyed.
    assert set(f["_version_data"]) == {
        "test_data",
        "test_data2",
        "test_data3",
        "group",
        "versions",
    }
    assert set(f["_version_data"]["group"]) == {"test_data4"}


def test_modify_metadata_dtype_fillvalue(vfile):
    """Test calling modify_metadata() to change both dtype and fillvalue,
    and the new fillvalue would not fit in the old dtype, so it needs to
    be updated _after_ changing the dtype.
    """
    setup_vfile(vfile)

    assert vfile["version1"]["test_data"].dtype == np.float64
    assert vfile["version2"]["test_data"].dtype == np.float64
    assert vfile["version1"]["test_data"].fillvalue == 1.0
    assert vfile["version2"]["test_data"].fillvalue == 1.0

    assert vfile["version1"]["test_data2"].dtype == np.int64
    assert vfile["version2"]["test_data2"].dtype == np.int64
    assert vfile["version1"]["test_data2"].fillvalue == 0
    assert vfile["version2"]["test_data2"].fillvalue == 0

    # Integer near 2**63; loses precision when converted to float64
    huge_int = 9223372036854775807
    assert int(float(huge_int)) != huge_int

    modify_metadata(vfile, "test_data", dtype=np.int64, fillvalue=huge_int)
    modify_metadata(vfile, "test_data2", dtype=np.float32, fillvalue=3.14)  # Was int64
    check_data(vfile, test_data_fillvalue=huge_int)

    assert vfile["version1"]["test_data"].dtype == np.int64
    assert vfile["version2"]["test_data"].dtype == np.int64
    assert vfile["version1"]["test_data"].fillvalue.dtype == np.int64
    assert vfile["version2"]["test_data"].fillvalue.dtype == np.int64
    assert vfile["version1"]["test_data"].fillvalue == huge_int
    assert vfile["version2"]["test_data"].fillvalue == huge_int

    assert vfile["version1"]["test_data2"].dtype == np.float32
    assert vfile["version2"]["test_data2"].dtype == np.float32
    np.testing.assert_allclose(vfile["version1"]["test_data2"].fillvalue, 3.14)
    np.testing.assert_allclose(vfile["version2"]["test_data2"].fillvalue, 3.14)


def test_delete_version(vfile):
    setup_vfile(vfile)
    f = vfile.f

    delete_version(f, "version2")
    check_data(vfile, version2=False)
    assert list(vfile) == ["version1"]
    assert set(f["_version_data"]) == {"group", "test_data", "test_data2", "versions"}
    assert set(f["_version_data"]["group"]) == {"test_data4"}
    assert not np.isin(2.0, f["_version_data"]["test_data"]["raw_data"][:])
    assert not np.isin(5, f["_version_data"]["group"]["test_data4"]["raw_data"][:])


def test_delete_versions(vfile):
    setup_vfile(vfile)
    with vfile.stage_version("version3") as g:
        g["test_data"][2000] = 3.0
        g.create_dataset("test_data4", data=[1, 2, 3, 4], chunks=(1000,))
    f = vfile.f

    delete_versions(f, ["version2", "version3"])

    check_data(vfile, version2=False)
    assert list(vfile) == ["version1"]
    assert set(f["_version_data"]) == {"group", "test_data", "test_data2", "versions"}
    assert set(f["_version_data"]["group"]) == {"test_data4"}
    assert not np.isin(2.0, f["_version_data"]["test_data"]["raw_data"][:])
    assert not np.isin(5, f["_version_data"]["group"]["test_data4"]["raw_data"][:])


def test_delete_versions_no_data(vfile):
    with vfile.stage_version("version1") as g:
        g.create_dataset(
            "data",
            maxshape=(None, None),
            chunks=(20, 20),
            shape=(5, 5),
            dtype=np.dtype("int8"),
            fillvalue=0,
        )

    with vfile.stage_version("version2") as g:
        g["data"][0] = 1

    f = vfile.f

    delete_versions(f, ["version2"])
    assert list(vfile) == ["version1"]
    assert list(vfile["version1"]) == ["data"]
    assert vfile["version1"]["data"].shape == (5, 5)
    assert np.all(vfile["version1"]["data"][:] == 0)


def test_delete_versions_no_data2(vfile):
    with vfile.stage_version("version1") as g:
        g.create_dataset(
            "data",
            maxshape=(None, None),
            chunks=(20, 20),
            shape=(5, 5),
            dtype=np.dtype("int8"),
            fillvalue=0,
        )

    with vfile.stage_version("version2") as g:
        g["data"][0] = 1

    f = vfile.f

    delete_versions(f, ["version1"])
    assert list(vfile) == ["version2"]
    assert list(vfile["version2"]) == ["data"]
    assert vfile["version2"]["data"].shape == (5, 5)
    assert np.all(vfile["version2"]["data"][1:] == 0)
    assert np.all(vfile["version2"]["data"][0] == 1)


def test_delete_versions_nested_groups(vfile):
    rng = np.random.default_rng(42)
    data = []

    with vfile.stage_version("r0") as sv:
        data_group = sv.create_group("group1/group2")
        data.append(np.arange(500))
        data_group.create_dataset(
            "test_data", maxshape=(None,), chunks=(1000), data=data[0]
        )

    for i in range(1, 11):
        with vfile.stage_version(f"r{i}") as sv:
            data.append(rng.integers(0, 1000, size=500))
            sv["group1"]["group2"]["test_data"][:] = data[-1]

    assert set(vfile) == {
        "r0",
        "r1",
        "r2",
        "r3",
        "r4",
        "r5",
        "r6",
        "r7",
        "r8",
        "r9",
        "r10",
    }
    for i in range(11):
        assert list(vfile[f"r{i}"]) == ["group1"], i
        assert list(vfile[f"r{i}"]["group1"]) == ["group2"]
        assert list(vfile[f"r{i}"]["group1"]["group2"]) == ["test_data"]
        np.testing.assert_equal(
            vfile[f"r{i}"]["group1"]["group2"]["test_data"][:], data[i]
        )

    delete_versions(vfile, ["r3", "r6"])

    assert set(vfile) == {"r0", "r1", "r2", "r4", "r5", "r7", "r8", "r9", "r10"}
    for i in range(11):
        if i in [3, 6]:
            continue
        assert list(vfile[f"r{i}"]) == ["group1"], i
        assert list(vfile[f"r{i}"]["group1"]) == ["group2"]
        assert list(vfile[f"r{i}"]["group1"]["group2"]) == ["test_data"]
        np.testing.assert_equal(
            vfile[f"r{i}"]["group1"]["group2"]["test_data"][:], data[i]
        )


def test_delete_versions_prev_version(vfile):
    with vfile.stage_version("r0") as g:
        g["foo"] = np.array([1, 2, 3])
    for i in range(1, 11):
        with vfile.stage_version(f"r{i}") as g:
            g["foo"][:] = np.array([1, i, 3])

    delete_versions(vfile, ["r1", "r5", "r8"])
    prev_versions = {
        "__first_version__": None,
        "r0": "__first_version__",
        "r2": "r0",
        "r3": "r2",
        "r4": "r3",
        "r6": "r4",
        "r7": "r6",
        "r9": "r7",
        "r10": "r9",
    }

    for v in vfile:
        assert vfile[v].attrs["prev_version"] == prev_versions[v]


def setup2(vfile):
    with vfile.stage_version("version1") as g:
        g.create_dataset(
            "test_data", data=np.arange(20000).reshape((1000, 20)), chunks=(101, 11)
        )

    with vfile.stage_version("version2") as g:
        g["test_data"][::200] = -g["test_data"][::200]


def test_recreate_raw_data(vfile):
    setup2(vfile)
    raw_data = vfile.f["_version_data/test_data/raw_data"][:]
    assert raw_data.shape == (3030, 11)

    chunks_map = _recreate_raw_data(vfile.f, "test_data", ["version1"])
    new_raw_data = vfile.f["_version_data/test_data/raw_data"][:]

    assert len(chunks_map) == 20

    for old, new in chunks_map.items():
        a = raw_data[old.raw]
        b = new_raw_data[new.raw]
        assert a.shape == b.shape
        np.testing.assert_equal(a, b)


def test_recreate_hashtable(vfile):
    setup2(vfile)
    chunks_map = _recreate_raw_data(vfile.f, "test_data", ["version1"])

    # Recreate a separate, independent version, with the dataset as it would
    # be with version1 deleted.
    with vfile.stage_version("version2_2", prev_version="") as g:
        g.create_dataset(
            "test_data2", data=np.arange(20000).reshape((1000, 20)), chunks=(101, 11)
        )
        g["test_data2"][::200] = -g["test_data2"][::200]

    _recreate_hashtable(vfile.f, "test_data", chunks_map, tmp=True)

    new_hash_table = Hashtable(vfile.f, "test_data", hash_table_name="_tmp_hash_table")

    new_hash_table2 = Hashtable(vfile.f, "test_data2")
    d1 = dict(new_hash_table)
    d2 = dict(new_hash_table2)
    assert d1.keys() == d2.keys()

    # The exact slices won't be the same because raw data won't be in the same
    # order
    for h in d1:
        np.testing.assert_equal(
            vfile.f["_version_data/test_data/raw_data"][d1[h].raw],
            vfile.f["_version_data/test_data2/raw_data"][d2[h].raw],
        )


def test_recreate_virtual_dataset(vfile):
    setup2(vfile)
    orig_virtual_dataset = vfile.f["_version_data/versions/version2/test_data"][:]

    chunks_map = _recreate_raw_data(vfile.f, "test_data", ["version1"])

    _recreate_hashtable(vfile.f, "test_data", chunks_map, tmp=False)

    _recreate_virtual_dataset(vfile.f, "test_data", ["version2"], chunks_map, tmp=True)

    new_virtual_dataset = vfile.f["_version_data/versions/version2/_tmp_test_data"][:]

    np.testing.assert_equal(orig_virtual_dataset, new_virtual_dataset)


def test_delete_versions2(vfile):
    setup2(vfile)
    data = np.arange(20000).reshape((1000, 20))
    data[::200] = -data[::200]

    assert vfile["version2"]["test_data"].shape == data.shape
    delete_versions(vfile, ["version1"])
    assert list(vfile) == ["version2"]
    assert list(vfile["version2"]) == ["test_data"]
    assert vfile["version2"]["test_data"].shape == data.shape
    np.testing.assert_equal(vfile["version2"]["test_data"][:], data)
    assert set(vfile.f["_version_data/test_data/raw_data"][:].flat) == set(data.flat)


def test_delete_versions_variable_length_strings(vfile):
    with vfile.stage_version("r0") as sv:
        data = np.array(["foo"], dtype="O")
        sv.create_dataset("bar", data=data, dtype=h5py.string_dtype(encoding="ascii"))

    for i in range(1, 11):
        with vfile.stage_version("r{}".format(i)) as sv:
            sv["bar"].resize((i + 1,))
            sv["bar"][i] = "foo"

    delete_versions(vfile, ["r2", "r4", "r6"])


def test_delete_versions_fillvalue_only_dataset(vfile):
    with vfile.stage_version("r0") as sv:
        sv.create_dataset(
            "fillvalue_only",
            shape=(6,),
            dtype=np.dtype("int64"),
            data=None,
            maxshape=(None,),
            chunks=(10000,),
            fillvalue=0,
        )
        sv.create_dataset(
            "has_data",
            shape=(6,),
            dtype=np.dtype("int64"),
            data=np.arange(6),
            maxshape=(None,),
            chunks=(10000,),
            fillvalue=0,
        )

    with vfile.stage_version("r1") as sv:
        sv["has_data"] = np.arange(5, -1, -1)

    delete_versions(vfile, ["r0"])

    with vfile.stage_version("r2") as sv:
        sv["fillvalue_only"][0] = 1

    assert set(vfile) == {"r1", "r2"}
    assert set(vfile["r1"]) == {"fillvalue_only", "has_data"}
    assert set(vfile["r2"]) == {"fillvalue_only", "has_data"}
    np.testing.assert_equal(vfile["r1"]["fillvalue_only"][:], 0)
    np.testing.assert_equal(
        vfile["r2"]["fillvalue_only"][:], np.array([1, 0, 0, 0, 0, 0])
    )
    np.testing.assert_equal(vfile["r1"]["has_data"][:], np.arange(5, -1, -1))
    np.testing.assert_equal(vfile["r2"]["has_data"][:], np.arange(5, -1, -1))


def test_delete_versions_current_version(vfile):
    with vfile.stage_version("r0") as sv:
        sv.create_dataset("bar", data=np.arange(10))

    for i in range(1, 11):
        with vfile.stage_version("r{}".format(i)) as sv:
            sv["bar"] = np.arange(10 + i)

    delete_versions(vfile, ["r2", "r4", "r6", "r8", "r9", "r10"])

    cv = vfile.current_version
    assert cv == "r7"
    np.testing.assert_equal(vfile[cv]["bar"][:], np.arange(17))


def test_delete_variable_length_strings(vfile):
    with vfile.stage_version("r0") as sv:
        g = sv.create_group("data")
        dt = h5py.string_dtype(encoding="ascii")
        g.create_dataset("foo", data=["foo", "bar"], dtype=dt)

    for i in range(1, 7):
        with vfile.stage_version(f"r{i}") as sv:
            sv["data/foo"] = np.array([f"foo{i}", f"bar{i}"], dtype="O")

    delete_versions(vfile, ["r1"])


def test_delete_empty_dataset(vfile):
    """Test that deleting an empty dataset executes successfully."""
    with vfile.stage_version("r0") as sv:
        sv.create_dataset(
            "key0",
            data=np.array([]),
            maxshape=(None,),
            chunks=(10000,),
            compression="lzf",
        )

    # Raw data should be filled with fillvalue, but actual current
    # version dataset should have size 0.
    assert vfile.f["_version_data/key0/raw_data"][:].size == 10000
    assert vfile[vfile.current_version]["key0"][:].size == 0

    # Create a new version, checking again the size
    with vfile.stage_version("r1") as sv:
        sv["key0"].resize((0,))
    assert vfile.f["_version_data/key0/raw_data"][:].size == 10000
    assert vfile[vfile.current_version]["key0"][:].size == 0

    # Deleting a prior version should not change the data in the current version
    delete_versions(vfile, ["r0"])
    assert vfile.f["_version_data/key0/raw_data"][:].size == 10000
    assert vfile[vfile.current_version]["key0"][:].size == 0

    # Create a new version, then check if the data is the correct size
    with vfile.stage_version("r2") as sv:
        sv["key0"].resize((0,))

    assert vfile.f["_version_data/key0/raw_data"][:].size == 10000
    assert vfile[vfile.current_version]["key0"][:].size == 0


@pytest.mark.skipif(shutil.which("h5repack") is None, reason="Requires h5repack to run")
def test_delete_string_dataset(tmp_path):
    """Test that delete_versions + h5repack works correctly for variable length string
    dtypes.

    When calling delete_versions, the dataset must be reconstructed from the remaining
    versions using a NoneType fillvalue. However, because we can't store a NoneType for
    the fillvalue of the dataset in the h5 file, it is instead stored as b''. Previously
    a bug in delete_versions would recreate the datset using the file's fillvalue of
    b'' rather than None, corrupting the data. See https://github.com/h5py/h5py/issues/941
    for more information about the bug in h5py responsible for this, and
    https://github.com/deshaw/versioned-hdf5/issues/238 for the versioned-hdf5
    discussion.
    """
    fname1 = tmp_path / "file1.h5"
    fname2 = tmp_path / "file2.h5"

    # Create two versions of variable-length string data, then delete the first,
    # forcing a reconstruction of the data
    with h5py.File(fname1, "w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("r0") as sv:
            sv.create_dataset(
                "foo",
                data=["abc"],
                chunks=(100,),
                dtype=h5py.string_dtype(encoding="ascii"),
            )

        with vf.stage_version("r1") as sv:
            sv["foo"][0] = "def"

        delete_versions(f, "r0")

    # Repack the data; the RuntimeError only appears if you do this
    subprocess.check_call(["h5repack", fname1, fname2])

    # Check that staging a new version after delete_versions + h5repack works
    with h5py.File(fname2, "r+") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("r3") as sv:
            pass


def test_delete_versions_speed(vfile):
    """Test that delete_versions only needs linear time to find the previous version
    for the versions that are being kept.
    """
    with vfile.stage_version("r0") as sv:
        sv.create_dataset(
            "values",
            data=np.zeros(100),
            fillvalue=0,
            chunks=(300,),
            maxshape=(None,),
            compression="lzf",
        )

    for i in range(1, 100):
        with vfile.stage_version(f"r{i}") as sv:
            sv["values"][:] = np.arange(i, i + 100)

    # keep only every 10th version
    versions_to_delete = []
    versions = sorted(
        [(v, vfile._versions[v].attrs["timestamp"]) for v in vfile._versions],
        key=lambda t: t[1],
    )
    for i, v in enumerate(versions):
        if i % 10 != 0:
            versions_to_delete.append(v[0])

    with mock.patch(
        "versioned_hdf5.replay._get_parent", wraps=_get_parent
    ) as mock_get_parent:
        delete_versions(vfile, versions_to_delete)

    # There are 90 versions to delete, and 10 to keep. Each of the 10 we are
    # keeping has to go up 9 versions from it's current previous version, for
    # a total of 90 calls.
    assert mock_get_parent.call_count == 90


def test_delete_versions_after_shrinking(vfile):
    """Test that if you shrink a dataset so that an edge chunk contains the same data of
    the previous edge chunk on disk, but trimmed to the new size, then you end up with a
    full copy of the edge chunk and you can safely delete the previous, larger version
    of it.

    See Also
    --------
    https://github.com/deshaw/versioned-hdf5/issues/411
    test_delete_versions_after_updates
    test_staged_changes::test_shrinking_does_not_reuse_partial_chunks
    """
    with vfile.stage_version("r1") as sv:
        sv.create_dataset("values", data=np.arange(26), chunks=(10,))
    with vfile.stage_version("r2") as sv:
        sv["values"].resize((17,))

    ht_before = Hashtable(vfile.f, "values").inverse()
    assert ht_before.keys() == {
        Slice(0, 10, 1),  # r1
        Slice(10, 20, 1),  # r1
        Slice(20, 26, 1),  # r1
        # Shrinking the r1[10:20] chunk triggered a deep copy of the remaining [10:17],
        # and now it has its own hash key and a non-overlapping slice, even if the
        # shared area is identical.
        Slice(30, 37, 1),  # r2
    }
    assert (ht_before[Slice(10, 20, 1)] != Slice(30, 37, 1)).any()
    raw_data = vfile.f["_version_data/values/raw_data"][:]
    np.testing.assert_equal(raw_data[30:37], raw_data[10:17])

    delete_versions(vfile, ["r1"])
    np.testing.assert_equal(vfile["r2"]["values"], np.arange(17))

    ht_after = Hashtable(vfile.f, "values").inverse()
    assert ht_after.keys() == {
        Slice(0, 10, 1),  # Same as before delete
        Slice(10, 17, 1),  # Was Slice(30, 37, 1)
    }
    np.testing.assert_equal(ht_after[Slice(0, 10, 1)], ht_before[Slice(0, 10, 1)])
    np.testing.assert_equal(ht_after[Slice(10, 17, 1)], ht_before[Slice(30, 37, 1)])


@hypothesis.settings(
    max_examples=20,
    # h5file is not reset between hypothesis examples
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
)
@given(delete_order=st.permutations(["r0", "r1", "r2", "r3", "r4", "r5"]))
def test_delete_versions_after_updates(vfile, delete_order):
    """Delete versions after various types of changes to each versions

    See Also
    --------
    https://github.com/deshaw/versioned-hdf5/issues/411
    test_delete_versions_after_shrinking
    test_staged_changes::test_shrinking_does_not_reuse_partial_chunks
    """
    with vfile.stage_version("r0") as sv:
        sv.create_dataset("values", data=np.arange(26), chunks=(10,))

    # Resize without updating. The resized chunk is a full copy of the original.
    with vfile.stage_version("r1") as sv:
        sv["values"].resize((17,))

    # Just update
    with vfile.stage_version("r2") as sv:
        sv["values"][:16] += 1

    # Resize after updating the chunk being resized. The resized chunk is brand new.
    with vfile.stage_version("r3") as sv:
        sv["values"][:14] += 1
        sv["values"].resize((15,))

    # Resize after updating an unrelated chunk. The resized chunk is brand new.
    with vfile.stage_version("r4") as sv:
        sv["values"][:5] += 1
        sv["values"].resize((14,))

    # Resize after completely wiping the previous contents.
    # Doesn't use StagedChangesArray.
    with vfile.stage_version("r5") as sv:
        sv["values"][:] += 1
        sv["values"].resize((12,))

    expect = {f"r{i}": vfile[f"r{i}"]["values"][:] for i in range(6)}

    for v in delete_order:
        delete_versions(vfile, v)
        del expect[v]
        for v2, expect_v2 in expect.items():
            np.testing.assert_equal(vfile[v2]["values"], expect_v2)


@pytest.mark.parametrize(
    ("obj", "metadata_opts"),
    [
        ("test_data2", {"compression": "gzip", "compression_opts": 3}),
        ("group/test_data4", {"compression": "gzip", "compression_opts": 3}),
    ],
)
def test_modify_metadata_compression_default_compression(vfile, obj, metadata_opts):
    """Test that setting compression via modify_metadata works for default
    compression.
    """
    setup_vfile(vfile)

    f = vfile.f

    # Check that the compression is unset for every dataset
    for dataset in ["test_data", "test_data2", "group/test_data4"]:
        for version in ["version1", "version2"]:
            assert vfile[version][dataset].compression is None
            assert vfile[version][dataset].compression_opts is None

        assert f["_version_data"][dataset]["raw_data"].compression is None
        assert f["_version_data"][dataset]["raw_data"].compression_opts is None

    modify_metadata(f, obj, **metadata_opts)
    check_data(vfile)

    # Check that the compression is set for the group that had its metadata modified
    for dataset in ["test_data", "test_data2", "group/test_data4"]:
        for version in ["version1", "version2"]:
            if dataset == obj:
                assert (
                    vfile[version][dataset].compression == metadata_opts["compression"]
                )
                assert (
                    vfile[version][dataset].compression_opts
                    == metadata_opts["compression_opts"]
                )
            else:
                assert vfile[version][dataset].compression is None
                assert vfile[version][dataset].compression_opts is None

        if dataset == obj:
            assert (
                f["_version_data"][dataset]["raw_data"].compression
                == metadata_opts["compression"]
            )
            assert (
                f["_version_data"][dataset]["raw_data"].compression_opts
                == metadata_opts["compression_opts"]
            )
        else:
            assert f["_version_data"][dataset]["raw_data"].compression is None
            assert f["_version_data"][dataset]["raw_data"].compression_opts is None

    # Make sure the tmp group group has been destroyed.
    assert set(f["_version_data"]) == {
        "test_data",
        "test_data2",
        "test_data3",
        "group",
        "versions",
    }
    assert set(f["_version_data"]["group"]) == {"test_data4"}


@pytest.mark.parametrize("obj", ["test_data2", "group/test_data4"])
@pytest.mark.parametrize("raw", [True, False])
def test_modify_metadata_blosc_compression(vfile, obj, raw):
    """Test that setting compression via modify_metadata works for third-party
    compression filters from hdf5plugin or pytables, such as Blosc.
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

    setup_vfile(vfile)

    f = vfile.f

    # Check that the compression is unset for every dataset
    for dataset in ["test_data", "test_data2", "group/test_data4"]:
        for version in ["version1", "version2"]:
            assert vfile[version][dataset].compression is None
            assert vfile[version][dataset].compression_opts is None

        raw_data = f["_version_data"][dataset]["raw_data"]
        assert raw_data.compression is None
        assert raw_data.compression_opts is None

    modify_metadata(f, obj, **kwargs)
    check_data(vfile)

    for dataset in ["test_data", "test_data2", "group/test_data4"]:
        for version in ["version1", "version2"]:
            if dataset == obj:
                assert vfile[version][dataset].compression == 32001
                # First four numbers are reserved for blosc compression;
                # others are actual compression options
                assert vfile[version][dataset].compression_opts[4:] == (7, 1, 2)
            else:
                assert vfile[version][dataset].compression is None
                assert vfile[version][dataset].compression_opts is None

        raw_data = f["_version_data"][dataset]["raw_data"]
        if dataset == obj:
            assert raw_data.compression is None
            assert raw_data.compression_opts is None
            assert raw_data._filters["32001"][4:] == (7, 1, 2)
        else:
            assert raw_data.compression is None
            assert raw_data.compression_opts is None

    # Make sure the tmp group group has been destroyed.
    assert set(f["_version_data"]) == {
        "test_data",
        "test_data2",
        "test_data3",
        "group",
        "versions",
    }
    assert set(f["_version_data"]["group"]) == {"test_data4"}


@pytest.mark.parametrize("raw", [True, False])
def test_modify_metadata_blosc_compression_opts(vfile, raw):
    """Test changing compression options of a custom compression filter,
    using hdf5plugin API.
    """
    hdf5plugin = pytest.importorskip("hdf5plugin")

    if raw:
        kwargs1 = {"compression": 32001, "compression_opts": (0, 0, 0, 0, 1, 1, 1)}
        kwargs2 = {"compression": 32001, "compression_opts": (0, 0, 0, 0, 9, 1, 1)}
    else:
        kwargs1 = {"compression": hdf5plugin.Blosc(clevel=1)}
        kwargs2 = {"compression": hdf5plugin.Blosc(clevel=9)}

    with vfile.stage_version("r0") as g:
        g.create_dataset("x", data=[1, 2, 3], **kwargs1)

    f = vfile.f
    raw_data = f["_version_data"]["x"]["raw_data"]
    assert raw_data._filters["32001"][4:] == (1, 1, 1)

    modify_metadata(f, "x", **kwargs2)
    raw_data = f["_version_data"]["x"]["raw_data"]
    assert raw_data._filters["32001"][4:] == (9, 1, 1)


def test_modify_metadata_decompress_gzip(vfile):
    """Use modify metadata to undo all compression."""
    with vfile.stage_version("r0") as g:
        g.create_dataset("x", data=[1, 2, 3], compression="gzip")

    f = vfile.f
    # No-op: omitting compression is not the same as setting compression=None
    modify_metadata(f, "x")
    raw_data = f["_version_data"]["x"]["raw_data"]
    assert raw_data.compression == "gzip"
    assert raw_data.compression_opts == 4  # Default

    modify_metadata(f, "x", compression=None)
    raw_data = f["_version_data"]["x"]["raw_data"]
    assert raw_data.compression is None
    assert raw_data.compression_opts is None


def test_modify_metadata_decompress_blosc(vfile):
    """Use modify metadata to undo all compression.
    This uses a third-party compression filter, which adds nuance because
    h5py.Dataset.compression returns None for third-party filters.
    """
    hdf5plugin = pytest.importorskip("hdf5plugin")
    with vfile.stage_version("r0") as g:
        g.create_dataset("x", data=[1, 2, 3], compression=hdf5plugin.Blosc())

    f = vfile.f
    raw_data = f["_version_data"]["x"]["raw_data"]
    # First four numbers are reserved for blosc compression;
    # others are the default compression options
    assert raw_data._filters["32001"][4:] == (5, 1, 1)

    # No-op: omitting compression is not the same as setting compression=None
    modify_metadata(f, "x")
    raw_data = f["_version_data"]["x"]["raw_data"]
    assert raw_data._filters["32001"][4:] == (5, 1, 1)

    modify_metadata(f, "x", compression=None)
    raw_data = f["_version_data"]["x"]["raw_data"]
    assert raw_data._filters == {}


@pytest.mark.parametrize(
    ("name", "default_value", "new_value"),
    [
        ("fletcher32", False, True),
        ("scaleoffset", None, 5),
        ("shuffle", False, True),
    ],
)
def test_modify_metadata_other_filters(vfile, name, default_value, new_value):
    """Test changing fletcher32, scaleoffset, shuffle parameters."""
    with vfile.stage_version("r0") as g:
        g.create_dataset("x", data=[1, 2, 3])

    f = vfile.f
    raw_data = f["_version_data"]["x"]["raw_data"]
    assert getattr(raw_data, name) == default_value

    modify_metadata(f, "x", **{name: new_value})
    raw_data = f["_version_data"]["x"]["raw_data"]
    assert getattr(raw_data, name) == new_value

    # Take care not to confuse default and False
    modify_metadata(f, "x")  # No-op: no value != False
    raw_data = f["_version_data"]["x"]["raw_data"]
    assert getattr(raw_data, name) == new_value

    modify_metadata(f, "x", **{name: default_value})
    raw_data = f["_version_data"]["x"]["raw_data"]
    assert getattr(raw_data, name) == default_value
