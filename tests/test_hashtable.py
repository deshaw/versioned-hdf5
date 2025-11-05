from unittest import mock

import h5py
import numpy as np
import pytest

from versioned_hdf5 import VersionedHDF5File
from versioned_hdf5.backend import create_base_dataset
from versioned_hdf5.hashtable import Hashtable

from .conftest import assert_slab_offsets


def test_hashtable(h5file):
    create_base_dataset(h5file, "test_data", data=np.empty((0,)))
    with Hashtable(h5file, "test_data") as h:
        assert len(h) == 0
        h[b"\xff" * 32] = slice(0, 1)
        assert len(h) == 1
        assert h[b"\xff" * 32] == slice(0, 1)
        assert h.largest_index == 1
        assert bytes(h.hash_table[0][0]) == b"\xff" * 32
        assert tuple(h.hash_table[0][1]) == (0, 1)
        assert h == {b"\xff" * 32: slice(0, 1)}

        with pytest.raises(TypeError):
            h["\x01" * 32] = slice(0, 1)
        with pytest.raises(ValueError):
            h[b"\x01"] = slice(0, 1)
        with pytest.raises(TypeError):
            h[b"\x01" * 32] = (0, 1)
        with pytest.raises(ValueError):
            h[b"\x01" * 32] = slice(0, 4, 2)


def test_from_raw_data(setup_vfile):
    with setup_vfile() as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("0") as sv:
            sv.create_dataset("test_data", data=np.arange(100), chunks=(10,))

        h = Hashtable(f, "test_data")
        h_dataset = h.hash_table_dataset
        h2 = Hashtable.from_raw_data(f, "test_data", hash_table_name="test_hash_table")
        h2_dataset = h2.hash_table_dataset
        assert h2_dataset.name == "/_version_data/test_data/test_hash_table"
        np.testing.assert_equal(h_dataset[:], h2_dataset[:])


def test_hashtable_multidimension(h5file):
    # Ensure that the same data with different shape hashes differently
    create_base_dataset(h5file, "test_data", data=np.empty((0,)))
    h = Hashtable(h5file, "test_data")
    assert h.hash(
        np.ones(
            (
                1,
                2,
                3,
            )
        )
    ) != h.hash(np.ones((3, 2, 1)))


def test_issue_208(setup_vfile):
    file = setup_vfile()
    filename = file.filename
    with file as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("0") as sv:
            sv.create_dataset("bar", data=np.arange(10))

    with h5py.File(filename, "r+") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version("1") as sv:
            sv["bar"].resize((12,))
            sv["bar"][8:12] = sv["bar"][6:10]
            sv["bar"][6:8] = [0, 0]


def test_object_dtype_hashes_values(tmp_path):
    """Test that object dtype arrays hash values, not element ids.

    See https://github.com/deshaw/versioned-hdf5/issues/256 for more
    information.
    """
    filename = tmp_path / "test.h5"
    N = 100
    with h5py.File(filename, mode="w") as f:
        file = VersionedHDF5File(f)
        s = ""
        for i in range(N):
            s += "a"
            arr = np.array([s], dtype=object)

            with file.stage_version(f"r{i}") as group:
                group.create_dataset(
                    "values", shape=(1,), dtype=h5py.string_dtype(length=None), data=arr
                )

    with h5py.File(filename, mode="r") as f:
        file = VersionedHDF5File(f)
        for i in range(N):
            assert file[f"r{i}"]["values"][()] == b"a" * (i + 1)


def test_object_dtype_hashes_concatenated_values(tmp_path):
    """Test that object dtype arrays hash values which concatenate
    to the same string to different hashes.
    See https://github.com/deshaw/versioned-hdf5/issues/288.
    """
    filename = tmp_path / "test.h5"
    with h5py.File(filename, mode="w") as f:
        file = VersionedHDF5File(f)
        with file.stage_version("r0") as group:
            group.create_dataset(
                "values",
                dtype=h5py.string_dtype(encoding="ascii"),
                data=np.array([b"a", b"b", b"cd"], dtype=object),
                maxshape=(None,),
                chunks=(100,),
            )
        with file.stage_version("r1") as group:
            group["values"] = np.array([b"ab", b"", b"cd"], dtype=object)
        with file.stage_version("r2") as group:
            group["values"] = np.array([b"ab", b"c", b"d"], dtype=object)

    with h5py.File(filename, mode="r") as f:
        file = VersionedHDF5File(f)
        np.testing.assert_equal(
            file["r0"]["values"][:], np.array([b"a", b"b", b"cd"], dtype=object)
        )
        np.testing.assert_equal(
            file["r1"]["values"][:], np.array([b"ab", b"", b"cd"], dtype=object)
        )
        np.testing.assert_equal(
            file["r2"]["values"][:], np.array([b"ab", b"c", b"d"], dtype=object)
        )


def test_verify_chunk_reuse_data_version_2(tmp_path, monkeypatch):
    """Test whether the issue with DATA_VERSION==2 would have been caught by
    _verify_new_chunk_reuse.

    The problem with the hash function for DATA_VERSION==2 was that it hashed
    the encoded data, not the data itself, meaning that "b'hello'" would hash
    to the same value as b'hello'.
    """

    monkeypatch.setenv("ENABLE_CHUNK_REUSE_VALIDATION", "1")

    def data_version_2_hash(self, data: np.ndarray):
        """
        Compute hash for `data` array.

        (Copied from commit 1f968f4 Hashtable.hash. This version hashes the encoded
        data, not the data itself.)
        """
        if data.dtype == "object":
            hash_value = self.hash_function()
            for value in data.flat:
                hash_value.update(bytes(str(value), "utf-8"))
            hash_value.update(bytes(str(data.shape), "utf-8"))
            return hash_value.digest()
        else:
            return self.hash_function(
                data.data.tobytes() + bytes(str(data.shape), "ascii")
            ).digest()

    with mock.patch.object(Hashtable, "hash", autospec=True) as mocked_hash:
        mocked_hash.side_effect = data_version_2_hash

        data1 = np.array(["b'hello'", "b'world'"], dtype="O")
        data2 = np.array([b"hello", b"world"], dtype="O")

        filename = tmp_path / "data.h5"
        with h5py.File(filename, mode="w") as f:
            vf = VersionedHDF5File(f)
            with vf.stage_version("r0") as group:
                group.create_dataset(
                    "values",
                    dtype=h5py.string_dtype(encoding="ascii"),
                    data=data1,
                    maxshape=(None,),
                    chunks=(2,),
                )

            with pytest.raises(ValueError), vf.stage_version("r1") as group:
                group["values"] = np.concatenate((data2, data2))


def test_verify_chunk_reuse_data_version_3(tmp_path, monkeypatch):
    """Test whether the issue with DATA_VERSION==3 would have been caught by
    _verify_new_chunk_reuse.

    The problem with the hash function for DATA_VERSION==3 was that it didn't
    include the length of string array elements in the hash, meaning that
    the hash of [b"a", b"b", b"cd"] would be the same as [b"ab", b"", b"cd"]
    (because hashing each element in the array is equivalent to hashing
    the concatenated elements of the array).
    """

    monkeypatch.setenv("ENABLE_CHUNK_REUSE_VALIDATION", "1")

    def data_version_3_hash(self, data: np.ndarray):
        """
        Compute hash for `data` array.

        (Copied from commit d382673 Hashtable.hash. This version didn't include the
        string length in the hash.)
        """
        if data.dtype == "object":
            hash_value = self.hash_function()
            for value in data.flat:
                if isinstance(value, str):
                    value = value.encode("utf-8")
                hash_value.update(value)
            hash_value.update(bytes(str(data.shape), "utf-8"))
            return hash_value.digest()
        else:
            return self.hash_function(
                data.data.tobytes() + bytes(str(data.shape), "ascii")
            ).digest()

    with mock.patch.object(Hashtable, "hash", autospec=True) as mocked_hash:
        mocked_hash.side_effect = data_version_3_hash

        filename = tmp_path / "data.h5"
        with h5py.File(filename, mode="w") as f:
            vf = VersionedHDF5File(f)
            with vf.stage_version("r0") as group:
                group.create_dataset(
                    "values",
                    dtype=h5py.string_dtype(encoding="ascii"),
                    data=np.array([b"a", b"b", b"cd"], dtype=object),
                    maxshape=(None,),
                    chunks=(100,),
                )

            with pytest.raises(ValueError), vf.stage_version("r1") as group:
                group["values"] = np.array([b"ab", b"", b"cd"], dtype=object)
            with pytest.raises(ValueError), vf.stage_version("r2") as group:
                group["values"] = np.array([b"ab", b"c", b"d"], dtype=object)


def test_chunk_reuse_nan(vfile):
    """Check that chunks are correctly verified when reused, even with nans."""
    data = np.array([1, 2, 3, 4, 5, np.nan])
    with vfile.stage_version("r0") as sv:
        sv.create_dataset("values", data=data, chunks=(6,))
    with vfile.stage_version("r1") as sv:
        sv["values"] = np.concatenate((data, data))
    with vfile.stage_version("r2") as sv:
        assert_slab_offsets(sv, "values", [0, 0])


def test_chunk_reuse_strings(vfile):
    """Check that that strings can be reused and verified."""
    data = np.array([b"a", b"b", b"cd"], dtype=object)
    with vfile.stage_version("r0") as sv:
        sv.create_dataset(
            "values",
            data=data,
            dtype=h5py.string_dtype(),
            maxshape=(None,),
            chunks=(3,),
        )
    with vfile.stage_version("r1") as sv:
        sv["values"] = np.concatenate((data, data))
    with vfile.stage_version("r2") as sv:
        assert_slab_offsets(sv, "values", [0, 0])


def test_chunk_reuse_bytes_strings_mix(vfile):
    """Check that that, when writing to a h5py.string_dtype dataset, strings
    and bytes can be used interchangeably."""

    data = np.array([b"a", "b", "a", b"b", b"c", "d"], dtype=object)
    with vfile.stage_version("r0") as sv:
        sv.create_dataset(
            "values",
            data=data,
            dtype=h5py.string_dtype(),
            maxshape=(None,),
            chunks=(2,),
        )
    with vfile.stage_version("r1") as sv:
        assert_slab_offsets(sv, "values", [0, 0, 2])
        sv["values"] = data[[4, 5, 2, 3, 0, 1]]
    with vfile.stage_version("r2") as sv:
        assert_slab_offsets(sv, "values", [2, 0, 0])


def test_chunk_reuse_multidim_1(vfile):
    """Check that we correctly handle chunk reuse verification for multi-dimensional
    Datasets.
    """
    with vfile.stage_version("r0") as sv:
        sv.create_dataset(
            "values",
            data=np.array([[i + (j % 3) for i in range(8)] for j in range(7)]),
            maxshape=(None, None),
            chunks=(3, 3),
        )
    with vfile.stage_version("r1") as sv:
        values_ds = sv["values"]
        values_ds.resize((8, 8))
        values_ds[:] = np.array([[i + (j % 3) for i in range(8)] for j in range(8)])
    with vfile.stage_version("r2") as sv:
        assert_slab_offsets(sv, "values", [[0, 3, 6], [0, 3, 6], [18, 21, 24]])


def test_verify_chunk_disabled_by_default(tmp_path, monkeypatch):
    """Check that we skip chunk reuse verification if the environment variable is not
    set.
    """
    monkeypatch.delenv("ENABLE_CHUNK_REUSE_VALIDATION", raising=False)

    # This is the same test as test_verify_chunk_reuse_data_version_2,
    # but with verification turned off with the environment variable.
    def data_version_2_hash(self, data: np.ndarray):
        """
        Compute hash for `data` array.

        (Copied from commit 1f968f4 Hashtable.hash. This version hashes the encoded
        data, not the data itself.)
        """
        if data.dtype == "object":
            hash_value = self.hash_function()
            for value in data.flat:
                hash_value.update(bytes(str(value), "utf-8"))
            hash_value.update(bytes(str(data.shape), "utf-8"))
            return hash_value.digest()
        else:
            return self.hash_function(
                data.data.tobytes() + bytes(str(data.shape), "ascii")
            ).digest()

    with mock.patch.object(Hashtable, "hash", autospec=True) as mocked_hash:
        mocked_hash.side_effect = data_version_2_hash

        data1 = np.array(["b'hello'", "b'world'"], dtype="O")
        data2 = np.array([b"hello", b"world"], dtype="O")

        filename = tmp_path / "data.h5"
        with h5py.File(filename, mode="w") as f:
            vf = VersionedHDF5File(f)
            with vf.stage_version("r0") as group:
                group.create_dataset(
                    "values",
                    dtype=h5py.string_dtype(encoding="ascii"),
                    data=data1,
                    maxshape=(None,),
                    chunks=(2,),
                )

            # This should raise an error, but will not because chunk
            # reuse verification is turned off.
            with vf.stage_version("r1") as group:
                group["values"] = np.concatenate((data2, data2))
