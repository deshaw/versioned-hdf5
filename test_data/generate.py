"""Generate bad datasets with old versions of versioned-hdf5.

Usage:
    1. install newest versioned-hdf5 without DATA_VERSION
    2. python generate.py
    3. install newest versioned-hdf5 with DATA_VERSION=2
    4. python generate.py
    5. install newest versioned-hdf5 with DATA_VERSION=3
    6. python generate.py
"""

import os
import uuid

import h5py
import numpy as np

from versioned_hdf5 import VersionedHDF5File

try:
    from versioned_hdf5.backend import DATA_VERSION  # noqa: F401
except ImportError:
    DATA_VERSION = 1


def generate_bad_data_version_1():
    """Generate versioned-hdf5 files in cwd with bad object dtype hash tables in them.

    See https://github.com/deshaw/versioned-hdf5/issues/256 for more information.
    """
    filename = "object_dtype_bad_hashtable_data.h5"
    with h5py.File(filename, mode="w") as f:
        vf = VersionedHDF5File(f)
        arr = np.array(
            [
                "abcd",
                "def",
                "ghi",
                "jkl",
            ],
            dtype=object,
        )

        with vf.stage_version("r0") as group:
            group.create_dataset(
                "data_with_bad_hashtable",
                dtype=h5py.string_dtype(length=None),
                data=arr,
            )

    filename = "object_dtype_bad_hashtable_data2.h5"
    with h5py.File(filename, mode="w") as f:
        vf = VersionedHDF5File(f)

        arr = np.array(
            [
                "abcd",
                "def",
                "ghi",
                "jkl",
            ],
            dtype=object,
        )
        arr2 = np.array(
            [
                "abcd",
                "pqrs",
                "df",
                "tuvw",
                "xyz",
            ],
            dtype=object,
        )
        arr3 = np.linspace(1, 100, 1000)

        with vf.stage_version("r0") as group:
            group.create_dataset(
                "data_with_bad_hashtable",
                dtype=h5py.string_dtype(length=None),
                data=arr,
            )
            group.create_dataset(
                "data_with_bad_hashtable2",
                dtype=h5py.string_dtype(length=None),
                data=arr2,
            )
            group.create_dataset("linspace", data=arr3)

        with vf.stage_version("r1") as group:
            group["data_with_bad_hashtable"][1] = "foo"
            group["linspace"][3] = 8

        with vf.stage_version("r2") as group:
            group["data_with_bad_hashtable"][2] = "bar"
            group["data_with_bad_hashtable2"][2] = "what"

    filename = "nested_data_old_data_version.h5"
    with h5py.File(filename, mode="w") as f:
        vf = VersionedHDF5File(f)

        with vf.stage_version("r0") as sv:
            data_group = sv.create_group("data")
            data_group.create_dataset(
                "values",
                data=np.array(["1", "2", "3"]),
                dtype=h5py.string_dtype(length=None),
            )

    filename = "multiple_nested_data_old_data_version.h5"
    with h5py.File(filename, mode="w") as f:
        vf = VersionedHDF5File(f)

        arr = np.array(
            [
                "abcd",
                "def",
                "ghi",
                "jkl",
            ],
            dtype=object,
        )

        with vf.stage_version("r0") as sv:
            g = sv.create_group("foo/bar/baz")
            g.create_dataset(
                "foo/bar/baz/values", dtype=h5py.string_dtype(length=None), data=arr
            )

    filename = "object_dtype_bad_hashtable_chunk_reuse.h5"
    with h5py.File(filename, mode="w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version(str(0)) as sv:
            sv.create_dataset(
                "values",
                data=np.arange(0).astype(str).astype("O"),
                dtype=h5py.string_dtype(length=None),
                chunks=(10,),
            )

    for i in range(1, 11):
        with h5py.File(filename, "r+") as f:
            vf = VersionedHDF5File(f)
            with vf.stage_version(str(i)) as sv:
                sv["values"] = np.arange(i).astype(str).astype("O")

    filename = "object_dtype_bad_hashtable_chunk_reuse_unicode.h5"
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
    with h5py.File(filename, mode="w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version(str(0)) as sv:
            sv.create_dataset(
                "values",
                data=np.array(numbers[:1], dtype="O"),
                dtype=h5py.string_dtype(length=None),
                chunks=(10,),
            )

    for i in range(1, 11):
        with h5py.File(filename, "r+") as f:
            vf = VersionedHDF5File(f)
            with vf.stage_version(str(i)) as sv:
                sv["values"] = np.array(numbers[: i + 1], dtype="O")

    filename = "object_dtype_bad_hashtable_chunk_reuse_multi_dim.h5"
    with h5py.File(filename, mode="w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version(str(uuid.uuid4())) as sv:
            sv.create_dataset(
                "values",
                data=np.array(
                    [
                        [chr(ord("a") + ((j + k) % 10)) * 3 for j in range(4)]
                        for k in range(4)
                    ],
                    dtype="O",
                ),
                dtype=h5py.string_dtype(length=None),
                chunks=(2, 2),
            )

    for i in range(1, 11):
        with h5py.File(filename, "r+") as f:
            vf = VersionedHDF5File(f)
            with vf.stage_version(str(uuid.uuid4())) as sv:
                sv["values"] = np.array(
                    [
                        [chr(ord("a") + ((i + j + k) % 10)) * 3 for j in range(4)]
                        for k in range(4)
                    ],
                    dtype="O",
                )


def generate_bad_data_version_2():
    """Generate versioned-hdf5 files in cwd with bad object dtype hash tables in them.

    See https://github.com/deshaw/versioned-hdf5/issues/256 for more information.
    """
    filename = "bad_hashtable_data_version_2.h5"
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
        with vf.stage_version("r1") as group:
            group["values"] = np.array([b"ab", b"", b"cd"], dtype=object)
        # with vf.stage_version("r2") as group:
        #     group["values"] = np.array([b"ab", b"c", b"d"], dtype=object)  # noqa: ERA001,E501


def generate_bad_data_version_3():
    """Generate versioned-hdf5 files in cwd with bad object dtype hash tables in them.

    See https://github.com/deshaw/versioned-hdf5/issues/256 for more information.
    """
    filename = "bad_hashtable_data_version_3.h5"
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
        with vf.stage_version("r1") as group:
            group["values"] = np.array([b"ab", b"", b"cd"], dtype=object)
        # with vf.stage_version("r2") as group:
        #     group["values"] = np.array([b"ab", b"c", b"d"], dtype=object)  # noqa: ERA001,E501


def run_on_version(func, target):
    msg = f"{func.__name__}: DATA_VERSION={DATA_VERSION}, target={target}..."
    if DATA_VERSION != target:
        print(msg, "SKIP")
    else:
        print(msg, "RUN")
        func()


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    run_on_version(generate_bad_data_version_1, 1)
    run_on_version(generate_bad_data_version_2, 2)
    run_on_version(generate_bad_data_version_3, 3)
