import os
import uuid

from pytest import fixture
from .helpers import setup_vfile
from ..api import VersionedHDF5File

# Run tests marked with @pytest.mark.slow last. See
# https://stackoverflow.com/questions/61533694/run-slow-pytest-commands-at-the-end-of-the-test-suite
def by_slow_marker(item):
    return bool(item.get_closest_marker('slow'))

def pytest_collection_modifyitems(items):
    items.sort(key=by_slow_marker)

@fixture
def filepath(tmp_path, request):
    file_name = os.path.join(tmp_path, 'file.hdf5')
    m = request.node.get_closest_marker('setup_args')
    if m is not None:
        if 'file_name' in m.kwargs.keys():
            file_name = m.kwargs['file_name']
    yield file_name

@fixture
def h5file(tmp_path, request):
    file_name = os.path.join(tmp_path, 'file.hdf5')
    version_name = None
    m = request.node.get_closest_marker('setup_args')
    if m is not None:
        if 'file_name' in m.kwargs.keys():
            file_name = m.kwargs['file_name']
        if 'name' in m.kwargs.keys():
            raise ValueError("The name argument is no longer used")
        if 'version_name' in m.kwargs.keys():
            version_name = m.kwargs['version_name']

    f = setup_vfile(file_name=file_name, version_name=version_name)
    yield f
    try:
        f.close()
    # Workaround upstream h5py bug. https://github.com/deshaw/versioned-hdf5/issues/162
    except ValueError as e:
        if e.args[0] == "Unrecognized type code -1":
            return
        raise
    except RuntimeError as e:
        if e.args[0] in ["Can't increment id ref count (can't locate ID)",
                         "Unspecified error in H5Iget_type (return value <0)",
                         "Can't retrieve file id (invalid data ID)"]:
            return
        raise


@fixture
def vfile(tmp_path, h5file):
    file = VersionedHDF5File(h5file)
    yield file
    file.close()


def generate_bad_data():
    """Generate versioned-hdf5 files with bad object dtype hash tables in them.

    See https://github.com/deshaw/versioned-hdf5/issues/256 for more information.

    Raises:
        ImportError: Raised if the user tries to generate bad data with newer versions
        of the library; you need an old version to replicate the hash table issue.
    """
    import numpy as np
    import h5py
    from versioned_hdf5 import VersionedHDF5File, __version__

    try:
        from versioned_hdf5.backend import DATA_VERSION  # noqa: F401
    except ImportError:
        DATA_VERSION = None

    if DATA_VERSION is not None:
        raise ImportError(
            f"versioned_hdf5=={__version__} installed; "
            "this file only generates bad data on versioned_hdf5 <= 1.3.14."
        )

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
            dtype=object
        )

        with vf.stage_version("r0") as group:
            group.create_dataset(
                "data_with_bad_hashtable",
                dtype=h5py.string_dtype(length=None),
                data=arr
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
            dtype=object
        )
        arr2 = np.array(
            [
                "abcd",
                "pqrs",
                "df",
                "tuvw",
                "xyz",
            ],
            dtype=object
        )
        arr3 = np.linspace(1, 100, 1000)

        with vf.stage_version("r0") as group:
            group.create_dataset(
                "data_with_bad_hashtable",
                dtype=h5py.string_dtype(length=None),
                data=arr
            )
            group.create_dataset(
                "data_with_bad_hashtable2",
                dtype=h5py.string_dtype(length=None),
                data=arr2
            )
            group.create_dataset("linspace", data=arr3)

        with vf.stage_version("r1") as group:
            group['data_with_bad_hashtable'][1] = "foo"
            group['linspace'][3] = 8

        with vf.stage_version("r2") as group:
            group['data_with_bad_hashtable'][2] = "bar"
            group['data_with_bad_hashtable2'][2] = "what"

    filename = "nested_data_old_data_version.h5"
    with h5py.File(filename, mode="w") as f:
        vf = VersionedHDF5File(f)

        with vf.stage_version('r0') as sv:
            data_group = sv.create_group('data')
            data_group.create_dataset('values',
                                      data=np.array(['1', '2', '3']),
                                      dtype=h5py.string_dtype(length=None))

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
            dtype=object
        )

        with vf.stage_version('r0') as sv:
            g = sv.create_group('foo/bar/baz')
            g.create_dataset(
                "foo/bar/baz/values",
                dtype=h5py.string_dtype(length=None),
                data=arr
            )

    filename = "object_dtype_bad_hashtable_chunk_reuse.h5"
    with h5py.File(filename, mode="w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version(str(0)) as sv:
            sv.create_dataset('values',
                              data=np.arange(0).astype(str).astype('O'),
                              dtype=h5py.string_dtype(length=None),
                              chunks=(10,))

    for i in range(1, 11):
        with h5py.File(filename, 'r+') as f:
            vf = VersionedHDF5File(f)
            with vf.stage_version(str(i)) as sv:
                sv['values'] = np.arange(i).astype(str).astype('O')

    filename = "object_dtype_bad_hashtable_chunk_reuse_unicode.h5"
    numbers = ['れい', 'いち', 'に', 'さん', 'し', 'ご', 'ろく', 'しち', 'はち', 'きゅう', 'じゅう']
    with h5py.File(filename, mode="w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version(str(0)) as sv:
            sv.create_dataset('values',
                              data=np.array(numbers[:1], dtype='O'),
                              dtype=h5py.string_dtype(length=None),
                              chunks=(10,))

    for i in range(1, 11):
        with h5py.File(filename, 'r+') as f:
            vf = VersionedHDF5File(f)
            with vf.stage_version(str(i)) as sv:
                sv['values'] = np.array(numbers[:i + 1], dtype='O')

    filename = "object_dtype_bad_hashtable_chunk_reuse_multi_dim.h5"
    with h5py.File(filename, mode="w") as f:
        vf = VersionedHDF5File(f)
        with vf.stage_version(str(uuid.uuid4())) as sv:
            sv.create_dataset('values',
                              data=np.array([[chr(ord('a') + ((j + k) % 10)) * 3 for j in range(4)]
                                             for k in range(4)], dtype='O'),
                              dtype=h5py.string_dtype(length=None),
                              chunks=(2, 2))

    for i in range(1, 11):
        with h5py.File(filename, 'r+') as f:
            vf = VersionedHDF5File(f)
            with vf.stage_version(str(uuid.uuid4())) as sv:
                sv['values'] = np.array([[chr(ord('a') + ((i + j + k) % 10)) * 3 for j in range(4)]
                                         for k in range(4)], dtype='O')
