import os
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
