import os
from pytest import fixture
from .helpers import setup
from ..api import VersionedHDF5File


@fixture
def h5file(tmp_path, request):
    file_name = os.path.join(tmp_path, 'file.hdf5')
    name = None
    version_name = None
    m = request.node.get_closest_marker('setup_args')
    if m is not None:
        if 'file_name' in m.kwargs.keys():
            file_name = m.kwargs['file_name']
        if 'name' in m.kwargs.keys():
            name = m.kwargs['name']
        if 'version_name' in m.kwargs.keys():
            version_name = m.kwargs['version_name']

    f = setup(file_name=file_name, name=name, version_name=version_name)
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
                         "Unspecified error in H5Iget_type (return value <0)"]:
            return
        raise


@fixture
def vfile(tmp_path, h5file):
    file = VersionedHDF5File(h5file)
    yield file
    file.close()
