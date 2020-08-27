import os
from pytest import yield_fixture
from .helpers import setup
from ..api import VersionedHDF5File


@yield_fixture
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
    f.close()


@yield_fixture
def vfile(tmp_path, h5file):
    file = VersionedHDF5File(h5file)
    yield file
    file.close()
