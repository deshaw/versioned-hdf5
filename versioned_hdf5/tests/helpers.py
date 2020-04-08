import h5py

from ..backend import initialize

# TODO: Use a fixture for this
def setup(name=None, version_name=None):
    # TODO: Use a temporary directory
    f = h5py.File('test.hdf5', 'w')
    initialize(f)
    if name:
        f['_version_data'].create_group(name)
    if version_name:
        f['_version_data/versions'].create_group(version_name)
    return f
