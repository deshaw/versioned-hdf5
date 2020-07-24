import h5py

from ..backend import initialize

# TODO: Use a fixture for this
def setup(file_name='test.hdf5', name=None, version_name=None):
    # TODO: Use a temporary directory
    f = h5py.File(file_name, 'w')
    initialize(f)
    if name:
        f['_version_data'].create_group(name)
    if version_name:
        if isinstance(version_name, str):
            version_name = [version_name]
        for name in version_name:
            f['_version_data/versions'].create_group(name)
    return f
