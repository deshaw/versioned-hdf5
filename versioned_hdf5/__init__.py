from importlib.metadata import version

from versioned_hdf5.api import VersionedHDF5File
from versioned_hdf5.replay import delete_version, delete_versions, modify_metadata

__version__ = version(__package__)

__all__ = ["VersionedHDF5File", "delete_version", "delete_versions", "modify_metadata"]
