from importlib.metadata import version

from .api import VersionedHDF5File
from .replay import delete_version, delete_versions, modify_metadata

__version__ = version(__package__)

__all__ = ["VersionedHDF5File", "delete_version", "delete_versions", "modify_metadata"]
