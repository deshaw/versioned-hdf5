from .api import VersionedHDF5File

__all__ = ['VersionedHDF5File']

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
