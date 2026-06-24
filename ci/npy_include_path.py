"""Print the path to the NumPy C headers, relative to the pwd."""

import contextlib
import os

import numpy

npy_include_path = numpy.get_include()
with contextlib.suppress(ValueError):
    # relpath is needed for editable installs.
    # On Windows, relpath can fail if the path is on a different drive;
    # in that case, just use the absolute path.
    npy_include_path = os.path.relpath(npy_include_path)

print(npy_include_path)
