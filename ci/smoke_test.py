"""Import h5py and versioned-hdf5.
Print out HDF5, h5py, and versioned-hdf5 versions.
"""

import h5py
import h5py.h5

try:
    # Print h5py and hdf5 versions even if versioned-hdf5 is broken
    # This import also triggers recompilation of any modified cython files
    # in editable installs, which is pretty verbose - so it's best to do it
    # before the actual smoke test prints anything.
    import versioned_hdf5  # noqa: E402

    exc = None
except Exception as e:
    exc = e

if __name__ == "__main__":
    print("HDF5 version:", ".".join(map(str, h5py.h5.get_libversion())))
    print("h5py version:", h5py.__version__, "(may not be accurate for git tip)")
    if exc is None:
        print("versioned_hdf5 version:", versioned_hdf5.__version__)
    else:
        raise exc
