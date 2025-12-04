"""Import h5py and versioned-hdf5.
Print out HDF5, h5py, and versioned-hdf5 versions.
Finally, create a versioned-hdf5 file on disk and write some data to it.
"""

import tempfile

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


def main():
    print("libhdf5       ", ".".join(map(str, h5py.h5.get_libversion())))
    print("h5py          ", h5py.__version__)
    if exc is None:
        print("versioned-hdf5", versioned_hdf5.__version__)
    else:
        raise exc

    with tempfile.TemporaryFile() as fh, h5py.File(fh, "w") as f:
        vf = versioned_hdf5.VersionedHDF5File(f)
        with vf.stage_version("v1") as v:
            v.create_dataset("data", data=[1, 2, 3, 4], chunks=(2,))
        with vf.stage_version("v2") as v:
            v["data"][0] = 5
        assert vf["v2"]["data"][:].tolist() == [5, 2, 3, 4]
    print("Smoke test successful!")


if __name__ == "__main__":
    main()
