"""Test if versioned-hdf5 is already installed in the current environment
(either regular installation or editable).
If not, install it in editable mode.
"""

import importlib.metadata


def install():
    import os
    import pathlib
    import subprocess
    import sys

    project_root = pathlib.Path(__file__).parent.parent
    build_dir = pathlib.Path(os.environ["CONDA_PREFIX"]) / "build"
    args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "--no-dependencies",
        "--editable",
        str(project_root),
        f"-Cbuild-dir={build_dir}",
        # De-obfuscate compilation errors
        # https://github.com/mesonbuild/meson-python/issues/820
        "-Ceditable-verbose=true",
    ]
    print("pip", *args)
    subprocess.check_call(args)


if __name__ == "__main__":
    try:
        version = importlib.metadata.version("versioned_hdf5")
        print(f"versioned_hdf5 {version} is already installed")
        do_install = False
    except importlib.metadata.PackageNotFoundError:
        do_install = True  # Shorten traceback in case of installation failure
    if do_install:
        install()
