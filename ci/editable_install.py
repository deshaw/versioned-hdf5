"""Test if versioned-hdf5 is already installed in the current environment
(either regular installation or editable).
If not, install it in editable mode.

The build dir, where the .so files live, is set to $CONDA_PREFIX/build
instead of the default $PROJECT_ROOT/build/cp314 (or other python version).
This is important for two reasons:
- when the user wipes the environment, they don't leave an orphaned build dir,
  and vice versa;
- when multiple environments use the same Python version, they will not share binaries.
  This is important when these environments use different versions of libhdf5.
"""

import argparse
import importlib.metadata
import os
import pathlib
import subprocess
import sys


def install():
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
        f"-Cbuild-dir={build_dir}",  # Important; read top of file
        # De-obfuscate compilation errors
        # https://github.com/mesonbuild/meson-python/issues/820
        "-Ceditable-verbose=true",
    ]
    print(*args)
    subprocess.check_call(args)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", "-f", action="store_true", help="Force re-installation"
    )
    args = parser.parse_args(argv)

    try:
        version = importlib.metadata.version("versioned_hdf5")
        print(f"versioned_hdf5 {version} is already installed")
        do_install = args.force
    except importlib.metadata.PackageNotFoundError:
        do_install = True  # Shorten traceback in case of installation failure
    if do_install:
        install()


if __name__ == "__main__":
    main()
