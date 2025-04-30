import argparse
import importlib
import os
import subprocess
import sys


def get_open_dlls() -> list[str]:
    """Yield location of loaded dynamic libraries (plus spurious open files)"""
    if sys.platform in ("win32", "cygwin"):
        # Note: this code path also works on Linux, but not on MacOS
        import psutil

        proc = psutil.Process()
        return [m.path for m in proc.memory_maps()]
    else:
        stdout = subprocess.check_output(
            f"lsof -p {os.getpid()} -Fn | grep '^n' | cut -c2-", shell=True
        )
        return [line for line in stdout.decode("utf-8").splitlines()]


def main():
    """Print out which .so file is actually loaded at runtime for libhdf5"""
    parser = argparse.ArgumentParser(
        description="Print out which .so file is actually loaded at runtime for libhdf5"
    )
    parser.add_argument("module", nargs="?", default="h5py", help="Module to import")
    args = parser.parse_args()

    print(f"import {args.module}")
    importlib.import_module(args.module)

    for fname in get_open_dlls():
        if "hdf5" in os.path.basename(fname):
            print(fname)


if __name__ == "__main__":
    main()
