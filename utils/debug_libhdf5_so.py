import argparse
import importlib
import os
import subprocess


def main():
    """Print out which .so file is actually loaded at runtime for libhdf5"""
    parser = argparse.ArgumentParser(
        description="Print out which .so file is actually loaded at runtime for libhdf5"
    )
    parser.add_argument("module", nargs="?", default="h5py", help="Module to import")
    args = parser.parse_args()

    print(f"import {args.module}")
    importlib.import_module(args.module)

    stdout = subprocess.check_output(["lsof", "-p", str(os.getpid())])
    for row in stdout.decode("utf-8").splitlines():
        if "libhdf5" in row:
            print(row.strip())


if __name__ == "__main__":
    main()
