import argparse
import importlib
import os.path

import psutil


def main():
    """Print out which .so file is actually loaded at runtime for libhdf5"""
    parser = argparse.ArgumentParser(
        description="Print out which .so file is actually loaded at runtime for libhdf5"
    )
    parser.add_argument("module", nargs="?", default="h5py", help="Module to import")
    args = parser.parse_args()

    print(f"import {args.module}")
    importlib.import_module(args.module)

    proc = psutil.Process()
    for m in proc.memory_maps():
        if "hdf5" in os.path.basename(m.path):
            print(m.path)


if __name__ == "__main__":
    main()
