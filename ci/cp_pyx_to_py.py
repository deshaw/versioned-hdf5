"""Create a symlink of Cython files with pure-python syntax (.py)[1] to .pyx.
[1] https://cython.readthedocs.io/en/latest/src/tutorial/pure.html

This is a hack to convince Meson to compile .py files with Cython.

NOTE: Don't use the 'ln -s' shell command, as it fails on conda-forge Windows CI.
FIXME: This causes editable installs on Windows to fail to automatically recompile
       after a change in one of the target modules, as the 'symlink' is just a copy.
"""

import pathlib

MODNAMES = ["cytools", "subchunk_map", "staged_changes"]

if __name__ == "__main__":
    proj_dir = pathlib.Path(__file__).parent.parent / "versioned_hdf5"
    for modname in MODNAMES:
        src = proj_dir / f"{modname}.pyx"
        dst = f"{modname}.py"
        print(f"Symlink {src} -> {dst}")
        src.unlink(missing_ok=True)
        src.symlink_to(dst)
