"""Make a copy of Cython files with pure-python syntax (.py)[1] to .pyx.
[1] https://cython.readthedocs.io/en/latest/src/tutorial/pure.html

This is a hack to convince Meson to compile .py files with Cython.

.. note::
    Don't use the 'cp' shell command, as it fails on conda-forge Windows CI.
"""

import pathlib
import shutil

if __name__ == "__main__":
    proj_dir = pathlib.Path(__file__).parent.parent / "versioned_hdf5"
    for modname in ("cytools", "subchunk_map", "staged_changes"):
        src = proj_dir / f"{modname}.py"
        dst = proj_dir / f"{modname}.pyx"
        print(f"Copy {src} -> {dst}")
        shutil.copyfile(src, dst)
