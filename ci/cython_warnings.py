"""Compile every Cython module and fail on any warning or performance hint.

Cython's '-Werror' promotes 'warning:' to an error, but it cannot promote
'performance hint:', which is emitted by Errors.performance_hint() regardless of
Options.warning_errors. So this compiles every module from scratch without '-Werror',
prints whatever Cython has to say, and exits non-zero if there is anything.

This is a standalone check rather than part of the build, because:
- pip hides the build log when the build succeeds, so hints would go unnoticed;
- a naked 'pip install' may run a different Cython version, whose extra warnings must
  not break third-party builds;
- the 'mindeps' environment pins Cython 3.0.10, whose diagnostics we don't control.
"""

import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

import Cython

ROOT = Path(__file__).parent.parent

# Above the Cython version pinned in the 'mindeps' environment (see pixi.toml) and
# below the version the other environments resolve to. Older Cython emits diagnostics
# that originate in legacy Cython code paths and in .pxd files we don't own.
MIN_CYTHON = (3, 2)

LANGUAGE_ARGS = [
    "-3",
]

# Must match the cython_args of versioned_hdf5/meson.build, which explains why each
# directive is safe.
DIRECTIVES = {
    "infer_types": True,
    "initializedcheck": False,
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
}
DIRECTIVE_ARGS = [
    arg for key, value in DIRECTIVES.items() for arg in ("-X", f"{key}={value}")
]

# Module name -> extra arguments. Keep in sync with versioned_hdf5/meson.build.
# Note: cytools, subchunk_map and staged_changes have pure Python syntax; meson builds
# them from a .pyx symlink made by ci/cp_pyx_to_py.py, which this script doesn't need.
STANDARD_MODULES = {
    "cytools": [],
    "subchunk_map": [],
    "staged_changes": ["--cplus"],
    "hash": [],
}

# Unlike the others, slicetools opts out of boundscheck, wraparound, cdivision and
# infer_types selectively, with per-function decorators, so it gets no -X flags.
NO_DIRECTIVE_MODULES = ["slicetools"]


def modules() -> list[tuple[str, list[str]]]:
    ret = [
        (mod, LANGUAGE_ARGS + DIRECTIVE_ARGS + extra)
        for mod, extra in STANDARD_MODULES.items()
    ]
    return ret + [(mod, LANGUAGE_ARGS) for mod in NO_DIRECTIVE_MODULES]


def get_source(mod: str) -> Path:
    """Locate the module source. The pure Python modules also have a .pyx symlink after
    a meson configure; compile the .py, which is the source of truth.
    """
    for suffix in (".py", ".pyx"):
        path = ROOT / "versioned_hdf5" / (mod + suffix)
        if path.exists():
            return path
    raise FileNotFoundError(f"No source found for {mod}")


def find_cython() -> str | None:
    """Locate the cython executable of the running interpreter's environment first;
    this is not on PATH when the script is invoked outside of 'pixi run'.
    """
    scripts = Path(sysconfig.get_path("scripts"))
    candidate = scripts / ("cython.exe" if os.name == "nt" else "cython")
    if candidate.is_file():
        return str(candidate)
    return shutil.which("cython")


def main() -> int:
    cython = find_cython()
    if cython is None:
        print("error: could not find the 'cython' executable", file=sys.stderr)
        return 1

    failed: set[str] = set()
    with tempfile.TemporaryDirectory() as tmpdir:
        for mod, args in modules():
            src = get_source(mod)
            dst = Path(tmpdir, mod).with_suffix(".cpp" if "--cplus" in args else ".c")
            proc = subprocess.run(
                [cython, *args, str(src), "-o", str(dst)],
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )
            # Cython reports warnings and performance hints on stderr and exits with 0
            diagnostics = proc.stdout + proc.stderr
            if diagnostics or proc.returncode:
                failed.add(mod)
                print(f"--- {src.relative_to(ROOT)} ---")
                print(diagnostics, end="", flush=True)
            if proc.returncode and not diagnostics:
                print(f"error: cython exited with {proc.returncode}", file=sys.stderr)

    if not failed:
        print("All Cython modules compiled without warnings.")
        return 0

    cython_version = tuple(int(part) for part in Cython.__version__.split(".")[:2])
    if cython_version < MIN_CYTHON:
        print(
            f"Cython {Cython.__version__} is older than "
            f"{MIN_CYTHON[0]}.{MIN_CYTHON[1]}; ignoring the diagnostics above.",
            file=sys.stderr,
        )
        return 0

    print(
        "error: Cython emitted warnings or performance hints for: "
        + ", ".join(sorted(failed)),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
