"""Build the project with meson and fail on any Cython warning or performance hint.

Cython's '-Werror' promotes 'warning:' to an error, but it cannot promote
'performance hint:', which is emitted by Errors.performance_hint() regardless of
Options.warning_errors. So this configures a throwaway meson build directory, builds
it, and scans meson's transcript of the cython runs.

Meson drives the compilation, so the module list, the input language and the '-X'
directives are the ones in meson.build and versioned_hdf5/meson.build, with no second
copy of them that could silently drift from the build.

This is a standalone check rather than part of the real build, because:
- pip hides the build log when the build succeeds, so hints would go unnoticed;
- a naked 'pip install' may run a different Cython version, whose extra warnings must
  not break third-party builds;
- the 'mindeps' environment pins Cython 3.0.10, whose diagnostics we don't control.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Above the Cython version pinned in the 'mindeps' environment (see pixi.toml) and
# below the version the other environments resolve to. Older Cython emits diagnostics
# that originate in legacy Cython code paths and in .pxd files we don't own.
MIN_CYTHON = (3, 2)

# Cython prints its diagnostics as "warning: <file>:<line>:<col>: <message>" and
# "performance hint: <file>:<line>:<col>: <message>". A C compiler instead prints
# "<file>:<line>:<col>: warning: <message>", which must not be picked up here.
DIAGNOSTIC_RE = re.compile(
    r"^(?P<kind>warning|performance hint): (?P<file>.+?):\d+:\d+: "
)


def meson(*args: str) -> subprocess.CompletedProcess[str]:
    """Run meson with the current interpreter, so that it picks up this environment's
    Cython, libhdf5, etc.
    """
    return subprocess.run(
        [sys.executable, "-m", "mesonbuild.mesonmain", *args],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )


def cython_version(build_dir: Path) -> tuple[int, ...]:
    """Version of the Cython executable that meson actually ran."""
    intro = build_dir / "meson-info" / "intro-compilers.json"
    info = json.loads(intro.read_text(encoding="utf-8"))
    version = info["host"]["cython"]["version"]
    return tuple(int(part) for part in version.split("."))


def find_diagnostics(log: str) -> dict[str, list[str]]:
    """Return {module: [diagnostic, ...]} for the Cython diagnostics in the log."""
    ret: dict[str, list[str]] = {}
    for line in log.splitlines():
        match = DIAGNOSTIC_RE.match(line)
        if match:
            module = Path(match["file"]).stem
            ret.setdefault(module, []).append(line)
    return ret


def main() -> int:
    try:
        import mesonbuild  # noqa: F401
    except ImportError:
        print(
            "error: meson is not installed; run this through 'pixi r cython-warnings'",
            file=sys.stderr,
        )
        return 1

    # A fresh build directory guarantees that every module is compiled from scratch,
    # rather than skipping the ones that were already built by whatever Cython and -X
    # flags happened to be in effect last.
    with tempfile.TemporaryDirectory(prefix="versioned-hdf5-cython-") as tmpdir:
        build_dir = Path(tmpdir)
        setup = meson("setup", str(build_dir), str(ROOT))
        if setup.returncode:
            print(setup.stdout + setup.stderr, end="", flush=True)
            print("error: meson setup failed", file=sys.stderr)
            return 1
        version = cython_version(build_dir)

        # Serial build: diagnostics from concurrent compilations would interleave
        proc = meson("compile", "-C", str(build_dir), "-j1")
        log = proc.stdout + proc.stderr
        if proc.returncode:
            print(log, end="", flush=True)
            print("error: meson failed to build the project", file=sys.stderr)
            return 1

        diagnostics = find_diagnostics(log)

    if not diagnostics:
        print("All Cython modules compiled without warnings.")
        return 0

    print(log, end="", flush=True)

    if version < MIN_CYTHON:
        print(
            f"Cython {'.'.join(map(str, version))} is older than "
            f"{MIN_CYTHON[0]}.{MIN_CYTHON[1]}; ignoring the diagnostics above.",
            file=sys.stderr,
        )
        return 0

    print(
        "error: Cython emitted warnings or performance hints for: "
        + ", ".join(sorted(diagnostics)),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
