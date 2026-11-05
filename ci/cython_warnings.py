"""Build the project with meson and fail on any Cython warning or performance hint.

Cython's '-Werror' promotes 'warning:' to an error, but it cannot promote 'performance
hint:', which is emitted by Errors.performance_hint() regardless of
Options.warning_errors.

This is a standalone check rather than part of the real build, because:

- pip hides the build log when the build succeeds, so hints would go unnoticed;
- a naked 'pip install' may run a different Cython version, whose extra warnings must
  not break third-party builds;
- the 'mindeps' environment pins an older version of Cython, whose diagnostics we don't
  control.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Cython prints its diagnostics as "warning: <file>:<line>:<col>: <message>" and
# "performance hint: <file>:<line>:<col>: <message>". A C compiler instead prints
# "<file>:<line>:<col>: warning: <message>", which must not be picked up here.
DIAGNOSTIC_RE = re.compile(
    r"^(?P<kind>warning|performance hint): (?P<file>.+?):\d+:\d+: "
)


def meson(*args: str) -> tuple[int, str]:
    """Run meson with the current interpreter, so that it picks up this environment's
    Cython, libhdf5, etc.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "mesonbuild.mesonmain", *args],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    log = proc.stdout + proc.stderr
    print(log)
    if proc.returncode:
        print("error: meson failed")
        return 1, log
    return 0, log


def main() -> int:
    # A fresh build directory guarantees that every module is compiled from scratch,
    # rather than skipping the ones that were already built by whatever Cython and -X
    # flags happened to be in effect last.
    with tempfile.TemporaryDirectory() as tmpdir:
        build_dir = Path(tmpdir)
        err, _ = meson("setup", str(build_dir), str(ROOT))
        if err:
            return err

        # Serial build: diagnostics from concurrent compilations would interleave
        err, log = meson("compile", "-C", str(build_dir), "-j1")
        if err:
            return err

    if any(DIAGNOSTIC_RE.match(line) for line in log.splitlines()):
        print("error: Cython emitted warnings or performance hints")
        return 1

    print("All Cython modules compiled without warnings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
