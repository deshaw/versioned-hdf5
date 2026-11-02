"""Decide whether Cython warnings must abort the build. Prints 'true' or 'false'.

meson calls this at configure time and appends '-Werror' to the Cython arguments if the
answer is 'true'. Both conditions below must be met:

1. VH5_CYTHON_WARNINGS_AS_ERRORS is set. Only the pixi build tasks set it, so that a
   naked 'pip install .' (and cibuildwheel, and asv_compare.py, and any third-party
   build) still succeeds on warnings, at the price of a noisy build log.

2. Cython is at least MIN_CYTHON. The 'mindeps' environment pins Cython 3.0.10, whose
   warnings mostly originate in legacy Cython code paths and in .pxd files we don't
   own, and which we have no intention of fixing. Only the modern Cython, as used by
   every other environment, is held to a zero-warnings standard.

Note that meson caches the answer in the build directory, so toggling the environment
variable requires re-configuring (or wiping) the build directory.
"""

import os

# Keep this above the Cython version pinned in the 'mindeps' environment (see
# [feature.mindeps.dependencies] in pixi.toml), and below the version that the other
# environments resolve to.
MIN_CYTHON = (3, 2)


def main() -> int:
    if not os.environ.get("VH5_CYTHON_WARNINGS_AS_ERRORS"):
        print("false")
        return 0

    import Cython

    major, minor = (int(part) for part in Cython.__version__.split(".")[:2])
    print("true" if (major, minor) >= MIN_CYTHON else "false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
