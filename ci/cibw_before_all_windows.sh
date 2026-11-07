#!/bin/bash

set -eo pipefail

if [[ "$1" == "" ]] ; then
    echo "Usage: $0 <PROJECT_PATH>"
    exit 1
fi
PROJECT_PATH="$1"

if [[ "$ARCH" == "ARM64" ]]; then
    export ZLIB_ROOT="$PROJECT_PATH/zlib-win-arm64"
    export HDF5_VSVERSION="17-arm64"
elif [[ "$ARCH" == "AMD64" ]]; then
    export ZLIB_ROOT="$PROJECT_PATH/zlib-win-x64"
    export HDF5_VSVERSION="17-64"
else
    echo "Got unexpected arch '$ARCH'"
    exit 1
fi

echo "Building zlib into $ZLIB_ROOT"
./ci/get_zlib_windows.sh "$ZLIB_ROOT"

EXTRA_PATH="$ZLIB_ROOT/bin"
export CL="/I$ZLIB_ROOT/include"
export LINK="/LIBPATH:$ZLIB_ROOT/lib"

export PATH="$PATH:$EXTRA_PATH"

# requests is used by ci/get_pkg_config_lite.py and ci/get_hdf5_win.py below.
pip install requests

# pkg-config: meson resolves dependency('hdf5') through pkg-config, but GitHub's
# Windows images don't ship one.
# Do NOT install it with chocolatey (see the 'choco install dependencies (Windows)'
# step in .github/workflows/wheels.yml): the community feed rate-limits CI traffic
# and answers 503/504, choco can even exit 0 while having installed nothing, and the
# resulting breakage only shows up much later inside meson, as
#   Dependency lookup for hdf5 with method 'pkg-config' failed: Pkg-config for
#   machine host machine not found.
# Install the very same pkg-config-lite archive that the pkgconfiglite package
# fetches, straight from SourceForge.
# $PROJECT_PATH is a Windows path (e.g. D:\a\...): keep a POSIX copy for bash, as
# backslashes are not path separators there.
PROJECT_PATH_POSIX=$(cygpath -u "$PROJECT_PATH")
# The version and its checksum live in ci/get_pkg_config_lite.py.
export PKG_CONFIG_LITE_DIR="$PROJECT_PATH_POSIX/pkg-config-lite"
# Create it up-front: find would fail (and abort the script under `set -e`) if it
# didn't exist yet.
mkdir -p "$PKG_CONFIG_LITE_DIR"
# The archive wraps everything in a versioned directory
# (pkg-config-lite-<version>/bin/pkg-config.exe); where exactly it ends
# up is a detail of the zip, so locate the executable instead of hardcoding
# its path.
find_pkg_config_exe() {
    find "$PKG_CONFIG_LITE_DIR" -type f -name 'pkg-config.exe' -print -quit
}
PKG_CONFIG_EXE=$(find_pkg_config_exe)
if [[ -z "$PKG_CONFIG_EXE" ]]; then
    echo "Installing pkg-config-lite into $PKG_CONFIG_LITE_DIR"
    python "$PROJECT_PATH/ci/get_pkg_config_lite.py" \
        "$(cygpath -m "$PKG_CONFIG_LITE_DIR")"
    PKG_CONFIG_EXE=$(find_pkg_config_exe)
fi
if [[ -z "$PKG_CONFIG_EXE" ]]; then
    echo "Could not find pkg-config.exe under $PKG_CONFIG_LITE_DIR" >&2
    exit 1
fi
PKG_CONFIG_DIR=$(dirname "$PKG_CONFIG_EXE")
export PATH="$PATH:$PKG_CONFIG_DIR"
if ! "$PKG_CONFIG_EXE" --version > /dev/null 2>&1; then
    echo "pkg-config at $PKG_CONFIG_EXE is not runnable" >&2
    exit 1
fi
echo "Using $("$PKG_CONFIG_EXE" --version) from $PKG_CONFIG_DIR"

# HDF5
export HDF5_VERSION="2.0.0"
export HDF5_DIR="$PROJECT_PATH/cache/hdf5/$HDF5_VERSION"

python $PROJECT_PATH/ci/get_hdf5_win.py

if [[ "$GITHUB_ENV" != "" ]] ; then
    # PATH on windows is special
    echo "$EXTRA_PATH" | tee -a $GITHUB_PATH
    echo "CL=$CL" | tee -a $GITHUB_ENV
    echo "LINK=$LINK" | tee -a $GITHUB_ENV
    echo "ZLIB_ROOT=$ZLIB_ROOT" | tee -a $GITHUB_ENV
    echo "HDF5_DIR=$HDF5_DIR" | tee -a $GITHUB_ENV
    # pkg-config for meson's dependency('hdf5'); PATH on windows is special
    echo "$(cygpath -m "$PKG_CONFIG_DIR")" | tee -a $GITHUB_PATH
fi
