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

# HDF5
export HDF5_VERSION="2.0.0"
export HDF5_DIR="$PROJECT_PATH/cache/hdf5/$HDF5_VERSION"

pip install requests
python $PROJECT_PATH/ci/get_hdf5_win.py

# OpenSSL (libcrypto) for versioned_hdf5/hash.pyx.
# `choco install openssl` (see .github/workflows/wheels.yml) installs the Shining Light
# distribution but ships no pkg-config files, and its default path contains spaces,
# which pkg-config handles poorly. Copy the bits we need into a space-free directory and
# synthesise the .pc files so that meson's dependency('openssl') resolves it.
# delvewheel vendors libcrypto-*.dll into the wheel during the repair step.
OPENSSL_SRC=""
for cand in "/c/Program Files/OpenSSL-Win64" "/c/Program Files/OpenSSL" ; do
    if [[ -d "$cand/include/openssl" ]]; then OPENSSL_SRC="$cand"; break; fi
done
if [[ "$OPENSSL_SRC" == "" ]]; then
    echo "Could not locate the choco-installed OpenSSL" >&2
    exit 1
fi

export OPENSSL_DIR="$PROJECT_PATH/openssl"
mkdir -p "$OPENSSL_DIR"
cp -r "$OPENSSL_SRC/include" "$OPENSSL_DIR/"
cp -r "$OPENSSL_SRC/lib" "$OPENSSL_DIR/"
cp -r "$OPENSSL_SRC/bin" "$OPENSSL_DIR/"

# Locate the libcrypto import library. slproweb's layout has changed across OpenSSL
# releases: older builds put libcrypto.lib directly under lib/, newer ones (3.5+/4.x)
# put it under lib/VC/x64/MD/. Prefer the MD (dynamically-linked CRT) variant to match
# the Python and HDF5 builds; never pick the *_static variant.
echo "OpenSSL .lib files found:"
find "$OPENSSL_DIR/lib" -iname '*.lib' | sort
LIBCRYPTO_LIB=""
for cand in \
    "$OPENSSL_DIR/lib/VC/x64/MD/libcrypto.lib" \
    "$OPENSSL_DIR/lib/VC/x64/MD/libcrypto64MD.lib" \
    "$OPENSSL_DIR/lib/libcrypto.lib" ; do
    if [[ -f "$cand" ]]; then LIBCRYPTO_LIB="$cand"; break; fi
done
if [[ "$LIBCRYPTO_LIB" == "" ]]; then
    LIBCRYPTO_LIB=$(find "$OPENSSL_DIR/lib" -iname 'libcrypto*.lib' ! -iname '*static*' | head -1)
fi
if [[ "$LIBCRYPTO_LIB" == "" ]]; then
    echo "Could not find the libcrypto import library under $OPENSSL_DIR/lib" >&2
    exit 1
fi
echo "Using libcrypto import library: $LIBCRYPTO_LIB"
LIBCRYPTO_NAME=$(basename "$LIBCRYPTO_LIB" .lib)               # e.g. libcrypto / libcrypto64MD
LIBCRYPTO_LIBDIR_WIN=$(cygpath -m "$(dirname "$LIBCRYPTO_LIB")")

OPENSSL_WIN=$(cygpath -m "$OPENSSL_DIR")  # forward-slash Windows path, no spaces
mkdir -p "$OPENSSL_DIR/lib/pkgconfig"
cat > "$OPENSSL_DIR/lib/pkgconfig/libcrypto.pc" <<EOF
prefix=$OPENSSL_WIN
libdir=$LIBCRYPTO_LIBDIR_WIN
includedir=\${prefix}/include

Name: OpenSSL-libcrypto
Description: OpenSSL cryptography library
Version: 3.0.0
Libs: -L\${libdir} -l$LIBCRYPTO_NAME
Cflags: -I\${includedir}
EOF
cat > "$OPENSSL_DIR/lib/pkgconfig/openssl.pc" <<EOF
Name: OpenSSL
Description: Secure Sockets Layer and cryptography libraries
Version: 3.0.0
Requires: libcrypto
EOF

if [[ "$GITHUB_ENV" != "" ]] ; then
    # PATH on windows is special
    echo "$EXTRA_PATH" | tee -a $GITHUB_PATH
    echo "CL=$CL" | tee -a $GITHUB_ENV
    echo "LINK=$LINK" | tee -a $GITHUB_ENV
    echo "ZLIB_ROOT=$ZLIB_ROOT" | tee -a $GITHUB_ENV
    echo "HDF5_DIR=$HDF5_DIR" | tee -a $GITHUB_ENV
    echo "OPENSSL_DIR=$OPENSSL_DIR" | tee -a $GITHUB_ENV
    # cmake fallback for meson's dependency('openssl') if pkg-config is bypassed
    echo "OPENSSL_ROOT_DIR=$OPENSSL_DIR" | tee -a $GITHUB_ENV
fi
