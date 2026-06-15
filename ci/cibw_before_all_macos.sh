#!/bin/bash

set -eo pipefail

if [[ "$1" == "" ]] ; then
    echo "Usage: $0 <PROJECT_PATH>"
    exit 1
fi
PROJECT_PATH="$1"
ARCH=$(uname -m)
export HDF5_VERSION="2.0.0"
export HDF5_DIR="$PROJECT_PATH/cache/hdf5/$HDF5_VERSION-$ARCH"
source $PROJECT_PATH/ci/get_hdf5_if_needed.sh

# libcrypto (OpenSSL) for versioned_hdf5/hash.pyx. delocate vendors libcrypto into the
# wheel during the repair step, so the wheel stays self-contained (h5py does NOT ship
# libcrypto, unlike libhdf5).
#
# We build OpenSSL from source rather than using Homebrew's openssl@3: Homebrew bottles
# are built for the runner's current macOS (e.g. 15.0), but our wheels target an older
# MACOSX_DEPLOYMENT_TARGET (11.0 arm64 / 10.9 intel, matching the HDF5 build above), and
# delocate refuses to vendor a dylib whose minimum target is newer than the wheel's.
# Building from source with the right -mmacosx-version-min produces a compatible libcrypto.
export OPENSSL_VERSION="3.4.1"
export OPENSSL_DIR="$PROJECT_PATH/cache/openssl/$OPENSSL_VERSION-$ARCH"

if [[ "$ARCH" == "arm64" ]]; then
    export MACOSX_DEPLOYMENT_TARGET="11.0"
    OSSL_TARGET="darwin64-arm64-cc"
else
    export MACOSX_DEPLOYMENT_TARGET="10.9"
    OSSL_TARGET="darwin64-x86_64-cc"
fi

if [ -f "$OPENSSL_DIR/lib/libcrypto.dylib" ]; then
    echo "Using cached OpenSSL build at $OPENSSL_DIR"
else
    echo "Building OpenSSL $OPENSSL_VERSION for $OSSL_TARGET (min macOS $MACOSX_DEPLOYMENT_TARGET)"
    pushd /tmp
    curl -fsSL -o "openssl-$OPENSSL_VERSION.tar.gz" \
        "https://github.com/openssl/openssl/releases/download/openssl-$OPENSSL_VERSION/openssl-$OPENSSL_VERSION.tar.gz"
    mkdir -p "openssl-$OPENSSL_VERSION"
    tar -xzf "openssl-$OPENSSL_VERSION.tar.gz" --strip-components=1 -C "openssl-$OPENSSL_VERSION"
    pushd "openssl-$OPENSSL_VERSION"
    ./Configure "$OSSL_TARGET" shared no-tests no-docs \
        --prefix="$OPENSSL_DIR" --openssldir="$OPENSSL_DIR/ssl" \
        "-mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET"
    make -j"$(sysctl -n hw.ncpu)"
    make install_sw
    popd
    popd
fi

if [[ "$GITHUB_ENV" != "" ]]; then
    echo "HDF5_DIR=$HDF5_DIR" | tee -a $GITHUB_ENV
    echo "OPENSSL_DIR=$OPENSSL_DIR" | tee -a $GITHUB_ENV
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" | tee -a $GITHUB_ENV
    echo "PKG_CONFIG_PATH=$PKG_CONFIG_PATH" | tee -a $GITHUB_ENV
fi
