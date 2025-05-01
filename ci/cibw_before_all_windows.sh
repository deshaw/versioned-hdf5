#!/bin/bash

set -eo pipefail

if [[ "$1" == "" ]] ; then
    echo "Usage: $0 <PROJECT_PATH>"
    exit 1
fi
PROJECT_PATH="$1"

# vcpkg -> pkgconf, so that meson can find HDF5
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
bootstrap-vcpkg.bat
vcpkg install pkgconf:x64-windows
export PKGCONFBIN="$GITHUB_WORKSPACE/vcpkg/installed/x64-windows/tools/pkgconf"

# Needed so that meson finds this one instead of the Strawberry Perl one, which
# doesn't work and is rejected by meson
mv $PKGCONFBIN/pkgconf.exe $PKGCONFBIN/pkg-config.exe

# HDF5
export HDF5_VERSION="1.14.6"
export HDF5_VSVERSION="17-64"
export HDF5_DIR="$PROJECT_PATH/cache/hdf5/$HDF5_VERSION"

pip install requests
python $PROJECT_PATH/ci/get_hdf5_win.py

# Work around glitch in generation of libhdf5 pkgconfig file
sed -i.bak /NOTFOUND/d $HDF5_DIR/lib/pkgconfig/hdf5.pc

if [[ "$GITHUB_ENV" != "" ]] ; then
    # PATH on windows is special
    echo "$EXTRA_PATH" | tee -a $GITHUB_PATH
    echo "$PKGCONFBIN" | tee -a $GITHUB_PATH
    echo "CL=$CL" | tee -a $GITHUB_ENV
    echo "LINK=$LINK" | tee -a $GITHUB_ENV
    echo "HDF5_DIR=$HDF5_DIR" | tee -a $GITHUB_ENV
    echo "PKG_CONFIG_PATH=$HDF5_DIR/lib/pkgconfig;$PKG_CONFIG_PATH" | tee -a $GITHUB_ENV
fi
