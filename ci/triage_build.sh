#!/bin/bash

set -eo pipefail

if [[ "$1" == "" ]] || [[ "$2" == "" ]]; then
    echo "Usage: $0 <ARCH> <SHA>"
    exit 1
fi

ARCH=$1
SHA=$2
MSG="$(git show -s --format=%s $SHA)"
KIND="$RUNNER_OS $ARCH"

CIBW_SKIP="pp* *musllinux*"
CIBW_BUILD="*_$ARCH"
echo "CIBW_BUILD=$CIBW_BUILD" | tee -a $GITHUB_ENV
echo "CIBW_SKIP=$CIBW_SKIP" | tee -a $GITHUB_ENV
