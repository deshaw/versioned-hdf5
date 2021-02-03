#!/bin/bash
# Run and update the benchmarks in the gh-pages branch. This commits to that
# branch but does not push. The resulting benchmarks are served at
# https://deshaw.github.io/versioned-hdf5/benchmarks/index.html
set -e
set -x

asv run -k
asv publish
git checkout gh-pages
git pull
rm -rf benchmarks
cp -R .asv/html benchmarks
git add benchmarks
git commit -m "Update benchmarks"
git checkout -
