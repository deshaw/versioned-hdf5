Changelog
=========

## 2.2.1 (2026-01-07)

Fix wheels publishing issue on Linux.

## 2.2.0 (2026-01-07)

### Major Changes

- Fixed binary incompatibility of versioned-hdf5 Linux wheels vs. the wheels for
  h5py >=3.15.0. Starting from this release, versioned-hdf5 wheels for Linux on PyPi
  require h5py >=3.15.0 wheels. MacOSX wheels, conda-forge packages, and builds from
  source still require h5py >=3.8.0.
- Added wheels for Python 3.14
- Dropped support for Python 3.9

### Minor Changes

Filters have been overhauled:
- Added `shuffle`, `fletcher32`, and `scaleoffset` parameters to `create_dataset` and
  `modify_metadata`
- Fixed bug where `modify_metadata` would revert `compression` and `compression_opts` to
  their default value when they are not explicitly listed. For example,
  `modify_metadata(ds, fillvalue=123)` would decompress a dataset. You now have to
  explicitly pass `modify_metadata(ds, compression=None)`.
- `.compression` and `.compression_opts`  properties now return the numerical IDs for
  custom compression filters (e.g. Blosc, Blosc2) in staged datasets. Note that this is
  unlike `h5py.Dataset.compression`, which incorrectly returns None.
- Fixed bug where the `.compression` and `.compression_opts` properties of staged datasets
  would incorrectly return `None`
- Fixed bug where passing a path to `create_dataset` would silently disregard the
  `compression` and `compression_opts` parameters

## 2.1.0 (2025-08-12)

### Major Changes

- Binaries are now available:
  - conda-forge packages for Linux, MacOSX, and Windows;
  - pip wheels for Linux and MacOSX (but not Windows).

  See [Installation](installation).
- Added support for StringDType, a.k.a. NpyStrings.
  Requires h5py >=3.14.0 and NumPy >=2.0.
  [Like in h5py](https://docs.h5py.org/en/stable/strings.html#numpy-variable-width-strings),
  StringDType is completely opt-in. [Read more](strings).
- The `astype()` method is now functional; it returns a lazy read-only accessor just
  like in h5py.
- Fixed breakage in `delete_versions()` after upgrading to h5py 3.14.0.

### Minor Changes

- Fixed crash when handling empty multidimensional datasets.
- Fixed bugs where `InMemoryArrayDataset.__getitem__()` and `__array__()`
  would return the wrong dtype.
- The `.chunks` property of a Dataset would occasionally return an unnecessary
  `ndindex.ChunkType`; it now always returns a tuple of ints like in h5py.
- Fixed bug where the `ENABLE_CHUNK_REUSE_VALIDATION` environment variable would incur
  in a false positive when the first element of an object array is bytes, but then later
  on there are str elements.
- Overhauled ASV benchmarks support.

- `resize()`:
  - Fixed crash in `InMemorySparseDataset.resize()`.
  - Fixed issue where calling `InMemoryArrayDataset.resize()` to enlarge a dataset was
    slow and caused huge memory usage.

- `create_dataset()`:
  - The `dtype=` parameter now accepts any DTypeLike, e.g. `"i4"` or `int`, in
    addition to actual `numpy.dtype` objects.
  - Fixed bug that could lead to overflow/underflow/loss of definition after calling
    `__setitem__` on the initial version of a dataset.
  - Avoid unnecessary double dtype conversions when passing a list to the `data=`
    parameter with an explicit `dtype=` parameter.
  - Fixed crashes when `chunks=` parameter was either a bare integer or True.
  - `chunks=False` is now explicitly disallowed
    (before it would lead to an uncontrolled crash).
  - Warn for all ignored kwargs, not just for `maxshape`.

- `modify_metadata()`:
  - Fixed bug where setting `fillvalue=0` would retain the previous fillvalue.
  - Speed up when the fillvalue doesn't change.
  - Fixed bug when changing dtype and fillvalue at the same time, which could cause the
    new fillvalue to be transitorily cast to the old dtype and overflow, underflow,
    or lose definition.

## 2.0.2 (2025-01-23)

### Minor Changes

- Fixed regression which would cause a crash when invoking `resize()` with a tuple of
  `numpy.int64` as argument instead of a tuple of ints, e.g. such as one constructed
  from `h5py.Dataset.size`.

## 2.0.1 (2025-01-22)

### Minor Changes

- Fixed regression, introduced in v2.0.0, which would cause the chunk hash map to become
  corrupted when calling `resize()` to shrink a dataset followed by `delete_versions()`.

## 2.0.0 (2024-12-05)

### Major Changes

- `stage_dataset` has been reimplemented from scratch. The new engine is
  expected to be much faster in most cases. [Read more here](staged_changes).
- `__getitem__` on staged datasets used to never cache data when reading from
  unmodified datasets (before the first call to `__setitem__` or `resize()`) and
  used to cache the whole loaded area on modified datasets (where the user had
  previously changed a single point anywhere within the same staged version).

  This has now been changed to always use the libhdf5 cache. As such cache is very
  small by default, users on slow disk backends may observe a slowdown in
  read-update-write use cases that don't overwrite whole chunks, e.g. `ds[::2] += 1`.
  They should experiment with sizing the libhdf5 cache so that it's larger than the
  work area, e.g.:

  ```python
  with h5py.File(path, "r+", rdcc_nbytes=2**30, rdcc_nslots=100_000) as f:
      vf = VersionedHDF5File(f)
      with vf.stage_version("r123") as sv:
          sv["some_ds"][::2] += 1
  ```

  (this recommendation applies to plain h5py datasets too).

  Note that this change exclusively impacts `stage_dataset`; `current_version`,
  `get_version_by_name`, and `get_version_by_timestamp` are not impacted and
  continue not to cache anything regardless of libhdf5 cache size.
- Added support for Ellipsis (...) in indexing.

## 1.8.2 (2024-11-21)

### Major Changes

- Integer array and boolean array indices are transparently converted to slices when
  possible, either globally or locally to each chunk.
  This can result in major speedups.
- Monotonic ascending integer array indices have been sped up from O(n^2) to O(n*logn)
  (where n is the number of chunks along the indexed axis).

### Minor Changes

- `as_subchunk_map` has been reimplemented in Cython, providing a speedup
- Improved the exceptions raised by `create_dataset`
- Fixed a libhdf5 resource leak in `build_data_dict`;
  the function has also been sped up.
- Slightly sped up hashing algorithm

## 1.8.0 (2024-08-09)

### Major Changes

- `slicetools` has been reimplemented in Cython, providing a significant speedup
- Only sdist will be published from here on out due to the dependency on MPI.
- Improved read/write performance for `InMemoryDataset`

### Minor Changes

- Force the master branch to be targeted when building docs
- `__version__` dunder added back in
- Update build workflows to test with `numpy==1.24` in addition to `numpy>=2`
- Chunk reuse verification fixed for string dtype arrays
- Cleaned up `pytest` configuration; added additional debugging output in test CI job
- Fixed a bug where `InMemoryGroup` child groups were not closed when the parent
  group is closed
- Nondefault compression handling is now supported
- Performance improvements to Hashtable initialization
- Various refinements to the documentation

## 1.7.0 (2024-06-10)

### Major Changes

- Added a new `VersionedHDF5File.get_diff` method
- Added a new `VersionedHDF5File.versions` property
- Updates to the build system to use `meson-python`
- Added numpy 2.0 support
- Make the `InMemoryGroup` repr more informative

### Minor Changes

- Optimizations to `_recreate_raw_dataset`, `InMemoryDataset.resize`
- Added an optional check for verifying that reused chunks contain the expected
  data. Can be turned on by setting the environment variable:
  `ENABLE_CHUNK_REUSE_VALIDATION = 1`
- The documentation for all published versions is now available!
- Various DevEx improvements: `pre-commit`, `pygrep-hooks`-fixes, and tests now
  will not produce unwanted artifacts
- Dataset names are now checked against a blocklist to avoid colliding with
  reserved words

## 1.6.0 (2023-11-15)

### Major Changes

- String lengths are now included with variable-length string array hashes. This
  solves an issue with hashing uniqueness with string arrays.

### Minor Changes

- Fixed an error message directly passing a file name to VersionedHDF5File.
- Removed all posixpath "pp" aliases, to prevent conflict with prettyprint debug
  command.
- Fix deprecated CI parameter.

## 1.5.0 (2023-10-19)

### Major Changes

- Fixed an issue with rebuilding hashtables; new `DATA_VERSION` released

### Minor Changes

- Rewrote a test for improved maintainability

## 1.4.3 (2023-10-05)

### Minor Changes

- Fix to avoid writing to temporary dataset during `delete_versions`, reducing
  the chance for out-of-memory errors.
- Fix to `_rebuild_hashtables` to allow for rebuilding hashtables of datasets
  nested arbitrarily deeply inside groups.
- A sensible default chunk size is now guessed by default if none is specified
  when creating 1D datasets.

## 1.4.2 (2023-09-13)

### Minor Changes

- Fix to prevent a quadratic runtime in `delete_versions`

## 1.4.1 (2023-09-13)

### Minor Changes

- Fix an issue where calling `delete_versions` on a dataset with a variable
  length string dtype before using `h5repack` on it would corrupt the file

## 1.4.0 (2023-09-11)

### Major Changes

- Fix an issue where hashes of object dtypes were not being correctly computed
  ([#257](https://github.com/deshaw/versioned-hdf5/pull/257)). This change
  modifies how the hashes are computed for object dtypes.

### Minor Changes

- Remove old rever-based version release system in favor of GitHub workflows.

## 1.3.14 (2023-07-27)

### Major Changes

- Fix issues with the latest version of h5py.

### Minor Changes

- Fix an issue where empty datasets deleted with `delete_versions()` could
  result in corrupted versions.

- Add info log showing stats about data upon exiting stage_version context.

- Drop support for Python 3.7.

- Remove all Travis CI configuration.

- Various minor cleanups to the test suite.

## 1.3.13 (2023-01-30)

### Minor Changes

- Fix the setting of `prev_version` in `delete_versions()`.

## 1.3.12 (2022-08-17)

### Minor Changes

- Fix variable length strings in _recreate_virtual_dataset. Note that for h5py
  3.7.0 they cause a segfault without the upstream fix
  [h5py/h5py#2111](https://github.com/h5py/h5py/pull/2111).

## 1.3.11 (2022-08-02)

### Minor Changes

- Improve the performance of creating versions in h5py 3.

## 1.3.10 (2022-06-16)

### Minor Changes

- Fix deleting the current version.
- Properly handle fillvalue-only datasets in `delete_versions()`.

## 1.3.9 (2022-04-28)

### Minor Changes

- Fix `delete_versions()` with nested groups.
- Fix `delete_versions()` with variable length strings.

## 1.3.8 (2022-03-29)

### Major Changes

- Versioning wrappers are now skipped when loading a file in read-only mode.
  This leads to better performance in these cases.

### Minor Changes

- Fix the test suite with newer versions of pytest.

## 1.3.7 (2022-01-27)

### Major Changes
- `delete_version` has been renamed to `delete_versions`, and now takes a list
  of versions to delete. The old `delete_version` is kept intact for backwards
  compatibility.
- `delete_versions` (née `delete_version`) is now much faster.

## 1.3.6 (2021-10-19)

### Minor Changes

- Create hashtable datasets with lzf compression enabled.

## 1.3.5 (2021-09-30)

### Minor Changes

- Fix a bug with the hashtable introduced in 1.3.4.

## 1.3.4 (2021-09-27)

### Minor Changes

- Make the hashtable dataset much smaller for datasets with no data or version
  history.

## 1.3.3 (2021-07-02)

### Minor Changes

- Fix a regression that prevented indexing a dataset with a mask from working.

## 1.3.2 (2021-07-28)

### Minor Changes

- Improve the performance of reading a reading a dataset that hasn't been
  written to (e.g., reading from an already committed version).

- Fix Versioned HDF5 to work with h5py 3.3.

- Fix an issue that would occur when using np.datetime64 objects for
  timestamps when the fractional part of the second was exactly 0.

## 1.3.1 (2021-05-20)

### Minor Changes

- Avoid some unnecessary precomputation in the hashtable object. This improves
  the performance for files that have many versions in them.

## 1.3 (2021-05-07)

### Major Changes

- Support h5py 3. h5py 2.10 is also still officially supported.

- Add functionality to replay versions. This allows mutating old versions.

- Add new helper functions {any}`delete_version` and {any}`modify_metadata`
  to delete a version and modify metadata on a dataset that must be the same
  across versions, such as chunk size or dtype.

- Add helper function {any}`recreate_dataset` for more advanced version
  replaying functionality.

### Minor Changes

- Disallow accessing vfile[version] before version has been committed. Doing
  so previously could lead to inconsistencies.

- Disallow accessing non-versioned groups from VersionedHDF5File.

- Better error messages from VersionedHDF5File when the underlying file is
  closed.

- Remove codecov.

- Some improvements to the benchmarking suite.

## 1.2.6 (2021-04-20)

### Minor Changes

- Fix a bug where chunks could be deleted from a dataset.

- Workaround an upstream h5py bug in the tests.

## 1.2.5 (2021-04-15)

### Minor Changes

- Fix a bug where attrs could be deleted from a dataset.

## 1.2.4 (2021-04-08)

### Major Changes

- Many improvements to performance throughout the library, particularly for
  datasets with many chunks, and for looking up versions by timestamp. This
  also sets up potential future performance improvements by automatically
  converting sparse staged datasets to fully in-memory.

- Add some additional benchmarks to the benchmark suite.

## 1.2.3 (2021-02-25)

### Minor Changes

- Fix the length of str dtype not being maintained from the fillvalue.

## 1.2.2 (2021-02-04)

### Minor Changes

- Many improvements to performance throughout the library.

- Added a benchmarking suite using [airspeed
  velocity](https://asv.readthedocs.io/en/stable/). Graphs of the benchmarks
  can be viewed at <https://deshaw.github.io/versioned-hdf5/benchmarks/>.

- Versioned HDF5 now depends on ndindex version 1.5.1 or greater.

## 1.2.1 (2020-12-30)

### Minor Changes

- Python 3.6 support has been dropped. The lowest version of Python now
  supported is 3.7.
- Fix creating a completely empty sparse dataset
- Use ndindex.ChunkSize internally. This is the beginning of an overhaul that
  improves the performance of many operations. ndindex 1.5 or newer is now
  required.

## 1.2 (2020-11-17)

### Major Changes

- Add support for sparse datasets (`data=None`).
- Store the chunks on an attribute of the dataset.
- versioned-hdf5 is currently pinned to `h5py<3`. h5py 3 support will be added
  in a future version.
- `VersionedHDF5File[timestamp]` now returns the closest version before
  `timestamp` if there is no version at `timestamp`.

## 1.1 (2020-09-15)

### Major Changes

* Added support for shape-0 datasets.
* Fix a memory leak where in-memory datasets would not be garbage collected.
* Added support for empty datasets (size 0).
* Make sure versioned data is read-only after closing and reopening the file.
* Allow deleting groups and datasets between versions. Note that currently
  dataset metadata cannot change between versions, even if they are deleted in
  between.

### Minor Changes

* Most tests now use a temporary directory instead of writing a file in the
  current directory.
* Fix logic for handling trailing slashes with `in`.
* Automatically create intermediate groups when creating a dataset.
* Make indices that should give a scalar object do so instead of giving a
  shape () array.

## 1.0 (2020-08-03)

### Major Changes

* First release of Versioned-HDF5.
