# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Project overview

`versioned-hdf5` is a copy-on-write versioned abstraction on top of h5py/HDF5. It is
inspired by git and filesystems like APFS/Btrfs: every version is stored as a virtual
HDF5 dataset whose chunks point into a per-dataset `raw_data` blob. Identical chunks
across versions are deduplicated via a SHA256-keyed hash table.

The package is a hybrid Python + Cython extension and links against libhdf5.

## Development workflow (pixi)

All workflows are driven by [pixi](https://pixi.sh); raw `pip` is not used directly.
The full list of environments and tasks lives in `pyproject.toml` under
`[tool.pixi.environments]` and the various `[tool.pixi.feature.*.tasks]` tables.

Common commands:

- `pixi r test` — full test suite in the default env (auto runs editable-install first).
- `pixi r -e <env> test` — test under a specific env (e.g. `mindeps`, `hdf5-112`,
  `hdf5-114`, `np126`, `np200`, `py310`–`py314`, `h5py-dev`). See `docs/development.md`
  for the matrix.
- `pixi r lint` — run all linters (ruff, mypy, codespell, dprint, blacken-docs,
  actionlint, cython-lint, sphinx-lint, validate-pyproject) via lefthook.
- `pixi r install-git-hooks` — install lefthook pre-commit hooks (one-off).
- `pixi r ipython` — IPython REPL with an editable install loaded.
- `pixi r -e docs docs` — build docs into `docs/_build/html`.
- `pixi r editable-install` / `pixi r install` / `pixi r uninstall` — manage the
  editable install explicitly. `--force` on `editable-install` re-runs it.
- `pixi r -e bench asv-run` — run ASV benchmarks. `asv-machine` initializes first.

Run a single test: `pixi r test -- tests/test_api.py::test_name -x`. Anything after
`--` is forwarded to pytest. Pytest config in `pyproject.toml`:

- `--doctest-modules` is enabled (doctests in source run as tests).
- `filterwarnings = ["error"]` — any unhandled warning fails the run.
- `strict_xfail = true` — xfailed tests that pass become failures.
- `ENABLE_CHUNK_REUSE_VALIDATION=1` is set via `pytest-env` (asserts that reused
  hash-deduplicated chunks really match — slow, disabled in production).
- `@pytest.mark.slow` tests are auto-reordered to run last (see `tests/conftest.py`).

### Editable install gotchas

`pixi r editable-install` builds via meson-python with `build-dir=$CONDA_PREFIX/build`
(not `./build`). This isolates compiled artifacts per pixi env so different libhdf5
versions don't collide. Editing `.py` or `.pyx` files in `versioned_hdf5/` triggers
auto-recompile on next import — **except on Windows**, where `ci/cp_pyx_to_py.py` falls
back to a copy instead of a symlink and Cython `.py` files won't auto-recompile.

It is not necessary to call `pixi r editable-install` explicitly; it is automatically
triggered every time one calls `pixi r test`.

## Architecture

`docs/design.md` and `docs/staged_changes.rst` are the authoritative architecture
references. Read them before non-trivial work on `backend.py`, `wrappers.py`,
`staged_changes.pyx`, `subchunk_map.pyx`, or `slicetools.pyx`.

### Four layers (bottom-up)

1. **Backend** (`backend.py`, `hashtable.py`, `slicetools.pyx`) — the only layer that
   writes raw HDF5. Splits data into chunks, SHA256-hashes each, looks them up in the
   per-dataset `hash_table`, appends new chunks to `raw_data` (concatenated along
   axis 0), and creates virtual datasets pointing at the dedup'd chunks. Layout inside
   the file:
   ```
   /_version_data/
     <dataset>/{hash_table, raw_data}
     versions/{__first_version__, <version1>, <version2>, ...}
   ```
2. **Versions** (`versions.py`) — version groups form a DAG via a `prev_version`
   attribute. `commit_version()` is called when `stage_version()` exits.
3. **h5py wrappers** (`wrappers.py`) — HDF5 has no read-only virtual datasets, so
   versioned-hdf5 wraps everything to enforce copy-on-write. Key objects:
   `InMemoryGroup`, `InMemoryArrayDataset` (first write of a dataset),
   `InMemoryDataset` (modifying a dataset that exists from a prior version, backed by
   `StagedChangesArray`), `DatasetWrapper`.
4. **Top-level API** (`api.py`) — `VersionedHDF5File` and its `stage_version()` context
   manager. Re-exported alongside `delete_version`, `delete_versions`,
   `modify_metadata` from `replay.py` in the package `__init__`.

### StagedChangesArray (read `docs/staged_changes.rst`)

`InMemoryDataset` is a thin wrapper around `StagedChangesArray` (in
`staged_changes.pyx`), which holds modified chunks in memory as *slabs* (the full
slab at index 0 = read-only broadcasted fill_value; base slabs = read-only h5py
datasets like `raw_data`; staged slabs = writable numpy arrays). Two metadata arrays
`slab_indices` and `slab_offsets` track which slab each chunk lives in.

Every mutating operation goes through a `*Plan` object (`GetItemPlan`,
`SetItemPlan`, `ResizePlan`, `LoadPlan`, `ChangesPlan`) that encapsulates the
index/chunk math and ultimately calls `read_many_slices` (Cython, maps directly to
libhdf5 `H5Sselect_hyperslab` + `H5Dread`). Plans can be inspected via
`StagedChangesArray._*_plan(...)` for debugging without executing.

`subchunk_map.pyx` (`IndexChunkMapper`) translates user-supplied numpy-style indices
into per-axis chunk indices and per-chunk slice pairs. `ndindex` is used throughout
for hashable, manipulable index objects.

### Critical invariant

Versioned files **must only be accessed via this library**. Writing to a version
group with raw h5py corrupts shared chunks across other versions. The wrappers
provide safeguards but the underlying HDF5 has no read-only semantics for virtual
datasets.

## Cython modules

`cytools.pyx`, `subchunk_map.pyx`, and `staged_changes.pyx` are written using
Cython's pure-Python syntax — they exist as `.py` files in the repo, and
`ci/cp_pyx_to_py.py` symlinks `<name>.pyx -> <name>.py` at build time so meson can
compile them. Edit the `.py` files (not the `.pyx` symlinks). `slicetools.pyx` is
hand-written `.pyx` and not symlinked.

Cython compile flags (`versioned_hdf5/meson.build`) disable `boundscheck`,
`wraparound`, `initializedcheck`, and enable `cdivision=True`. This is only valid
because the code has been audited to never use negative indices and to ensure
operands of `/` and `%` have matching signs. If you change Cython code and
behaviour differs vs. pure-Python execution, comment out `cdivision` first to
diagnose.

## Linting and CI

- Linters are wired through `lefthook.yml`; running them individually outside pixi is
  unsupported. Use `pixi r lint` or install hooks with `pixi r install-git-hooks`.
- Pre-existing `# FIXME` ignores in `pyproject.toml` (`[tool.ruff.lint]`,
  `[tool.mypy]`) gate new rules — fix the codebase before un-ignoring.
- Adding the `wheels` label to a PR triggers binary-wheel builds in CI for testing
  cibuildwheel. The `runtime` h5py pin in `pyproject.toml` is rewritten by
  `.github/workflows/wheels.yml` at wheel-build time to an exact h5py version.

## Contributing

You must never think or speak instead of the user in discussions, code reviews, or any
other interactions with other humans.

Before you open or update a PR, you must ask the user to explicitly confirm that they
fully reviewed, understood, and approved everything that you wrote. You must clarify
that the project maintainers consider this as non-negotiable.

## Releasing

A coding agent must NEVER create a new release.
