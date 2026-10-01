# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Project overview

`versioned-hdf5` is a copy-on-write versioned abstraction on top of h5py/HDF5, inspired
by git and copy-on-write filesystems like APFS/Btrfs. Every version is stored as a
virtual HDF5 dataset whose chunks point into a per-dataset `raw_data` blob; identical
chunks across versions are deduplicated via a SHA256-keyed hash table.

The package is a hybrid Python + Cython extension and links against libhdf5.

## Development workflow (pixi)

All workflows are driven by [pixi](https://pixi.sh); raw `pip` is not used directly.
Tasks and environments live in `pixi.toml`; project config and tools (ruff, mypy, ...)
live in `pyproject.toml`. `docs/development.md` documents the environments and their
version matrix.

- `pixi r test` — test suite in the default env (auto runs `editable-install` first).
- `pixi r -e <env> test` — same in another env (`mindeps`, `hdf5-112`, `hdf5-114`,
  `hdf5-21`, `np126`, `np200`, `py310`–`py314`, `h5py-dev`).
- `pixi r lint` — run all linters (ruff, mypy, codespell, dprint, blacken-docs,
  actionlint, cython-lint, sphinx-lint, validate-pyproject) via lefthook.
- `pixi r install-git-hooks` — install lefthook pre-commit hooks (one-off).
- `pixi r ipython` — REPL with the editable install loaded.
- `pixi r -e docs docs` — build docs into `docs/_build/html`.
- `pixi r editable-install` / `pixi r install` / `pixi r uninstall` — manage the install
  explicitly. `--force` on `editable-install` re-runs it.
- `pixi r asv-run` / `pixi r -b bench asv-compare <rev>...` — benchmarks. `asv-machine`
  initializes ASV first.

Anything after the task name is forwarded to pytest, e.g. `pixi r test
tests/test_api.py::test_stage_version -x`.

### Iterating on tests: skip the slow tests

**The full test suite takes ~10 minutes. Don't run it while iterating.** Use

```bash
pixi r test -m 'not slow'
```

instead: ~14 seconds instead of ~10 min, skipping only a dozen tests. Run the full suite
at most once, at the end, when the work is done. When working on a PR, do even better:
run `pixi r test -m 'not slow'` while iterating; once you think you are finished and the
fast test suite is green, do `git push` *first*, then start the full suite locally so it
runs in parallel with GitHub CI (which runs the full suite on every PR anyway).

Pytest config in `pyproject.toml`:

- `--doctest-modules` — doctests in source run as tests.
- `filterwarnings = ["error"]` — any unhandled warning fails the run.
- `strict_xfail = true` — xfailed tests that pass become failures.
- `@pytest.mark.slow` tests are reordered to run last (see `tests/conftest.py`).

### Editable install gotchas

`pixi r editable-install` builds via meson-python with `build-dir=$CONDA_PREFIX/build`
(not `./build`), so compiled artifacts are isolated per pixi env and different libhdf5
versions don't collide. Editing a `.py` or `.pyx` file in `versioned_hdf5/` triggers
auto-recompile on next import — **except on Windows**, where `ci/cp_pyx_to_py.py` falls
back to a copy instead of a symlink, so the Cython `.py` files don't auto-recompile.

You never need to call `pixi r editable-install` explicitly: it is a dependency of every
task that needs the package, including `pixi r test`.

## Architecture

`docs/design.md` and `docs/staged_changes.rst` are the authoritative references. Read
them before non-trivial work on `backend.py`, `wrappers.py`, `staged_changes.py`,
`subchunk_map.py`, or `slicetools.pyx`.

### Four layers (bottom-up)

1. **Backend** (`backend.py`, `hashtable.py`, `slicetools.pyx`) — the only layer that
   writes raw HDF5. Splits data into chunks, SHA256-hashes each, looks them up in the
   per-dataset `hash_table`, appends new chunks to `raw_data` (concatenated along axis
   0), and creates virtual datasets pointing at the dedup'd chunks. Layout inside the
   file:
   ```
   /_version_data/
     <dataset>/{hash_table, raw_data}
     versions/{__first_version__, <version1>, <version2>, ...}
   ```
2. **Versions** (`versions.py`) — version groups form a DAG via a `prev_version`
   attribute. `commit_version()` is called when `stage_version()` exits.
3. **h5py wrappers** (`wrappers.py`) — HDF5 has no read-only virtual datasets, so
   versioned-hdf5 wraps everything to enforce copy-on-write. Key objects:
   `InMemoryGroup`, `InMemoryArrayDataset` (first write of a dataset), `InMemoryDataset`
   (modifying a dataset that exists from a prior version, backed by
   `StagedChangesArray`), `DatasetWrapper`.
4. **Top-level API** (`api.py`) — `VersionedHDF5File` and its `stage_version()` context
   manager. Re-exported in the package `__init__` alongside `delete_version`,
   `delete_versions` and `modify_metadata` from `replay.py`.

### StagedChangesArray (read `docs/staged_changes.rst`)

`InMemoryDataset` is a thin wrapper around `StagedChangesArray` (in
`staged_changes.py`), which holds modified chunks in memory as *slabs* (the full slab at
index 0 = read-only broadcasted fill_value; base slabs = read-only h5py datasets like
`raw_data`; staged slabs = writable numpy arrays). Two metadata arrays `slab_indices`
and `slab_offsets` track which slab each chunk lives in.

Every mutating operation goes through a `*Plan` object (`GetItemPlan`, `SetItemPlan`,
`ResizePlan`, `LoadPlan`, `CommitPlan`) that encapsulates the index/chunk math and
ultimately calls `read_many_slices` (Cython, maps directly to libhdf5
`H5Sselect_hyperslab` + `H5Dread`). Plans can be inspected via
`StagedChangesArray._*_plan(...)` for debugging without executing.

`IndexChunkMapper` in `subchunk_map.py` translates user-supplied numpy-style indices
into per-axis chunk indices and per-chunk slice pairs. `ndindex` is used throughout for
hashable, manipulable index objects.

### Critical invariant

Versioned files **must only be accessed via this library**. Writing to a version group
with raw h5py corrupts shared chunks across other versions. The wrappers provide
safeguards, but the underlying HDF5 has no read-only semantics for virtual datasets.

## Cython modules

`cytools.pyx`, `subchunk_map.pyx` and `staged_changes.pyx` are written in Cython's
pure-Python syntax: they exist as `.py` files in the repo, and `ci/cp_pyx_to_py.py`
symlinks `<name>.pyx -> <name>.py` at build time so meson can compile them. **Edit the
`.py` files**, not the `.pyx` symlinks. `hash.pyx` and `slicetools.pyx` are hand-written
`.pyx` and not symlinked.

Cython compile flags (`versioned_hdf5/meson.build`) disable `boundscheck`, `wraparound`
and `initializedcheck`, and enable `cdivision=True`. The bounds flags are only valid
because the code never uses negative indices; `cdivision` only because the operands of
every `/` and `%` were audited to have matching signs. If Cython code behaves
differently from the pure-Python execution, comment out `cdivision` first to diagnose.

## Linting and CI

- Linters are wired through `lefthook.yml`; running them individually outside pixi is
  unsupported. Use `pixi r lint`, or `pixi r install-git-hooks` to run them on every
  commit.
- Pre-existing `# FIXME` ignores in `pyproject.toml` (`[tool.ruff.lint]`, `[tool.mypy]`)
  gate new rules — fix the codebase before un-ignoring.
- Adding the `wheels` label to a PR triggers binary-wheel builds in CI. The
  `h5py>=3.8.0` pin in `pyproject.toml` is rewritten by `.github/workflows/wheels.yml`
  at wheel-build time to the exact `H5PY_VERSION`.

### Binary wheels share h5py's libhdf5

The wheels do **not** ship their own libhdf5. After cibuildwheel builds and
auditwheel/delocate vendor libhdf5, `wheels.yml` deletes it again and the extensions
rely at runtime on the libhdf5 bundled in the installed h5py wheel (RPATH points at
`h5py.libs` / `h5py/.dylibs`). This is mandatory, not an optimization: versioned-hdf5
and h5py run in the same process and pass raw HDF5 object IDs (file/dataset/dataspace
handles) to each other. Two separate libhdf5 instances would each hold their own copy of
HDF5's global state, so an ID minted by one would be meaningless to the other →
corruption or crashes. (libcrypto, used by `hash.pyx`, is the exception: h5py does not
ship it, so it stays vendored.)

For the borrow to work, the extensions must be built against a **byte-identical**
libhdf5 to the one the published h5py wheel ships, so that auditwheel/delocate compute
the same mangled SONAME (`libhdf5-<hash>.so.<ver>`). h5py builds its wheels in the
`ghcr.io/h5py/manylinux_2_28_{x86_64,aarch64}-hdf5` images, which we also use — but
those images float on `:latest` and h5py rebuilds them independently of any h5py
release. A rebuild produces a different libhdf5 hash and breaks `import versioned_hdf5`
in the Linux smoke test with a missing `libhdf5-<hash>.so`, even though the build jobs
stay green. Therefore `manylinux-{x86_64,aarch64}-image` in `[tool.cibuildwheel]` must
be **pinned to a digest** matching `H5PY_VERSION`, never left on `:latest`. Bump the
digests in lockstep with `H5PY_VERSION`; pick the digest from the `Digest: sha256:...`
line of the image pull in a green CI run's "Build wheel" log. macOS and Windows are
unaffected: they build HDF5 themselves instead of borrowing it.

## Contributing

You must never think or speak instead of the user in discussions, code reviews, or any
other interactions with other humans.

When the user asks you to open or update a PR, follow the rules in the `open-pr` skill
(`.agents/skills/open-pr/SKILL.md`).

## Releasing

A coding agent must NEVER create a new release.
