# Development Guide

versioned-hdf5 workflows are based on [pixi](https://pixi.sh).
The full list of tasks and environments available can be seen in `pixi.toml`.

## Run tests

To run the full test suite in the default environment:

```bash
pixi r test
```

You can also select a different test environment:

```bash
pixi r -e <environment> test
```

You can choose among the following:

| Environment | Python | NumPy  | hdf5        | h5py      | ndindex | notes                                           |
| ----------- | ------ | ------ | ----------- | --------- | ------- | ----------------------------------------------- |
| `mindeps`   | 3.10   | 1.24.4 | 1.10 ~ 1.14 | 3.8 ~ 3.9 | 1.5.1   | Versions depend on platform; pinned build stack |
| `hdf5-112`  | 3.10   | 1.24.4 | 1.12        | 3.8       | latest  | Not available on Windows                        |
| `hdf5-114`  | 3.10   | 2.2    | 1.14        | latest    | latest  |                                                 |
| `hdf5-21`   | 3.10   | 2.2    | 2.1         | latest    | latest  |                                                 |
| `np126`     | 3.10   | 1.26   | latest      | latest    | latest  |                                                 |
| `np200`     | 3.10   | 2.0    | latest      | latest    | latest  |                                                 |
| `py310`     | 3.10   | 2.2    | latest      | latest    | latest  |                                                 |
| `py311`     | 3.11   | latest | latest      | latest    | latest  |                                                 |
| `py312`     | 3.12   | latest | latest      | latest    | latest  |                                                 |
| `py313`     | 3.13   | latest | latest      | latest    | latest  |                                                 |
| `py314`     | 3.14   | latest | latest      | latest    | latest  |                                                 |
| `py315`     | 3.15   | latest | latest      | latest    | latest  |                                                 |
| `py314t`    | 3.14t  | latest | latest      | latest    | latest  | Free-threaded, GIL disabled                     |
| `py315t`    | 3.15t  | wheel  | latest      | sdist     | latest  | Free-threaded, GIL disabled, h5py from sources  |
| `default`   | 3.14   | latest | latest      | latest    | latest  |                                                 |
| `h5py-dev`  | 3.15   | latest | latest      | git tip   | latest  |                                                 |

*Note:* hdf5 2.0 is not available in conda-forge and is untested.

*Note:* hdf5plugin has no cp315t build, so `py315t` cannot install it and the tests
needing third-party filters skip there.

## Free-threaded environments

`py314t` and `py315t` use conda-forge `python-freethreading` pinned to the matching
major.minor version (3.14 / 3.15), so the interpreter starts with the GIL disabled.
conda-forge has no `cp315t` builds of h5py or numpy, so `py315t` also installs numpy's
free-threaded wheel and builds h5py from its sdist against conda-forge's hdf5; its
environment therefore sets `no-default-feature = true`.

These environments must **not** set `PYTHON_GIL=0`. Importing a compiled extension
that does not declare `Py_mod_gil` re-enables the GIL and makes CPython emit a
`RuntimeWarning`; `filterwarnings = ["error"]` in `pyproject.toml` turns that warning
into a pytest failure. `PYTHON_GIL=0` suppresses both the warning and the GIL
re-enable, so a non-conforming extension would go completely unnoticed.
`tests/test_free_threading.py` also asserts `not sys._is_gil_enabled()`, which catches
a re-enable that happened before pytest installed its warning filters (for example in
an entry-point plugin), and the root `conftest.py` checks again at session end. All
five Cython modules are compiled with `-X freethreading_compatible=True` (see below).

hdf5plugin needs no scrutiny here: it ships no CPython extension module, only
`libh5*.so` filter plugins that libhdf5 `dlopen`s directly, so it cannot re-enable the
GIL. `py314t` installs it; `py315t` cannot, as conda-forge has no cp315t build.

Both environments run the normal `test` task and an additional parallel test task:

```bash
pixi r -e py314t test-parallel
```

`test-parallel` runs the whole configured test suite (including doctests) with
`pytest --parallel-threads 4`. Tests that genuinely cannot run concurrently are marked
`thread_unsafe` / listed in `thread_unsafe_fixtures` and still execute on one thread.
Shared fixture files are thread-unique via `make_name()` in `tests/conftest.py`.

CI runs `smoke`, `test -v` and `test-parallel -v` for both `py314t` and `py315t`. CI
runs `test-parallel` on the full suite, including the slow tests; locally use
`pixi r -e py314t test-parallel -m 'not slow'` while iterating.

### Thread safety model

h5py serializes all libhdf5 calls with its interpreter-wide, reentrant `phil` lock.
versioned-hdf5 reuses that same lock (`versioned_hdf5._lock`) as its *only* lock.
Each public API call, wrapper method, version setup/commit step and cache mutation
acquires `phil` for the duration of that individual call. `stage_version()` does not
hold `phil` across user-controlled code: setup and commit are separate locked
critical sections, and staged mutations are individually locked. Direct h5py I/O in
another thread therefore cannot deadlock behind a versioned-hdf5 context. There is no
second lock, so there is no lock-order inversion against `phil`. Pure `nogil`
computation in the Cython modules is unchanged. HDF5 work still stays serialized even
with the GIL off; free-threading allows non-HDF5 work to run concurrently.

## Editable install

All pixi environments use an editable install of versioned-hdf5. This means that after
changing any file, including Cython ones, in the `versioned_hdf5/` directory, changes
will be immediately reflected in all commands (such as `pixi r test`).

## Cython warnings

`pixi r cython-warnings` tests that Cython does not emit any `warning:` or `performance
hint:` messages in meson's log. It is a standalone check and not part of the build.
`-Werror` promotes `warning:` but not `performance hint:` to an error, so the build
cannot enforce a clean compilation. Users pip-installing versioned-hdf5 from sources may
still get warnings if they use a Cython version that's different from CI.

## Interactive terminal

`pixi r ipython` opens an interactive IPython terminal with versioned-hdf5 (editable
install from `versioned_hdf5/`) and all dependencies available.

## Documentation

- `pixi r -e docs docs` builds the documentation for the current version in
  `docs/_build/html`.
- `pixi r -e docs docs-multiversion` builds the documentation for all versions, present
  and past, in `docs/_build/html/<version>`. This is what is published to the public
  documentation page.

## Linting

versioned-hdf5 uses static type checkers, enforced by CI.
You should run them with `pixi r lint` before submitting a PR.
Alternatively, you may run once `pixi r install-git-hooks` to have the static type
checkers run on every commit.

## Benchmarks

versioned-hdf5 uses ASV for performance benchmarks.
At the moment, historical benchmarking is not configured.

- To initialise ASV: `pixi r asv-machine`
- To run ASV: `pixi r asv-run`

These commands are available in the `default` and `mindeps` environments.

### Comparing revisions

`pixi r -b bench asv-compare <rev1> <rev2> [... <revn>]` benchmarks two or more git revisions and
prints `asv compare` of each of them against the first one:

```console
$ pixi r asv-compare 1c10d29 nocpp -b TimeCommitPlan
[...]
| Change   | Before [1c10d295] <commit>   | After [9653508e] <nocpp>   |   Ratio | Benchmark (Parameter)                            |
|----------|------------------------------|----------------------------|---------|--------------------------------------------------|
| +        | 2.74±0.05ms                  | 11.7±0.6ms                 |    4.27 | ...TimeCommitPlan.time_commit_plan('all_new')    |
|          | 89.0±1ms                     | 93.9±1ms                   |    1.06 | ...TimeCommitPlan.time_commit_plan('fragmented') |
| +        | 1.03±0.02ms                  | 3.65±0.05ms                |    3.55 | ...TimeCommitPlan.time_commit_plan('no_changes') |
```

Each revision is built from a temporary git worktree. Your editable install of the
working tree is restored at the end.

`benchmarks/` always comes from the working tree, never from the revisions, so you can
benchmark revisions that predate the benchmark you're running on them (as long as it
doesn't crash on them).

Each benchmark is sampled 3 times per process to measure noise (`-n/--repeat`), where
`pixi r asv-run` measures it once. Pass `-b/--bench <regex>` to select individual tests.
Benchmarks marked as `@slow` are skipped unless one sets the environment variable
`ASV_RUNSLOW=1`.

## Automated Contributions and AI Policy

AI and automated tools may be used to assist in code development, documentation, and
testing, subject to the same standards that apply to contributions authored entirely by
humans. In particular:

- You (the human contributor) are responsible for everything you submit.
- Do not use tools to think or speak for you in discussions, code reviews, or any other
  interactions.
- Disclose use of AI and automated tools in accordance with the CLA and in the commit
  message or PR description.
- Before you open a PR, fully review, understand, and approve everything AI or other
  automated tool wrote or otherwise provided as output.
- The AI agent may open **draft** PRs that are clearly tagged as being written by
  unsupervised AI. The agent should use the `open-pr` skill to do so.

### TL;DR version

**Maintainers will close, without reading it, any issue or PR that looks written by AI.**
First draft written by AI is OK. Final PR is not.
