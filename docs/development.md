# Development Guide

versioned-hdf5 workflows are based on [pixi](https://pixi.sh).
The full list of tasks and environments available can be seen in `pyproject.toml`.

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

| Environment | Python | NumPy | libhdf5 | h5py | ndindex | notes |
|---|---|---|---|---|---|---|
| `mindeps` | 3.10 | 1.24.4 | 1.10 ~ 1.14 | 3.8 ~ 3.9 | 1.5.1 | Versions depend on platform; pinned build stack |
| `hdf5-112` | 3.10 | 1.24.4 | 1.12 | 3.8 | latest | Not available on Windows |
| `np126` | 3.10 | 1.26 | latest | latest | latest | |
| `np200` | 3.10 | 2.0 | latest | latest | latest | |
| `py310` | 3.10 | 2.2 | latest | latest | latest | |
| `py311` | 3.11 | latest | latest | latest | latest | |
| `py312` | 3.12 | latest | latest | latest | latest | |
| `py313` | 3.13 | latest | latest | latest | latest | |
| `py314` | 3.14 | latest | latest | latest | latest | |
| `default` | latest | latest | latest | latest | latest | |
| `h5py-dev` | latest | latest | latest | git tip | latest | |

## Editable install
All pixi environments use a editable install of versioned-hdf5. This means that after
changing any file, including Cython ones, in the `versioned_hdf5/` directory, changes
will be immediately reflected in all commands (such as `pixi r test`).

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
