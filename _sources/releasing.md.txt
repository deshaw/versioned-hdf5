# Releasing

## Steps before a release

- Before making a release, you need to make a changelog with the changes. The
  changelog is in `docs/changelog.md` (see {any}`changelog`).

- It's a good idea to push up your changes as a GitHub PR before doing the
  final release cut so that the CI runs all the tests. You can merge the PR
  once the release is made.

- Make sure to update version pinning in setup.py if necessary, e.g., if the
  minimum supported version of h5py or ndindex has changed.

## Doing the release

Versioned HDF5 uses [rever](https://regro.github.io/rever-docs/) for releases.
To do a release, install rever then run

```
$ rever <VERSION>
```

where `<VERSION>` is the version number for the release, like `rever 1.0`.

This will automatically run all the steps in the `rever.xsh` file.

```{note}
If you see an error like `json.decoder.JSONDecodeError: Expecting
value: line 1 column 1 (char 0)` this is due to a [bug in
rever](https://github.com/regro/rever/issues/229). Simply run the `rever`
command again and it will pisk up where it left off.
```

You will need push access to
[GitHub](https://github.com/deshaw/versioned-hdf5/pull/202) to push the tag
and push access to [PyPI](https://pypi.org/project/versioned-hdf5/) to push up
the release tarball. If you have never done a release before, the script will
ask you to authenticate.

## Updating conda-forge

The bot will make a pull request to the [conda-forge
feedstock](https://github.com/conda-forge/versioned-hdf5-feedstock/pulls)
automatically. You will need to be listed as a maintainer in the recipe to be
able to merge it. If you aren't, make a PR adding yourself.
