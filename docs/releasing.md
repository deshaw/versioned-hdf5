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

To do a release, simply create a new release from the GitHub web ui. Click
`Draft a new release`, then create a new tag containing the version number.

Once the release has been made, the `pypi_publish.yml` workflow will run,
building wheels and publishing them to PYPI. The workflow uses trusted
publishing, meaning that github itself uploads the release to PYPI without
needing authentication from the maintainer.

## Updating conda-forge

The bot will make a pull request to the [conda-forge
feedstock](https://github.com/conda-forge/versioned-hdf5-feedstock/pulls)
automatically. You will need to be listed as a maintainer in the recipe to be
able to merge it. If you aren't, make a PR adding yourself.
