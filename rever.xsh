from rever.activity import activity
from rever.conda import run_in_conda_env

$PROJECT = 'versioned-hdf5'

@activity
def run_tests():
    # Don't use the built-in pytest action because that uses Docker, which is
    # overkill and requires installing Docker
    with run_in_conda_env(['python=3.8', 'pytest', 'numpy', 'h5py',
                           'ndindex', 'pyflakes', 'pytest-cov',
                           'scipy', 'pytest-doctestplus', 'pytest-flakes',
                           'doctr', 'sphinx']):
        pyflakes versioned_hdf5/
        python -We:invalid -We::SyntaxWarning -m compileall -f -q versioned_hdf5/
        pytest versioned_hdf5/

@activity
def build_docs():
    with run_in_conda_env(['python=3.8', 'sphinx', 'myst-parser', 'numpy',
                           'h5py', 'ndindex']):
        cd docs
        make html
        cd ..

@activity
def annotated_tag():
    # https://github.com/regro/rever/issues/212
    git tag -a -m "$GITHUB_REPO $VERSION release" $VERSION

$ACTIVITIES = [
            'run_tests',
            # 'build_docs',
            'annotated_tag', # Creates a tag for the new version number
            'push_tag',  # Pushes the tag up to the $TAG_REMOTE
            # 'ghrelease',  # Creates a Github release entry for the new tag
            'pypi',  # Sends the package to pypi
            # 'ghpages',
]

$PUSH_TAG_REMOTE = 'git@github.com:deshaw/versioned-hdf5.git'  # Repo to push tags to

$GITHUB_ORG = 'deshaw'  # Github org for Github releases and conda-forge
$GITHUB_REPO = 'versioned-hdf5'  # Github repo for Github releases and conda-forge

$GHPAGES_REPO = 'git@github.com:deshaw/versioned-hdf5.git'
$GHPAGES_COPY = $GHPAGES_COPY = (
    ('docs/_build/html', '$GHPAGES_REPO_DIR'),
)
