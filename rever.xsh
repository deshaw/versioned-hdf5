from rever.activity import activity
from rever.conda import run_in_conda_env

$PROJECT = 'versioned-hdf5'

@activity
def run_tests():
    # Don't use the built-in pytest action because that uses Docker, which is
    # overkill and requires installing Docker
    with run_in_conda_env(['python=3.8', 'pytest', 'numpy', 'h5py',
                           'ndindex', 'pyflakes', 'pytest-cov']):
        pyflakes .
        python -We:invalid -We::SyntaxWarning -m compileall -f -q ndindex/
        pytest

@activity
def build_docs():
    with run_in_conda_env(['python=3.8', 'sphinx']):
        cd docs
        make html
        cd ..

@activity
def annotated_tag():
    # https://github.com/regro/rever/issues/212
    git tag -a -m "$GITHUB_REPO $VERSION release" $VERSION

$ACTIVITIES = [
            'authors',
            'run_tests',
            'build_docs',
            'changelog',  # Uses files in the news folder to create a changelog for release
            'annotated_tag', # Creates a tag for the new version number
            'push_tag',  # Pushes the tag up to the $TAG_REMOTE
            'ghrelease',  # Creates a Github release entry for the new tag
            'pypi'  # Sends the package to pypi
]

$PUSH_TAG_REMOTE = 'git@github.com:deshaw/versioned-hdf5.git'  # Repo to push tags to

$GITHUB_ORG = 'deshaw'  # Github org for Github releases and conda-forge
$GITHUB_REPO = 'versioned-hdf5'  # Github repo for Github releases and conda-forge
$CHANGELOG_FILENAME = 'docs/CHANGELOG.md'
$AUTHORS_FILENAME = 'AUTHORS.md'



