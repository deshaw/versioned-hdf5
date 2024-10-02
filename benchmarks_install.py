"""Install benchmarks for versioned-hdf5.

This is the script used by asv run to install dependencies. It should not be
called directly.

This is needed because we have to install specific versions of ndindex
depending on what commit we are on, because some backwards incompatible
changes in ndindex were made in tandem with corresponding commits in
versioned-hdf5.
"""
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

# The first commit in versioned-hdf5 that is not compatible with ndindex 1.5;
# this was released in versioned-hdf5 1.2.2
ndindex_1_5_pin_version = '1.2.2'

def main():
    env_dir, wheel_file, commit = sys.argv[1:]

    # asv default:
    # run(["in-dir={env_dir}" python -mpip install {wheel_file}"],
    run(["python", "-m", "pip", "install", wheel_file])
    install_dependencies(commit)


def run(command, *args, **kwargs):
    print(" ".join(command), flush=True)
    kwargs.setdefault("check", True)
    return subprocess.run(command, *args, **kwargs)


def install_dependencies(commit):
    # Check if HEAD is after 1.2.2
    # See https://stackoverflow.com/questions/3005392/how-can-i-tell-if-one-commit-is-a-descendant-of-another-commit
    p = run(
        ["git", "merge-base", "--is-ancestor", ndindex_1_5_pin_version, commit],
        check=False,
    )
    if p.returncode == 1:
        print(
            "Overriding ndindex version 1.5 because commit"
            f"{commit} comes before release {ndindex_1_5_pin_version}.",
            flush=True,
        )
        # Early versions of versioned-hdf5 pin ndindex>=1.5, but are actually incompatible
        # with ndindex>=1.5.1. This was fixed since versioned-hdf5==1.2.2, but here we
        # overwrite the ndindex version to ensure compatibility.
        run(["python", "-m", "pip", "install", "ndindex==1.5"], check=True)
    elif p.returncode == 0:
        return
    else:
        raise RuntimeError(
            "Error checking commit history for benchmarks install: "
            f'"git merge-base --is-ancestor {ndindex_1_5_pin_version} {commit}" '
            f"gave return code {p.returncode}"
        )


if __name__ == "__main__":
    main()
