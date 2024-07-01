"""Install benchmarks for versioned-hdf5.

This is the script used by asv run to install dependencies. It should not be
called directly.

This is needed because we have to install specific versions of ndindex
depending on what commit we are on, because some backwards incompatible
changes in ndindex were made in tandem with corresponding commits in
versioned-hdf5.
"""

import os
import subprocess
import sys

# The first commit in versioned-hdf5 that is not compatible with ndindex 1.5;
# this was released in versioned-hdf5 1.2.2
# ndindex_1_5_pin_version = '1.2.2'
ndindex_16_commit = "af9ba2313c73cf00c10f490407956ed3c0e6467e"


def run(command, *args, **kwargs):
    print(" ".join(command), flush=True)
    kwargs.setdefault("check", True)
    return subprocess.run(command, *args, **kwargs)


def main():
    commit, env_dir, build_dir = sys.argv[1:]

    copy_env_dir(env_dir, commit)
    install_versioned_hdf5(build_dir)
    install_dependencies(commit, env_dir)


def copy_env_dir(env_dir, commit):
    # asv reuses the env dir between runs. But it's simpler for us if we just
    # restart from scratch, rather than trying to build an uninstall script.
    # So what we do is copy the raw env dir into a template directory, then
    # each time we install, we replace the env dir with that template
    # directory.
    template_dir = env_dir + "-template"
    if not os.path.exists(template_dir):
        # This is the first time we've run
        print("Creating template env directory", template_dir, flush=True)
        run(["cp", "-R", env_dir, template_dir])
    run(["rm", "-rf", env_dir])
    run(["cp", "-R", template_dir, env_dir])
    # asv checks out the project in the env directory, which we just reset. So
    # checkout it out to the correct commit.
    os.chdir(os.path.join(env_dir, "project"))
    run(["git", "checkout", commit])
    os.chdir(env_dir)


def install_dependencies(commit, env_dir):
    # Check if HEAD is after the ndindex_16_commit.
    # See https://stackoverflow.com/questions/3005392/how-can-i-tell-if-one-commit-is-a-descendant-of-another-commit
    p = run(
        ["git", "merge-base", "--is-ancestor", ndindex_16_commit, commit], check=False
    )
    if p.returncode == 1:
        print("Installing ndindex 1.5", flush=True)
        install(env_dir, ndindex_version="==1.5")
    elif p.returncode == 0:
        print("Installing ndindex >=1.5.1", flush=True)
        install(env_dir, ndindex_version=">=1.5.1")
    else:
        raise RuntimeError(
            f"Error checking commit history for benchmarks install (git gave return code {p.returncode})"
        )


def install_versioned_hdf5(build_dir):
    print("Installing versioned HDF5", flush=True)
    run(["python", "-m", "pip", "install", build_dir])


def install(env_dir, ndindex_version=">=1.5"):
    # Early versions of versioned-hdf5 pin ndindex>=1.5, but are actually incompatible
    # with ndindex>=1.5.1. This was fixed since versioned-hdf5==1.2.2, but here we
    # overwrite the ndindex version to ensure compatibility.
    deps = [
        "ndindex" + ndindex_version,
    ]
    run(["python", "-m", "pip", "install", *deps], check=True)


if __name__ == "__main__":
    main()
