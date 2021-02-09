"""
This is the script used by asv run to install dependencies. It should not be
called directly.

This is needed because we have to install specific versions of ndindex
depending on what commit we are on, because some backwards incompatible
changes in ndindex were made in tandem with corresponding commits in
versioned-hdf5.
"""

import builtins
import sys
import os
import subprocess

# The first commit in versioned-hdf5 that is not compatible with ndindex 1.5
ndindex_16_commit = 'af9ba2313c73cf00c10f490407956ed3c0e6467e'

def print(*args):
    # If we don't flush stdout, print output is out of order with run()
    # output in the asv run -v log.
    builtins.print(*args)
    sys.stdout.flush()

def run(command, *args, **kwargs):
    print(' '.join(command))
    kwargs.setdefault('check', True)
    return subprocess.run(command, *args, **kwargs)

def main():
    commit, env_dir, build_dir = sys.argv[1:]

    copy_env_dir(env_dir, commit)
    install_dependencies(commit, env_dir)

    install_versioned_hdf5(build_dir)

def copy_env_dir(env_dir, commit):
    # asv reuses the env dir between runs. But it's simpler for us if we just
    # restart from scratch, rather than trying to build an uninstall script.
    # So what we do is copy the raw env dir into a template directory, then
    # each time we install, we replace the env dir with that template
    # directory.
    template_dir = env_dir + '-template'
    if not os.path.exists(template_dir):
        # This is the first time we've run
        print("Creating template env directory", template_dir)
        run(['cp', '-R', env_dir, template_dir])
    run(['rm', '-rf', env_dir])
    run(['cp', '-R', template_dir, env_dir])
    # asv checks out the project in the env directory, which we just reset. So
    # checkout it out to the correct commit.
    os.chdir(os.path.join(env_dir, 'project'))
    run(['git', 'checkout', commit])
    os.chdir(env_dir)

def install_dependencies(commit, env_dir):
    # Check if HEAD is after the ndindex_16_commit.
    # See https://stackoverflow.com/questions/3005392/how-can-i-tell-if-one-commit-is-a-descendant-of-another-commit
    p = run(['git', 'merge-base', '--is-ancestor', ndindex_16_commit, commit],
            check=False)
    if p.returncode == 1:
        print("Installing ndindex 1.5")
        install(env_dir, ndindex_version='==1.5')
    elif p.returncode == 0:
        print("Installing ndindex >=1.5.1")
        install(env_dir, ndindex_version='>=1.5.1')
    else:
        raise RuntimeError(f"Error checking commit history for benchmarks install (git gave return code {p.returncode})")

def install_versioned_hdf5(build_dir):
    print("Installing versioned HDF5")
    run(['python', '-m', 'pip', 'install', build_dir])

def install(env_dir, ndindex_version='>=1.5', h5py_version='<3'):
    deps = [
        'h5py' + h5py_version,
        'ndindex' + ndindex_version,
    ]
    run(['conda', 'install', '-c', 'conda-forge', '-p', env_dir, *deps], check=True)

if __name__ == '__main__':
    main()
