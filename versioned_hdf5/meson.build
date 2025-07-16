py.install_sources(
    [
        '__init__.py',
        'api.py',
        'backend.py',
        'h5py_compat.py',
        'hashtable.py',
        'replay.py',
        'tools.py',
        'typing_.py',
        'versions.py',
        'wrappers.py',
    ],
    subdir: 'versioned_hdf5',
)

# Adapted from https://numpy.org/doc/2.1/reference/random/examples/cython/meson.build.html
# Note:
# numpy 2.x -> site-packages/numpy/_core/include
# numpy 1.x -> site-packages/numpy/core/include
npy_include_path = run_command(
    py,
    ['-c', 'import numpy; print(numpy.get_include())'],
    check: true
).stdout().strip()

py.extension_module(
    'slicetools',
    'slicetools.pyx',
    install: true,
    subdir: 'versioned_hdf5',
    dependencies: compiled_deps,
    include_directories: [npy_include_path],
)

cython_args = [
    # '-a',  # Generate HTML report
    '-X', 'infer_types=True',
    # Comment these out if you experience segfaults to get a clear exception instead
    '-X', 'initializedcheck=False',
    '-X', 'boundscheck=False',
    # Note: this is possible because we went through all view accesses in the module
    # and made sure they always use non-negative indices.
    '-X', 'wraparound=False',
    # Note: this is possible only because we went through every single // and % in
    # the module and made sure the sign of the two operands is always the same.
    # Comment this out if behaviour changes when not compiled.
    '-X', 'cdivision=True',
]

# To generate HTML compilation reports:
# (First double check that flags below match those above)
#
# cythonize -3 -a -X infer_types=True -X initializedcheck=False -X boundscheck=False -X wraparound=False -X cdivision=True versioned_hdf5/{slicetools.pyx,cytools.py,subchunk_map.py,staged_changes.py}
#
# Or uncomment the -a flag above and then run:
# meson build && pushd build && meson compile && popd && firefox $(find build/ -name "*.html")
# (you'll need to install all build dependencies first).

# Hack to convince Meson to compile pure-python .py files with Cython
run_command('python', '../ci/cp_pyx_to_py.py', check: true)

py.extension_module(
    'cytools',
    'cytools.pyx',
    install: true,
    subdir: 'versioned_hdf5',
    dependencies: compiled_deps,
    cython_args: cython_args,
)

py.extension_module(
    'subchunk_map',
    'subchunk_map.pyx',
    install: true,
    subdir: 'versioned_hdf5',
    dependencies: compiled_deps,
    cython_args: cython_args,
)

py.extension_module(
    'staged_changes',
    'staged_changes.pyx',
    install: true,
    subdir: 'versioned_hdf5',
    dependencies: compiled_deps,
    cython_args: cython_args,
)
