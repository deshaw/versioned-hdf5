py.install_sources(
    [
        '__init__.py',
        'api.py',
        'backend.py',
        'hashtable.py',
        'replay.py',
        'versions.py',
        'wrappers.py',
        'data_dict.py',
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

compiled_deps = [
    dependency('hdf5'),
    dependency('mpi'),
]

py.extension_module(
    'slicetools',
    'slicetools.pyx',
    install: true,
    subdir: 'versioned_hdf5',
    dependencies: compiled_deps,
    include_directories: [npy_include_path],
    cython_args: ['--cplus'],
    override_options : ['cython_language=cpp'],
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
# cythonize -3 -a -X infer_types=True -X initializedcheck=False -X boundscheck=False -X wraparound=False -X cdivision=True versioned_hdf5/*.pyx
#
# Or uncomment the -a flag above and then run:
# meson build && pushd build && meson compile && popd && firefox $(find build/ -name "*.html")

py.extension_module(
    'subchunk_map',
    # FIXME This is a symlink. Can't find a way to convince Meson to compile
    # with Cython pure-python .py files
    # (read: https://cython.readthedocs.io/en/latest/src/tutorial/pure.html)
    'subchunk_map.pyx',
    install: true,
    subdir: 'versioned_hdf5',
)
