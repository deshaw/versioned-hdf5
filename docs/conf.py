# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# -- Project information -----------------------------------------------------

project = "Versioned HDF5"
copyright = "2020, Quansight"
author = "Quansight"

# Enable warnings for all bad cross references. These are turned into errors
# with the -W flag in the Makefile.
nitpicky = True

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    "sphinx_multiversion",
]

# sphinx-multiversion configuration
smv_tag_whitelist = r"^v?[0-9].[0-9].*$"
smv_branch_whitelist = "master"
smv_remote_whitelist = None
smv_released_pattern = r"^tags/.*$"
smv_outputdir_format = "{ref.name}"
smv_prefer_remote_refs = False

graphviz_output_format = "svg"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_type_aliases = {"File": "h5py.File"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

html_theme_options = {
    "github_user": "deshaw",
    "github_repo": "versioned-hdf5",
    "github_banner": False,  # https://github.com/bitprophet/alabaster/issues/166
    "github_button": False,
    "travis_button": False,
    "show_related": True,
    # Remove gray background from inline code
    "code_bg": "#FFFFFF",
    # Originally 940px
    "page_width": "1000px",
    # Fonts
    "font_family": (
        "Palatino, 'goudy old style', 'minion pro', 'bell mt', Georgia, "
        "'Hiragino Mincho Pro', serif",
    ),
    "font_size": "18px",
    "code_font_family": (
        "'Menlo', 'DejaVu Sans Mono', 'Consolas', 'Bitstream Vera Sans Mono', monospace"
    ),
    "code_font_size": "0.85em",
}

html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
        "versioning.html",
    ]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Lets us use single backticks for code
default_role = "code"

autodoc_typehints = "none"
