# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

import time

project = 'citk'
copyright = '2020, Dmytro Androsov, Volodymyr Sydorskyy'
author = 'Dmytro Androsov, Volodymyr Sydorskyy'

# The full version, including alpha/beta/rc tags
release = '0.1a'

version = '0.1'

date = '%d.%m.%Y'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 
'sphinx.ext.githubpages', 
'sphinx.ext.intersphinx', 
'sphinx.ext.mathjax', 
'rst2pdf.pdfbuilder',
'notfound.extension'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

html_title='Computational Intelligence Toolkit'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

stylesheets="fruity.json"

pdf_documents = [('index', u'citk_0.1', u'CITK: Computational Intelligence Toolkit', author),]

pdf_stylesheets = ['sphinx', 'kerning', 'a4']

language='en'

html_last_updated_fmt = '%d.%m.%Y'

today_fmt = '%d.%m.%Y'