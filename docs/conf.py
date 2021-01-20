# -*- coding: utf-8 -*-

# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.

import os
import sys
import shutil
import onnxmltools
import sphinx_readable_theme
import tabulate
import sphinx_gallery.gen_gallery




# -- Project information -----------------------------------------------------

project = 'onnxmltools'
copyright = '2018-2020, Microsoft'
author = 'Microsoft'
version = onnxmltools.__version__
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    "sphinx.ext.autodoc",
    'sphinx.ext.githubpages',
    "sphinx_gallery.gen_gallery",
    'sphinx.ext.autodoc',
]

templates_path = ['_templates']
source_suffix = ['.rst']

master_doc = 'index'
language = "en"
exclude_patterns = []
pygments_style = 'default'

# -- Options for HTML output -------------------------------------------------

html_static_path = ['_static']
html_theme = "readable"
html_theme_path = [sphinx_readable_theme.get_html_theme_path()]
html_logo = "ONNXMLTools_logo_main.png"

# -- Options for graphviz ----------------------------------------------------

graphviz_output_format = "svg"

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'https://docs.python.org/': None}

# -- Options for Sphinx Gallery ----------------------------------------------

sphinx_gallery_conf = {
     'examples_dirs': 'examples',
     'gallery_dirs': 'auto_examples',
}

# -- Setup actions -----------------------------------------------------------

def setup(app):
    # Placeholder to initialize the folder before
    # generating the documentation.
    return app

