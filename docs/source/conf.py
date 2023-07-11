# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import toml

with open("../../pyproject.toml") as f:
    data = toml.load(f)

project = "Council"
copyright = "2023, ChainML"
author = "ChainML"
release = data["project"]["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.duration",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinxcontrib.mermaid",
]

# nbsphinx
nbsphinx_execute = "never"
# somehow we need to disable requirejs to so that mermaid keeps working
nbsphinx_requirejs_path = ""

# autodoc
autodoc_default_options = {
    "show-inheritance": True,
    "members": None,
    "inherited-members": True,
}

# sphinx.ext.todo
todo_include_todos = True

# sphinx.ext.napoleon
napoleon_include_init_with_doc = True

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = f"{project} {release}"
html_theme = "furo"
html_show_copyright = False
html_static_path = ["_static"]
html_theme_options = {"dark_logo": "00_chainml_logo.png", "light_logo": "02_chainml_logo_black.png"}
