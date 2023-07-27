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
html_short_title = f"{release}"
html_theme = "furo"
html_show_copyright = False
html_show_sphinx = False
html_static_path = ["_static"]
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
    "council.css",
]
# fmt: off
# so that black does not mess with it
html_theme_options = {
    "footer_icons": [
        {
            "name": "Discord",
            "url": "https://discord.gg/uhusYQcP",
            "html": "",
            "class": "fa-brands fa-solid fa-discord",
        },
        {
            "name": "Github",
            "url": "https://github.com/chain-ml/council",
            "html": "",
            "class": "fa-brands fa-solid fa-github",
        },
    ],
    "dark_logo": "Council_RGB_Horizontal_DarkBKG_Gradient.png",
    "light_logo": "Council_RGB_Horizontal_LightBKG_Gradient.png",
    "sidebar_hide_name": False,
}
# fmt: on
