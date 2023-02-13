# docs/conf.py
"""Sphinx configuration."""
project = "flops-profiler"
author = "Cheng Li"
copyright = f"2020, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    'sphinx.ext.viewcode',
    "sphinx_autodoc_typehints",
    'sphinx.ext.todo',
]