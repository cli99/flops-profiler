from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# docs/conf.py
"""Sphinx configuration."""
project = 'flops-profiler'
author = 'Cheng Li'
copyright = f'2020, {author}'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx.ext.todo',
]
