"""
Descent
=======

A Python package for performing first-order optimization

Modules
-------
algorithms      - Various gradient based optimization algorithms
utils           - Useful utilities for trasforming parameter data structures

For more information, see the accompanying README.md

"""

__all__ = [
    'main',
    'algorithms',
    'utils',
    'callbacks',
    'classify'
    ]

from .algorithms import *
from .main import *
from .callbacks import *
from .utils import *

__version__ = '0.0'
