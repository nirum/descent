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
    'gradient_algorithms',
    'proximal_algorithms',
    'utils',
    ]

from .main import *
from .gradient_algorithms import *
from .proximal_algorithms import *
from .utils import *

__version__ = '0.0.1'
