"""
Descent
=======

A Python package for performing first-order optimization

For more information, see the accompanying README.md

"""

__all__ = [
    'algorithms',
    'proxops',
    'utils',
    'main',
    ]

from .algorithms import *
from .proxops import *
from .utils import *
from .main import *

__version__ = '0.1.2'
