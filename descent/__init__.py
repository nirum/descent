"""
Descent
=======

A Python package for performing first-order optimization
"""

__version__ = '0.2.0'

from .algorithms import *
from .proxops import *
from .utils import *
from .main import *

__all__ = [
    *algorithms.__all__,
    *proxops.__all__,
    *utils.__all__,
    *main.__all__,
]
