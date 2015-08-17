"""
Useful callback functions
"""

from __future__ import print_function
import numpy as np
import tableprint as tp
from toolz import keyfilter, curry
from .utils import destruct

__all__ = ['disp', 'store']


@curry
def disp(d, every=1):
    """Print updates to the console"""

    # initial call
    if d['iter'] == 0:
        print('\n'.join((tp.hr(3),
                        tp.header(['Iteration', 'Objective', '||Grad||']),
                        tp.hr(3))), flush=True)

    if d['iter'] % every == 0:
        print(tp.row([d['iter'], d['obj'], np.linalg.norm(destruct(d['grad']))]), flush=True)


@curry
def store(db, data, keys=None):
    """Save the data in a list"""

    if keys is None:
        keys = ['obj', 'iter']

    db.append(keyfilter(lambda k: k in keys, data))
