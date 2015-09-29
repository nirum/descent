"""
Useful callback functions
"""

from __future__ import print_function
import numpy as np
import tableprint as tp
from toolz import keyfilter, curry
from .utils import destruct
from collections import namedtuple

__all__ = ['disp', 'store']
datum = namedtuple('Datum', ['iteration', 'objective', 'gradient', 'parameters', 'runtime'])


@curry
def disp(d, every=1):
    """Print updates to the console"""

    # initial call
    if d.iteration == 0:
        print('\n'.join((tp.hr(3),
                        tp.header(['Iteration', 'Objective', '||Grad||']),
                        tp.hr(3))), flush=True)

    if d.iteration % every == 0:
        print(tp.row([d.iteration, d.obj,
                     np.linalg.norm(destruct(d.grad))],
                     precision='5f'), flush=True)


@curry
def store(db, d, keys=None):
    """Save the data in a list"""

    if keys is None:
        keys = ['obj', 'iteration']

    db.append(keyfilter(lambda k: k in keys, d._asdict()))
