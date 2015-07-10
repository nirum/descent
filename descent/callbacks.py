"""
Useful callback functions
"""

import numpy as np
import tableprint as tp
from toolz import get, keyfilter, curry


@curry
def disp(d, every=1):
    """Print updates to the console"""

    # initial call
    if d['iter'] == 0:
        print(('\n'.join((tp.hr(3),
                        tp.header(['Iteration', 'Objective', '||Grad||']),
                        tp.hr(3)))))

    if d['iter'] % every == 0:

        k, obj, grad = get(['iter', 'obj', 'grad'], d)
        print((tp.row([k, obj, np.linalg.norm(grad)])))


@curry
def store(db, keys, d):
    """Save the data in a list"""

    db.append(keyfilter(lambda k: k in keys, d))
