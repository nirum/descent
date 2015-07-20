"""
Visualization tools
"""

import numpy as np
import matplotlib.pyplot as plt
from toolz import get, keyfilter, curry


@curry
def objective(values, iters=None, color='k', scale='log'):

    #assert type(results) == list, "Input must be a list"
    #assert results, "List must not be empty"
    assert scale in ['linear', 'log'], "Scale is linear or log"

    #if type(results[0]) == dict,
        #obj

    if iters is None:
        iters = np.arange(len(values))

    plt.plot(iters, values, '-')
    plt.gca().set_xscale(scale)
