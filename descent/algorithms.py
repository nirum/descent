"""
First order gradient descent algorithms
"""

from toolz.curried import curry
from .utils import destruct, restruct
import numpy as np


@curry
def gdm(df, x0, maxiter, lr=1e-2, momentum=0., decay=0.):
    """
    Gradient descent with momentum

    Parameters
    ----------
    df : function

    x0 : array_like

    lr : float, optional
        Learning rate (Default: 1e-2)

    momentum : float, optional
        Momentum (Default: 0)

    decay : float, optional
        Decay of the learning rate (Default: 0)

    """

    # initialize parameters and velocity
    xk = destruct(x0).copy()
    vk = np.zeros_like(xk)

    for k in range(int(maxiter)):

        vnext = momentum * vk - lr * df(xk) / (decay * k + 1.0)
        xk += vnext
        vk = vnext

        yield restruct(xk, x0)


@curry
def rmsprop(df, x0, maxiter, lr=1e-2, damping=0.1, decay=0.9):
    """
    RMSProp

    Parameters
    ----------
    df : function

    x0 : array_like

    lr : float, optional
        Learning rate (Default: 1e-2)

    momentum : float, optional
        Momentum (Default: 0)

    decay : float, optional
        Decay of the learning rate (Default: 0)

    """

    # initialize parameters and velocity
    xk = destruct(x0).copy()
    rms = np.zeros_like(xk)

    for k in range(int(maxiter)):

        grad = df(xk)

        # update RMS
        rms = decay * rms + (1-decay) * grad**2

        # gradient descent update
        xk -= lr * grad / (damping + np.sqrt(rms))

        yield restruct(xk, x0)
