"""
First order gradient descent algorithms

"""
import numpy as np
from collections import deque
from itertools import count
from .main import gradient_optimizer

__all__ = ['sgd', 'nag', 'rmsprop', 'sag', 'smorms', 'adam']


@gradient_optimizer
def sgd(lr=1e-3, mom=0.):
    """
    Stochastic gradient descent (SGD)

    Parameters
    ----------
    lr : float, optional
        Learning rate (Default: 1e-3)

    mom : float, optional
        Momentum (Default: 0.0)
    """

    # initial parameters
    xk = yield

    # initialize state variables
    vk = np.zeros_like(xk)

    # iterate
    for k in count():
        grad = yield xk
        vk = mom * vk - lr * grad
        xk += vk


@gradient_optimizer
def nag(lr=1e-3):
    """
    Nesterov's accelerated gradient (NAG)

    Parameters
    ----------
    lr : float, optional
        Learning rate (Default: 1e-3)
    """
    xk = yield
    yk = xk.copy()

    for k in count():
        grad = yield yk
        xprev = xk.copy()
        xk = yk - lr * grad
        yk = xk + (k / (k + 3.)) * (xk - xprev)


@gradient_optimizer
def rmsprop(lr=1e-3, damping=1e-12, decay=0.9):
    """
    RMSProp

    Parameters
    ----------
    lr : float, optional
        Learning rate (Default: 1e-3)

    damping : float, optional
        Damping term (Default: 1e-12)

    decay : float, optional
        Decay of the learning rate (Default: 0)

    """
    xk = yield
    rms = np.zeros_like(xk)

    for k in count():
        grad = yield xk
        rms *= decay
        rms += (1 - decay) * grad**2
        xk -= lr * grad / (damping + np.sqrt(rms))


@gradient_optimizer
def sag(nterms=10, lr=1e-3):
    """
    Stochastic Average Gradient (SAG)

    Parameters
    ----------
    nterms : int, optional
        Number of gradient evaluations to use in the average (Default: 10)

    lr : float, optional
        (Default: 1e-3)
    """
    xk = yield
    gradients = deque([], nterms)

    for k in count():
        grad = yield xk

        # push the new gradient onto the deque, update the average
        gradients.append(grad)

        # update
        xk -= lr * np.mean(gradients, axis=0)


@gradient_optimizer
def smorms(lr=1e-3, epsilon=1e-8):
    """
    Squared mean over root mean squared cubed (SMORMS3)

    Notes
    -----
    by Simon Funk
    http://sifter.org/~simon/journal/20150420.html

    Parameters
    ----------
    lr : float, optional
        (Default: 1e-3)

    epsilon : float, optional
        (Default: 1e-8)
    """
    xk = yield
    mem = np.ones_like(xk)
    g = np.zeros_like(xk)
    g2 = np.zeros_like(xk)

    for k in count():
        grad = yield xk

        r = 1 / (mem + 1)
        r_1 = 1. - r

        g *= r_1
        g += r * grad

        g2 *= r_1
        g2 += r * grad ** 2

        glr = g ** 2 / (g2 + epsilon)
        mem = 1 + mem * (1 - glr)

        xk -= grad * np.minimum(lr, glr) / (np.sqrt(g2) + epsilon)


@gradient_optimizer
def adam(lr=1e-3, beta=(0.9, 0.999), epsilon=1e-8):
    """
    ADAM

    Parameters
    ----------
    lr : float, optional
        Learnin rate (Default: 1e-3)

    beta : (float, float)
        (Default: (0.9, 0.999))

    epsilon : float
        (Default: 1e-8)
    """
    xk = yield
    mk = np.zeros_like(xk)
    vk = np.zeros_like(xk)
    b1, b2 = beta

    for k in count(start=1):
        grad = yield xk

        mk *= b1
        mk += (1. - b1) * grad

        # update velocity
        vk *= b2
        vk += (1. - b2) * (grad ** 2)

        # normalize
        momentum_norm = mk / (1 - b1 ** k)
        velocity_norm = np.sqrt(vk / (1 - b2 ** k))

        # gradient descent update
        xk -= lr * momentum_norm / (epsilon + velocity_norm)
