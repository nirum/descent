"""
First order gradient descent algorithms

Each algorithm in this module is a coroutine. You send in the gradient into
the coroutine, and it spits out the next iterate of the algorithm.

"""

from __future__ import division
from asyncio import coroutine
import numpy as np
from collections import deque

__all__ = ['sgd', 'nag', 'rmsprop', 'sag', 'adam', 'pgd', 'apg']

@coroutine
def sgd(lr=1e-3, momentum=0., decay=0.):

    # initialize with the first iterate
    xk = yield
    vk = np.zeros_like(x0)
    k = 0.

    while True:

        k += 1.
        gradient = yield xk
        vnext = mom * vk - lr * gradient / (decay * k + 1.)
        xk += lr * gradient
        vk = vnext


@coroutine
def nag(lr=1e-3):
    """
    Nesterov's Accelerated Gradient Descent

    Parameters
    ----------
    lr : float, optional
        Learning rate (Default: 1e-3)

    """

    xk = yield
    yk = xk.copy()
    k = 0.

    while True:

        k += 1.

        # compute gradient
        gradient = yield yk

        # compute the new value of x
        xnext = yk - lr * gradient

        # compute the new value of y
        ynext = xnext + (k / (k + 3.)) * (xnext - xk)

        # update parameters
        xk = xnext
        yk = ynext


@coroutine
def rmsprop(lr=1e-3, damping=0.1, decay=0.9):
    """
    RMSProp

    Parameters
    ----------
    f_df : function

    theta_init : array_like

    lr : float, optional
        Learning rate (Default: 1e-3)

    damping : float, optional
        Damping term (Default: 0)

    decay : float, optional
        Decay of the learning rate (Default: 0)

    """

    xk = yield
    rms = np.zeros_like(xk)
    k = 0.

    while True:

        k += 1.

        # compute objective and gradient
        gradient = yield xk

        # update RMS
        rms = decay * rms + (1-decay) * gradient**2

        # gradient descent update
        xk -= lr * gradient / (damping + np.sqrt(rms))


@coroutine
def sag(nterms=10, lr=1e-3):
    """
    Stochastic Average Gradient (SAG)

    Parameters
    ----------
    theta_init : array_like
        Initial parameters

    nterms : int, optional
        Number of gradient evaluations to use in the average (Default: 10)

    lr : float, optional
        (Default: 1e-3)

    """

    xk = yield
    gradients = deque([], nterms)
    k = 0.

    while True:

        k += 1.

        # compute the objective and gradient
        gradient = yield xk

        # push the new gradient onto the deque, update the average
        gradients.append(grad)

        # update
        xk -= lr * np.mean(gradients, axis=0)


@coroutine
def adam(lr=1e-3, beta=(0.9, 0.999), epsilon=1e-8):

    xk = yield
    momentum = np.zeros_like(xk)
    velocity = np.zeros_like(xk)
    b1, b2 = beta

    # current iteration
    k = 0

    while True:

        # update the iteration
        k += 1

        # send in the gradient
        gradient = yield xk

        # update momentum
        momentum = b1 * momentum + (1 - b1) * grad

        # update velocity
        velocity = b2 * velocity + (1 - b2) * (gradient ** 2)

        # normalize
        momentum_normalized = momentum / (1 - b1 ** k)
        velocity_normalized = np.sqrt(velocity / (1 - b2 ** k))

        # gradient descent update
        xk -= lr * momentum_normalized / (epsilon + velocity_normalized)


@coroutine
def pgd(proxop, lr=1e-3):
    """
    Proximal gradient descent

    Parameters
    ----------
    x0 : array_like
        Initial parameters

    proxop : ProximalOperator
        (e.g. from the proximal_operators module)

    lr : float, optional
        (default: 0.001)

    """

    xk = yield
    k = 0.

    while True:

        k += 1.
        gradient = yield xk
        xk = proxop(xk - lr * grad, 1. / lr)


@coroutine
def apg(proxop, lr=1e-3):
    """
    Accelerated Proximal Gradient

    Parameters
    ----------
    x0 : array_like
        Initial parameters

    proxop : ProximalOperator
        (e.g. from the proximal_operators module)

    lr : float, optional
        (default: 0.001)

    """

    xk = yield
    xprev = xk.copy()
    yk = xk.copy()
    k = 0.

    while True:

        k += 1.

        omega = k / (k + 3.)

        # update y's
        yk = xk + omega * (xk - xprev)

        # compute the gradient
        gradient = yield yk

        # update previous
        xprev = xk

        # compute the new iterate
        xk = proxop(yk - lr * gradient, 1. / lr)
