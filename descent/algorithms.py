"""
First order gradient descent algorithms

Each algorithm in this module is a coroutine. You send in the gradient into
the coroutine, and it spits out the next iterate in the descent algorithm.

"""

from __future__ import division
from .utils import destruct, wrap
import numpy as np
from collections import deque, namedtuple
from builtins import super

__all__ = ['sgd', 'nag', 'rmsprop', 'sag', 'adam']


def sgd(x0, learning_rate=1e-3, momentum=0., decay=0.):
    """
    Stochastic Gradient Descent (with momentum)

    Parameters
    ----------
    theta_init : array_like
        Initial parameters

    learning_rate : float, optional
        Learning rate (Default: 1e-3)

    momentum : float, optional
        Momentum parameter (Default: 0)

    decay : float, optional
        Decay of the learning rate. Every iteration the learning rate decays
        by a factor of 1/(decay+1), (Default: 0)

    """

    xk = x0.copy()
    vk = np.zeros_like(xk)
    k = 0.

    while True:

        k += 1.

        # compute gradient
        grad = yield xk

        # update velocity
        vnext = momentum * vk - learning_rate * grad / (decay * k + 1.)

        # update parameters
        xk += vnext
        vk = vnext


def nag(x0, learning_rate=1e-3):
    """
    Nesterov's Accelerated Gradient Descent

    Parameters
    ----------
    theta_init : array_like
        Initial parameters

    learning_rate : float, optional
        Learning rate (Default: 1e-3)

    """

    xk = x0.copy()
    yk = x0.copy()
    k = 0.

    while True:

        k += 1.

        # compute gradient
        grad = yield yk

        # compute the new value of x
        xnext = yk - learning_rate * grad

        # compute the new value of y
        ynext = xnext + (k / (k + 1)) * (xnext - xk)

        # update parameters
        xk = xnext
        yk = ynext


def rmsprop(x0, learning_rate=1e-3, damping=0.1, decay=0.9):
    """
    RMSProp

    Parameters
    ----------
    f_df : function

    theta_init : array_like

    learning_rate : float, optional
        Learning rate (Default: 1e-3)

    damping : float, optional
        Damping term (Default: 0)

    decay : float, optional
        Decay of the learning rate (Default: 0)

    """

    xk = x0.copy()
    rms = np.zeros_like(xk)
    k = 0.

    while True:

        k += 1.

        # compute objective and gradient
        grad = yield xk

        # update RMS
        rms = decay * rms + (1-decay) * grad**2

        # gradient descent update
        xk -= learning_rate * grad / (damping + np.sqrt(rms))


def sag(x0, nterms=10, learning_rate=1e-3):
    """
    Stochastic Average Gradient (SAG)

    Parameters
    ----------
    theta_init : array_like
        Initial parameters

    nterms : int, optional
        Number of gradient evaluations to use in the average (Default: 10)

    learning_rate : float, optional
        (Default: 1e-3)

    """

    xk = x0.copy()
    gradients = deque([], nterms)
    k = 0.

    while True:

        k += 1.

        # compute the objective and gradient
        grad = yield xk

        # push the new gradient onto the deque, update the average
        gradients.append(grad)

        # update
        xk -= learning_rate * np.mean(gradients, axis=0)


def adam(x0, learning_rate=1e-3, beta=(0.9, 0.999), epsilon=1e-8):

    xk = x0.copy()
    momentum = np.zeros_like(xk)
    velocity = np.zeros_like(xk)
    b1, b2 = beta

    # current iteration
    k = 0

    while True:

        # update the iteration
        k += 1

        # send in the gradient
        grad = yield xk

        # update momentum
        momentum = b1 * momentum + (1 - b1) * grad

        # update velocity
        velocity = b2 * velocity + (1 - b2) * (grad ** 2)

        # normalize
        momentum_normalized = momentum / (1 - b1 ** k)
        velocity_normalized = np.sqrt(velocity / (1 - b2 ** k))

        # gradient descent update
        xk -= learning_rate * momentum_normalized / (epsilon + velocity_normalized)


def pgd(x0, proxop, learning_rate=1e-3):
    """
    Proximal gradient descent

    Parameters
    ----------
    x0 : array_like
        Initial parameters

    proxop : ProximalOperator
        (e.g. from the proximal_operators module)

    learning_rate : float, optional
        (default: 0.001)

    """

    xk = x0.copy()
    k = 0.

    while True:

        k += 1.
        grad = yield xk
        xk = proxop(xk - learning_rate * grad, 1. / learning_rate)


def apg(x0, proxop, learning_rate=1e-3):
    """
    Accelerated Proximal Gradient

    Parameters
    ----------
    x0 : array_like
        Initial parameters

    proxop : ProximalOperator
        (e.g. from the proximal_operators module)

    learning_rate : float, optional
        (default: 0.001)

    """

    xk = x0.copy()
    xprev = x0.copy()
    yk = x0.copy()
    k = 0.

    while True:

        k += 1.

        omega = k / (k + 3.)

        # update y's
        yk = xk + omega * (xk - xprev)

        # compute the gradient
        grad = yield yk

        # update previous
        xprev = xk

        # compute the new iterate
        xk = proxop(yk - learning_rate * grad, 1. / learning_rate)
