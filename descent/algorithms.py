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

__all__ = ['adam']


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
