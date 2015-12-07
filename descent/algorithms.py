"""
First order gradient descent algorithms

"""

from __future__ import division
import numpy as np
from collections import deque
from abc import ABCMeta, abstractmethod

__all__ = ['sgd', 'StochasticGradientDescent',
           'nag', 'NesterovAcceleratedGradient',
           'rmsprop', 'RMSProp',
           'sag', 'StochasticAverageGradient',
           'adam', 'ADAM']


class Algorithm(metaclass=ABCMeta):

    def __init__(self, xinit):
        self.k = 0.
        self.xk = xinit.copy()

    def __next__(self):
        """Called to update every iteration"""
        self.k += 1.0

    @abstractmethod
    def __call__(self, gradient):
        raise NotImplementedError


class StochasticGradientDescent(Algorithm):

    def __init__(self, xinit, lr=1e-3, momentum=0., decay=0.):

        super().__init__(xinit)
        self.vk = np.zeros_like(xinit)
        self.lr = lr
        self.momentum = momentum
        self.decay = decay

    def __call__(self, gradient):

        # increase the iteration
        super().__next__()

        # velocity
        self.vk = self.momentum * self.vk - self.lr * gradient \
            / (self.decay * self.k + 1.)

        # updated parameters
        self.xk += self.vk

        return self.xk


class NesterovAcceleratedGradient(Algorithm):

    def __init__(self, xinit, lr=1e-3):
        """
        Nesterov's Accelerated Gradient Descent

        Parameters
        ----------
        lr : float, optional
            Learning rate (Default: 1e-3)

        """

        super().__init__(xinit)
        self.yk = self.xk.copy()
        self.lr = lr

    def __call__(self, gradient):

        # increase the iteration
        super().__next__()

        xprev = self.xk.copy()
        self.xk = self.yk - self.lr * gradient
        self.yk = self.xk + (self.k / (self.k + 3.)) * (self.xk - xprev)
        return self.yk


class RMSProp(Algorithm):

    def __init__(self, xinit, lr=1e-3, damping=0.1, decay=0.9):
        """
        RMSProp

        Parameters
        ----------
        theta_init : array_like

        lr : float, optional
            Learning rate (Default: 1e-3)

        damping : float, optional
            Damping term (Default: 0)

        decay : float, optional
            Decay of the learning rate (Default: 0)

        """

        super().__init__(xinit)
        self.rms = np.zeros_like(self.xk)
        self.lr = lr
        self.damping = damping
        self.decay = decay

    def __call__(self, gradient):

        super().__next__()

        # update RMS
        self.rms = self.decay * self.rms + (1 - self.decay) * gradient**2

        # gradient descent update
        self.xk -= self.lr * gradient / (self.damping + np.sqrt(self.rms))

        return self.xk


class StochasticAverageGradient(Algorithm):

    def __init__(self, xinit, nterms=10, lr=1e-3):
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

        super().__init__(xinit)
        self.gradients = deque([], nterms)
        self.lr = lr

    def __call__(self, gradient):

        super().__next__()

        # push the new gradient onto the deque, update the average
        self.gradients.append(gradient)

        # update
        self.xk -= self.lr * np.mean(self.gradients, axis=0)

        return self.xk


class ADAM(Algorithm):

    def __init__(self, xinit, lr=1e-3, beta=(0.9, 0.999), epsilon=1e-8):

        super().__init__(xinit)
        self.momentum = np.zeros_like(self.xk)
        self.velocity = np.zeros_like(self.xk)
        self.lr = lr
        self.b1, self.b2 = beta
        self.eps = epsilon

    def __call__(self, gradient):

        # update the iteration
        super().__next__()

        # update momentum
        self.momentum = self.b1 * self.momentum + (1. - self.b1) * gradient

        # update velocity
        self.velocity = self.b2 * self.velocity + (1 - self.b2) * (gradient ** 2)

        # normalize
        momentum_norm = self.momentum / (1 - self.b1 ** self.k)
        velocity_norm = np.sqrt(self.velocity / (1 - self.b2 ** self.k))

        # gradient descent update
        self.xk -= self.lr * momentum_norm / (self.eps + velocity_norm)

        return self.xk


# aliases
sgd = StochasticGradientDescent
nag = NesterovAcceleratedGradient
rmsprop = RMSProp
sag = StochasticAverageGradient
adam = ADAM
