"""
First order gradient descent algorithms
"""

from __future__ import division
from .main import Optimizer
from .utils import destruct, wrap
import numpy as np
from collections import deque, namedtuple
from builtins import super

__all__ = ['GradientDescent', 'NesterovAcceleratedGradient', 'RMSProp', 'Adam',
           'StochasticAverageGradient']

class GradientDescent(Optimizer):
    """
    Gradient Descent (with momentum)

    Parameters
    ----------
    f_df : function
        Objective and gradient function

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

    def __init__(self, f_df, theta_init, learning_rate=1e-3, momentum=0., decay=0.):
        self.lr = learning_rate
        self.momentum = momentum
        self.decay = decay

        # initializes objective and gradient
        self.obj, self.gradient = wrap(f_df, theta_init)
        super().__init__(theta_init)

    def __iter__(self):
        """
        Initialize the generator
        """

        xk = destruct(self.theta).copy().astype('float')
        vk = np.zeros_like(xk)

        for k in range(self.maxiter):
            with self as state:

                # compute gradient
                grad = state.gradient(xk)

                # update velocity
                vnext = state.momentum * vk - state.lr * grad / (state.decay * k + 1.)

                # update parameters
                xk += vnext
                vk = vnext

                yield xk


class NesterovAcceleratedGradient(Optimizer):
    """
    Nesterov's Accelerated Gradient Descent

    Parameters
    ----------
    f_df : function
        Objective and gradient function

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

    def __init__(self, f_df, theta_init, learning_rate=1e-3):

        # store the learning rate
        self.lr = learning_rate

        # initializes objective and gradient
        self.obj, self.gradient = wrap(f_df, theta_init)
        super().__init__(theta_init)

    def __iter__(self):
        """
        Initialize the generator
        """

        xk = destruct(self.theta).copy().astype('float')
        yk = xk.copy()

        for k in range(self.maxiter):
            with self as state:

                # compute gradient
                grad = state.gradient(yk)

                # compute the new value of x
                xnext = yk - state.lr * grad

                # compute the new value of y
                ynext = xnext + (k / (k + 1)) * (xnext - xk)

                # update parameters
                xk = xnext
                yk = ynext

                yield xk


class RMSProp(Optimizer):
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

    def __init__(self, f_df, theta_init, learning_rate=1e-3, damping=0.1, decay=0.9):

        self.lr = learning_rate
        self.damping = damping
        self.decay = decay

        # initializes objective and gradient
        self.obj, self.gradient = wrap(f_df, theta_init)
        super().__init__(theta_init)

    def __iter__(self):

        xk = destruct(self.theta).copy().astype('float')
        rms = np.zeros_like(xk)

        for k in range(self.maxiter):
            with self as state:

                # compute objective and gradient
                grad = state.gradient(xk)

                # update RMS
                rms = state.decay * rms + (1-state.decay) * grad**2

                # gradient descent update
                xk -= state.lr * grad / (state.damping + np.sqrt(rms))

                yield xk


class StochasticAverageGradient(Optimizer):
    """
    Stochastic Average Gradient (SAG)

    Parameters
    ----------
    f_df : function
        Objective & gradient

    theta_init : array_like
        Initial parameters

    nterms : int
        Number of gradient evaluations to use in the average

    learning_rate : float, optional
        (Default: 1e-3)

    """

    def __init__(self, f_df, theta_init, nterms, learning_rate=1e-3):
        self.lr = learning_rate
        self.nterms = nterms

        # initializes objective and gradient
        self.obj, self.gradient = wrap(f_df, theta_init)
        super().__init__(theta_init)

    def __iter__(self):

        # initialize parameters
        xk = destruct(self.theta).copy().astype('float')

        # initialize gradients
        gradients = deque([], self.nterms)

        for k in range(self.maxiter):
            with self as state:

                # compute the objective and gradient
                grad = state.gradient(xk)

                # push the new gradient onto the deque, update the average
                gradients.append(grad)

                # update
                xk -= state.lr * np.mean(gradients, axis=0)

                yield xk


class Adam(Optimizer):
    """
    ADAM

    See: http://arxiv.org/abs/1412.6980

    Parameters
    ----------
    f_df : function
        Objective and gradient function

    theta_init : array_like
        Initial parameters

    learning_rate : float, optional
        Learning rate (Default: 1e-3)

    beta : (b1, b2), optional
        Exponential decay rates for the moment estimates

    epsilon : float, optional
        Damping factor

    """

    def __init__(self, f_df, theta_init, learning_rate=1e-3, beta=(0.9, 0.999), epsilon=1e-8):

        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon

        # initializes objective and gradient
        self.obj, self.gradient = wrap(f_df, theta_init)
        super().__init__(theta_init)

    def __iter__(self):

        # initialize parameters and velocity
        xk = destruct(self.theta).copy().astype('float')
        momentum = np.zeros_like(xk)
        velocity = np.zeros_like(xk)
        b1, b2 = self.beta

        for k in range(self.maxiter):
            with self as state:

                # compute objective and gradient
                grad = state.gradient(xk)

                # update momentum
                momentum = b1 * momentum + (1 - b1) * grad

                # update velocity
                velocity = b2 * velocity + (1 - b2) * (grad ** 2)

                # normalize
                momentum_normalized = momentum / (1 - b1 ** (k + 1))
                velocity_normalized = np.sqrt(velocity / (1 - b2 ** (k + 1)))

                # gradient descent update
                xk -= state.lr * momentum_normalized / (state.epsilon + velocity_normalized)

                yield xk
