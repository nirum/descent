"""
First order gradient descent algorithms
"""

from .classify import GradientOptimizer, ProximalOptimizer
import numpy as np
from collections import deque

__all__ = ['GradientDescent', 'RMSProp', 'Adam', 'StochasticAverageGradient',
           'adam', 'rmsprop', 'sgd', 'sag']

class GradientDescent(GradientOptimizer):

    def __init__(self, f_df, theta_init, learning_rate=1e-3, momentum=0., decay=0.):
        self.lr = learning_rate
        self.momentum = momentum
        self.decay = decay
        super().__init__(f_df, theta_init)

    def __iter__(self):
        """
        Initialize the generator
        """

        xk = self.theta_init.copy().astype('float')
        vk = np.zeros_like(xk)

        for k in range(self.maxiter):
            with self as state:

                # compute objective and gradient
                obj = state.obj(xk)
                grad = state.gradient(xk)

                # update velocity
                vnext = state.momentum * vk - state.lr * grad / (state.decay * k + 1.)

                # update parameters
                xk += vnext
                vk = vnext

                yield obj, xk, grad


class RMSProp(GradientOptimizer):

    def __init__(self, f_df, theta_init, learning_rate=1e-2, damping=0.1, decay=0.9):
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

        self.lr = learning_rate
        self.damping = damping
        self.decay = decay
        super().__init__(f_df, theta_init)

    def __iter__(self):

        xk = self.theta_init.copy().astype('float')
        rms = np.zeros_like(xk)

        for k in range(self.maxiter):
            with self as state:

                # compute objective and gradient
                obj = state.obj(xk)
                grad = state.gradient(xk)

                # update RMS
                rms = state.decay * rms + (1-state.decay) * grad**2

                # gradient descent update
                xk -= state.lr * grad / (state.damping + np.sqrt(rms))

                yield obj, xk, grad


class StochasticAverageGradient(GradientOptimizer):
    def __init__(self, f_df, theta_init, nterms, learning_rate=1e-2):
        """
        Stochastic Average Gradient (SAG)

        Parameters
        ----------
        df : function

        x0 : array_like

        lr : float, optional

        """

        self.lr = learning_rate
        self.nterms = nterms
        super().__init__(f_df, theta_init)

    def __iter__(self):

        # initialize parameters
        xk = self.theta_init.copy().astype('float')

        # initialize gradients
        gradients = deque([], self.nterms)

        for k in range(self.maxiter):
            with self as state:

                # compute the objective and gradient
                obj = state.obj(xk)
                grad = state.gradient(xk)

                # push the new gradient onto the deque, update the average
                gradients.append(grad)
                avg_grad = np.mean(gradients, axis=0)

                # update
                xk -= state.lr * avg_grad

                yield obj, xk, avg_grad


class Adam(GradientOptimizer):

    def __init__(self, f_df, theta_init, learning_rate=1e-3, beta=(0.9, 0.999), epsilon=1e-8):
        """
        ADAM

        See: http://arxiv.org/abs/1412.6980

        Parameters
        ----------
        df : function

        x0 : array_like

        maxiter : int

        lr : float, optional
            Learning rate (Default: 1e-2)

        beta : (b1, b2), optional
            Exponential decay rates for the moment estimates

        epsilon : float, optional
            Damping factor

        """

        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        super().__init__(f_df, theta_init)

    def __iter__(self):

        # initialize parameters and velocity
        xk = self.theta_init.copy().astype('float')
        momentum = np.zeros_like(xk)
        velocity = np.zeros_like(xk)
        b1, b2 = self.beta

        for k in range(self.maxiter):
            with self as state:

                # compute objective and gradient
                obj = state.obj(xk)
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

                yield obj, xk, grad

# aliases
sgd = GradientDescent
adam = Adam
sag = StochasticAverageGradient
rmsprop = RMSProp

class ProximalDescent(ProximalOptimizer):

    def __init__(self, proxop, rho, gradient, lr=1e-3):
        self.proxop = proxop
        self.rho = rho
        self.lr = lr
        self.df = gradient

    def __iter__(self):

        xk = self.xinit.copy().astype('float')

        for k in range(self.maxiter):
            with self as state:
                obj, grad = state.f_df(state.xk)
                xk = state.proxop(xk - state.lr * grad, state.rho)
                yield obj, xk


class ADMM(object):

    def __init__(self, f, g, rho, xinit):
        self.f = f
        self.g = g
        self.rho = rho
        self.xinit = xinit

    def __iter__(self):
        """
        Initialize the generator
        """

        self.xk = self.xinit.copy().astype('float')
        self.zk = self.xinit.copy().astype('float')
        self.uk = np.zeros(self.xk.size)
        self.k = 0

        return self

    def __next__(self):

        with self as state:
            state.xk = state.f(state.zk - state.uk, state.rho)
            state.zk = state.g(state.xk + state.uk, state.rho)
            state.uk += state.xk - state.zk

        return self.xk
