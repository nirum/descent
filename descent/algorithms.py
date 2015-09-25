"""
First order gradient descent algorithms
"""

from toolz.curried import curry
import numpy as np

__all__ = ['gdm', 'rmsprop']


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
    xk = x0.copy().astype('float')
    vk = np.zeros_like(xk)

    for k in range(int(maxiter)):

        vnext = momentum * vk - lr * df(xk) / (decay * k + 1.0)
        xk += vnext
        vk = vnext

        yield xk


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
    xk = x0.copy().astype('float')
    rms = np.zeros_like(xk)

    for k in range(int(maxiter)):

        grad = df(xk)

        # update RMS
        rms = decay * rms + (1-decay) * grad**2

        # gradient descent update
        xk -= lr * grad / (damping + np.sqrt(rms))

        yield xk


@curry
def sag(df, x0, maxiter, nterms, lr=1e-2):
    """
    Stochastic Average Gradient (SAG)

    Parameters
    ----------
    df : function

    x0 : array_like

    maxiter : int

    lr : float, optional

    """

    # initialize gradients
    gradients = [df(x0, j) for j in range(nterms)]

    # initialize parameters
    xk = x0.copy().astype('float')

    for k in range(int(maxiter)):

        # choose a random subfunction to update
        idx = np.random.randint(nterms)
        gradients[idx] = df(xk, idx)

        # compute the average gradient
        grad = np.mean(gradients, axis=0)

        # update
        xk -= lr * grad

        yield xk

@curry
def adam(df, x0, maxiter, lr=1e-3, beta=(0.9, 0.999), epsilon=1e-8):
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

    # initialize parameters and velocity
    xk = x0.copy().astype('float')
    momentum = np.zeros_like(xk)
    velocity = np.zeros_like(xk)
    b1, b2 = beta

    for k in range(int(maxiter)):

        grad = df(xk)

        # update momentum
        momentum = b1 * momentum + (1 - b1) * grad

        # update velocity
        velocity = b2 * velocity + (1 - b2) * (grad ** 2)

        # normalize
        momentum_normalized = momentum / (1 - b1 ** (k+1))
        velocity_normalized = np.sqrt(velocity / (1 - b2 ** (k+1)))

        # gradient descent update
        xk -= lr * momentum_normalized / (epsilon + velocity_normalized)

        yield xk

