"""
First order gradient descent algorithms
"""

from toolz.curried import curry


@curry
def gd(df, x0, eta=0.1):
    """Gradient descent"""

    xk = x0
    while True:
        xk -= eta * df(xk)
        yield xk


@curry
def agd(df, x0, eta=0.1, gamma=0.1):
    """Accelerated gradient descent"""

    xk = x0
    vk = 0
    while True:

        vnext = xk - eta * df(xk)
        xk = (1-gamma) * vnext + gamma * vk
        vk = vnext

        yield xk
