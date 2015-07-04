"""
First order gradient descent algorithms
"""


def gd(df, x0, eta=0.1):
    """Gradient descent"""

    xk = x0
    while True:
        xk -= eta * df(xk)
        yield xk


def agd(df, x0, eta=0.1, gamma=0.1):
    """Accelerated gradient descent"""

    xk = x0
    vk = 0
    while True:

        vnext = xk - eta * df(xk)
        xk = (1-gamma) * vnext + gamma * vk
        vnext = vk

        yield xk


def loop(algorithm, df, x0, maxiter=10000, **kwargs):
    """Main loop"""

    opt = algorithm(df, x0, **kwargs)

    xk = []
    for k in range(int(maxiter)):
        xk.append(next(opt))

    return xk
