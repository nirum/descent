"""
First order gradient descent algorithms
"""

from toolz.curried import curry


@curry
def gdm(df, x0, maxiter, eta=1e-3, mu=0.0):
    """
    Gradient descent with momentum

    Parameters
    ----------
    df : function

    x0 : array_like

    eta : float, optional
        Learning rate (Default: 0.01)

    mu : float, optional
        Momentum (Default: 0)

    """

    xk = x0.copy()
    vk = 0
    for k in range(maxiter):

        vnext = mu * vk - eta * df(xk)
        xk += vnext
        vk = vnext

        yield xk
