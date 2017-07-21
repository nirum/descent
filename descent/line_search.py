"""
Line search methods
"""
import numpy as np


def backtracking(f_df, x0, dx, alpha=1., rho=0.5, c=1e-4):
    """
    Backtracking-armijo line search

    Params
    ------
    f_df : objective and gradient
    x0 : current iterate
    dx : proposed step
    alpha : initial step length
    rho : backtracking factor
    c : sufficient decrease condition constant

    Returns
    -------
    alpha : step size
    f : new objective value
    df : new gradient
    xk : new iterate
    """
    # current objective value
    f0 = f_df(x0)[0]

    # next iterate
    f, df = f_df(x0 + alpha * dx)

    while f > (f0 + c * alpha * np.inner(dx, df)):
        alpha = rho * alpha
        f, df = f_df(x0 + alpha * dx)

    return alpha
