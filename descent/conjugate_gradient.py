"""
Conjugate gradient methods
"""
import numpy as np


def cg(mvp, b, x0, maxiter=100, epsilon=1e-5):
    """
    Conjugate gradient method for solving Ax=b

    Parameters
    ----------
    mvp : function to compute matrix vector products
    x0 : initial guess
    """
    x = x0.copy()
    r = b.copy()
    threshold = epsilon * np.linalg.norm(b)

    rho_1 = np.linalg.norm(r) ** 2
    rho_2 = 1.
    p = 0.

    xs = [x.copy()]

    for k in range(maxiter):
        if np.sqrt(rho_1) <= threshold:
            break

        p = r + (rho_1 / rho_2) * p

        w = mvp(p)

        alpha = rho_1 / np.inner(p, w)

        x += alpha * p
        r -= alpha * w

        rho_2 = rho_1
        rho_1 = np.linalg.norm(r) ** 2

        xs.append(x.copy())

    return np.stack(xs)
