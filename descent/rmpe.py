"""
Regularized nonlinear acceleration
"""
import numpy as np

__all__ = ['rmpe']


def getU(xs):
    """xs is an (n x k) matrix"""
    return np.diff(xs, axis=1)


def rmpe(U, lmbda=1e-5):
    A = U.T @ U + lmbda * np.eye(U.shape[1],)
    z = np.linalg.solve(A, np.ones(U.shape[1],))
    return z / z.sum()


def xhat(xs):
    U = getU(xs)
    z = rmpe(U, lmbda=1e-5)
    return xs[:, :-1] @ z
