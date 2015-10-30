"""
Proximal operators / mappings

"""

from __future__ import division
import numpy as np
from collections import Callable
from functools import partial

try:
    from scipy.optimize import minimize as scipy_minimize
    from scipy.sparse import spdiags
    from scipy.sparse.linalg import spsolve
except ImportError: # pragma no cover
    print("Package 'scipy' not found. L-BFGS and smooth proximal operators will not work.")

__all__ = ['nucnorm', 'sparse', 'nonneg', 'linsys', 'squared_error', 'columns',
           'lbfgs', 'tvd', 'smooth', 'linear', 'fantope', 'ProximalOperator']


def _getproxop(name, *args, **kwargs):
    """
    Loads a proximal operator

    Parameters
    ----------

    name : string or callable
        If string, then it must be the name of one of the functions in the
        proximal_operators module. If callable, it should subclass the
        ProximalOperator class and have a __call__ function that takes as input
        a set of parameters and a parameter rho, and returns the output of a
        proximal map.

    *args, **kwargs : optional arguments that are bound to the proximal map

    """

    if isinstance(name, Callable):
        return partial(name, *args, **kwargs)

    elif type(name) is str:
        assert name in __all__, name + " is not a valid operator!"
        return globals()[name](*args, **kwargs)

    else:
        raise ValueError("First argument must be a string or callable. (see documentation)")


class ProximalOperator:
    """
    Superclass for all proximal operators.
    """

    def __call__(self, x0, rho):
        raise NotImplementedError

    def objective(self, theta):
        return np.nan

    def apply(self, x0, rho):
        self.__call__(x0, rho)


class nucnorm(ProximalOperator):

    def __init__(self, penalty, newshape=None):
        """
        Proximal operator for the nuclear norm penalty

        Parameters
        ----------
        penalty : float
            Weight on the nuclear norm (lambda)

        """
        self.penalty = penalty
        self.newshape = newshape

    def __call__(self, x0, rho):
        """
        Applies the nuclear norm proximal operator

        Parameters
        ----------
        x0 : array_like
            Initial parameters (matrix)

        rho : float
            Quadratic penalty weight

        """
        orig_shape = x0.shape

        if self.newshape is not None:
            x0 = x0.reshape(self.newshape)

        u, s, v = np.linalg.svd(x0, full_matrices=False)
        sthr = np.maximum(s - (self.penalty / float(rho)), 0)
        # return np.linalg.multi_dot((u, np.diag(sthr), v))
        return (u.dot(np.diag(sthr)).dot(v)).reshape(orig_shape)

    def objective(self, theta):
        singular_values = np.linalg.svd(theta, full_matrices=False, compute_uv=False)
        return np.sum(singular_values)


class columns(ProximalOperator):
    """
    Applies a proximal operator to the columns of a matrix

    """

    def __init__(self, proxop):

        self.proxop = proxop

    def __call__(self, x0, rho):

        xnext = np.zeros_like(x0)

        for ix in range(x0.shape[1]):
            xnext[:,ix] = self.proxop(x0[:,ix], rho)

        return xnext


class sparse(ProximalOperator):

    def __init__(self, penalty):
        """
        Proximal operator for the l1-norm: soft thresholding

        Parameters
        ----------
        penalty : float
            Strength or weight on the l1-norm

        """
        self.penalty = penalty

    def __call__(self, v, rho):
        """
        Applies the sparsifying (l1-norm) proximal operator

        Parameters
        ----------
        x0 : array_like
            Initial parameters

        rho : float
            Quadratic penalty weight

        """
        lmbda = float(self.penalty) / rho
        return (v - lmbda) * (v >= lmbda) + (v + lmbda) * (v <= -lmbda)

    def objective(self, theta):
        return np.linalg.norm(theta.ravel(), 1)


class nonneg(ProximalOperator):

    def __init__(self):
        """
        Proximal operator for the indicator function over the
        non-negative orthant (projection onto non-negative orthant)
        """
        pass

    def __call__(self, v, rho):
        """
        Projection onto the non-negative orthant

        Parameters
        ----------
        x0 : array_like
            Initial parameters

        rho : float
            Quadratic penalty weight (unused)

        """
        return np.maximum(v, 0)

    def objective(self, theta):
        if np.all(theta >= 0):
            return 0
        else:
            return np.Inf


class linsys(ProximalOperator):

    def __init__(self, A, b):
        """
        Proximal operator for solving a linear least squares system, Ax = b

        Parameters
        ----------
        A : array_like
            Sensing matrix (Ax = b)

        b : array_like
            Responses (Ax = b)

        """
        self.A = A
        self.b = b
        self.P = A.T.dot(A)
        self.q = A.T.dot(b)

    def __call__(self, v, rho):
        """
        Proximal operator for the linear approximation Ax = b

        Minimizes the function:

        .. math:: f(x) = (1/2)||Ax-b||_2^2

        Parameters
        ----------
        x0 : array_like
            The starting or initial point used in the proximal update step

        rho : float
            Momentum parameter for the proximal step (larger value -> stays closer to x0)

        """
        return np.linalg.solve(rho * np.eye(self.q.size) + self.P, rho * v + self.q)

    def objective(self, theta):
        return 0.5 * np.linalg.norm(self.A.dot(theta) - self.b, 2) ** 2


class squared_error(ProximalOperator):

    def __init__(self, x_obs):
        """
        Proximal operator for squared error (l2 or Fro. norm)

        Parameters
        ----------
        x_obs : array_like
            'Observed' array or matrix that you want to stay close to

        """
        self.x_obs = x_obs.copy()

    def __call__(self, x0, rho):
        """
        Proximal operator for the sum of squared differences between two matrices

        Parameters
        ----------
        x0 : array_like
            The starting or initial point used in the proximal update step

        rho : float
            Momentum parameter for the proximal step (larger value -> stays closer to x0)

        """
        return (x0 + self.x_obs / rho) / (1 + 1 / rho)

    def objective(self, theta):
        return np.linalg.norm(self.x_obs.ravel() - theta.ravel(), 2)


class lbfgs(ProximalOperator):

    def __init__(self, f_df, numiter=20.):
        self.f_df = f_df
        self.numiter = numiter

    def f_df_augmented(self, theta):
        f, df = self.f_df(theta)
        obj = f + (self.rho / 2.) * np.linalg.norm(theta.ravel() - self.v.ravel()) ** 2
        grad = df + self.rho * (theta - self.v)
        return obj, grad

    def __call__(self, x0, rho):

        self.v = np.array(x0).copy()
        self.rho = rho

        res = scipy_minimize(self.f_df_augmented, x0, jac=True, method='L-BFGS-B',
                             options={'maxiter': self.numiter, 'disp': False})

        return res.x

    def objective(self, theta):
        return self.f_df(theta)[0]


class tvd(ProximalOperator):

    def __init__(self, penalty):
        """
        Total variation denoising proximal operator

        Parameters
        ----------
        penalty : float

        """
        self.penalty = penalty

    def __call__(self, x0, rho):
        """
        Proximal operator for the total variation denoising penalty

        Requires scikit-image be installed

        Parameters
        ----------
        x0 : array_like
            The starting or initial point used in the proximal update step

        rho : float
            Momentum parameter for the proximal step (larger value -> stays closer to x0)

        Raises
        ------
        ImportError
            If scikit-image fails to be imported

        """
        try:
            from skimage.restoration import denoise_tv_bregman
        except ImportError:
            print('Error: scikit-image not found. TVD will not work.')
            return x0

        return denoise_tv_bregman(x0, rho / self.penalty)


class smooth(ProximalOperator):

    def __init__(self, axis, penalty):
        """
        Applies a smoothing operator along one dimension

        currently only accepts a matrix as input
        """
        self.axis = axis
        self.penalty = penalty


    def __call__(self, x0, rho):

        # Apply Laplacian smoothing (l2 norm on the parameters multiplied by
        # the laplacian)
        n = x0.shape[self.axis]
        lap_op = spdiags([(2 + rho / self.penalty) * np.ones(n), -1 * np.ones(n), -1 * np.ones(n)], [0, -1, 1], n, n, format='csc')

        x_out = np.rollaxis(spsolve(self.penalty * lap_op, rho * np.rollaxis(x0, self.axis, 0)), self.axis, 0)

        return x_out

    def objective(self, x):
        n = x.shape[self.axis]
        # lap_op = spdiags([(2 + rho / self.penalty) * np.ones(n), -1 * np.ones(n), -1 * np.ones(n)], [0, -1, 1], n, n, format='csc')
        # TODO: add objective for this operator
        return np.nan


class sdcone(ProximalOperator):
    """
    Projection onto the semi-definite cone

    """

    def __init__(self):
        """
        Projection onto the semidefinite cone
        """
        pass

    def __call__(self, x0, rho):

        U, V = np.linalg.eigh(x0)
        return V.dot(np.diag(np.maximum(U, 0)).dot(V.T))


class linear(ProximalOperator):
    """
    Proximal operator for a linear function

    """

    def __init__(self, weights):
        self.w = weights

    def __call__(self, x0, rho):
        return x0 - self.w / rho

    def objective(self, theta):
        return np.inner(theta.ravel(), self.w.ravel())


class fantope(ProximalOperator):
    """
    Projection onto the fantope

    TODO: add citation

    """

    def __init__(self, dim, tol=1e-4):
        self.d = dim
        self.tol = tol

    @staticmethod
    def threshold(u):
        return np.minimum(np.maximum(u, 0), 1)

    def bisect(self, u):

        # assert np.all(u >= 0)
        # u = np.minimum(u, 0)

        # print('==== sum(u) is {} ===='.format(u.sum()))

        minval, maxval = np.maximum(u.min(), 0), np.maximum(u.max(), 20 * self.d)

        theta = 0.5 * (maxval + minval)
        thr_eigvals = self.threshold(u - theta)
        constraint = np.sum(thr_eigvals)

        while True:
            # print('({}, {}) {}'.format(minval, maxval, constraint))
            # import time
            # time.sleep(0.1)

            if np.abs(constraint - self.d) <= self.tol:
                # print('Breaking!')
                break

            elif constraint < self.d:
                maxval = theta

            elif constraint > self.d:
                minval = theta

            else:
                break

            theta = 0.5 * (maxval + minval)
            thr_eigvals = self.threshold(u - theta)
            constraint = np.sum(thr_eigvals)

        return thr_eigvals

    def __call__(self, x0, rho):

        U, V = np.linalg.eigh(x0)

        thr_eigvals = self.bisect(U)

        nz_inds = np.where(thr_eigvals > 0)[0]

        # return np.linalg.multi_dot((u, np.diag(sthr), v))
        return V.dot(np.diag(thr_eigvals)).dot(V.T)
