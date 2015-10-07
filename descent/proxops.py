"""
playing around with proximal operators
"""
import numpy as np
from collections import Callable
from functools import partial

try:
    from scipy.optimize import minimize as scipy_minimize
    from scipy.sparse import spdiags
    from scipy.sparse.linalg import spsolve
except ImportError:
    print('Scipy not found. L-BFGS not available as a proximal operator.')

__all__ = ['getprox', 'nucnorm', 'sparse', 'nonneg', 'linsys', 'squared_error',
           'lbfgs', 'tvd', 'smooth', 'ProximalOperator']


def getprox(name, *args, **kwargs):
    """
    Loads a proximal operator
    """

    if isinstance(name, Callable):
        return partial(name, *args, **kwargs)

    elif type(name) is str:
        assert name in __all__, name + " is not a valid operator!"
        return globals()[name](*args, **kwargs)

    else:
        raise ValueError("First argument must be a string or callable (see docs for more info)")


class ProximalOperator:

    def __call__(v, rho):
        raise NotImplementedError

    def objective(theta):
        raise NotImplementedError

    def apply(self, v, rho):
        self.__call__(v, rho)


class nucnorm(ProximalOperator):

    def __init__(self, penalty):
        self.penalty = penalty

    def __call__(self, x0, rho):
        u, s, v = np.linalg.svd(x0, full_matrices=False)
        sthr = np.maximum(s - (self.penalty / float(rho)), 0)
        # return np.linalg.multi_dot((u, np.diag(sthr), v))
        return u.dot(np.diag(sthr)).dot(v)

    def objective(self, theta):
        singular_values = np.linalg.svd(theta, full_matrices=False, compute_uv=False)
        return np.sum(singular_values)


class sparse(ProximalOperator):

    def __init__(self, penalty):
        """
        Soft thresholding
        """
        self.penalty = penalty

    def __call__(self, v, rho):
        lmbda = float(self.penalty) / rho
        return (v - lmbda) * (v >= lmbda) + (v + lmbda) * (v <= -lmbda)

    def objective(self, theta):
        return np.linalg.norm(theta.ravel(), 1)


class nonneg(ProximalOperator):
    def __init__(self):
        pass

    def __call__(self, v, rho):
        return np.maximum(v, 0)

    def objective(self, theta):
        if np.all(theta >= 0):
            return 0
        else:
            return np.Inf


class linsys(ProximalOperator):

    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.P = A.T.dot(A)
        self.q = A.T.dot(b)

    def __call__(self, v, rho):
        """
        Proximal operator for the linear approximation Ax = b

        Minimizes the function:

        .. math:: f(x) = (1/2)||Ax-b||_2^2 = (1/2)x^TA^TAx - (b^TA)x + b^Tb

        Parameters
        ----------
        x0 : array_like
            The starting or initial point used in the proximal update step

        rho : float
            Momentum parameter for the proximal step (larger value -> stays closer to x0)

        P : array_like
            The symmetric matrix A^TA, where we are trying to approximate Ax=b

        q : array_like
            The vector A^Tb, where we are trying to approximate Ax=b

        Returns
        -------
        theta : array_like
            The parameter vector found after running the proximal update step

        """
        return np.linalg.solve(rho * np.eye(self.q.size) + self.P, rho * v + self.q)

    def objective(self, theta):
        return np.linalg.norm(A.dot(theta) - b, 2)


class squared_error(ProximalOperator):

    def __init__(self, x_obs):
        self.x_obs = x_obs.copy()

    def __call__(self, x0, rho):
        """
        Proximal operator for the pairwise difference between two matrices (Frobenius norm)

        Parameters
        ----------
        x0 : array_like
            The starting or initial point used in the proximal update step

        rho : float
            Momentum parameter for the proximal step (larger value -> stays closer to x0)

        x_obs : array_like
            The true matrix that we want to approximate. The error between the parameters and this matrix is minimized.

        Returns
        -------
        x0 : array_like
            The parameter vector found after running the proximal update step

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

        self.v = x0.copy()
        self.rho = rho

        res = scipy_minimize(self.f_df_augmented, x0, jac=True, method='L-BFGS-B',
                             options={'maxiter': self.numiter, 'disp': False})

        return res.x

    def objective(self, theta):
        return self.f_df(theta)[0]


class tvd(ProximalOperator):

    def __init__(self, gamma):
        self.gamma = gamma

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

        gamma : float
            A constant that weights how strongly to enforce the constraint

        Returns
        -------
        theta : array_like
            The parameter vector found after running the proximal update step

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

        return denoise_tv_bregman(x0, rho / self.gamma)


class smooth(ProximalOperator):

    def __init__(self, axis, gamma):
        """
        Applies a smoothing operator along one dimension

        currently only accepts a matrix as input
        """
        self.axis = axis
        self.gamma = gamma


    def __call__(self, x0, rho):

        # Apply Laplacian smoothing (l2 norm on the parameters multiplied by
        # the laplacian)
        n = x0.shape[self.axis]
        lap_op = spdiags([(2 + rho / self.gamma) * np.ones(n), -1 * np.ones(n), -1 * np.ones(n)], [0, -1, 1], n, n, format='csc')

        x_out = np.rollaxis(spsolve(self.gamma * lap_op, rho * np.rollaxis(x0, self.axis, 0)), self.axis, 0)

        return x_out

    def objective(self, x):
        n = x.shape[self.axis]
        # lap_op = spdiags([(2 + rho / self.gamma) * np.ones(n), -1 * np.ones(n), -1 * np.ones(n)], [0, -1, 1], n, n, format='csc')
        # TODO: add objective for this operator
        return 0
