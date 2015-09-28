"""
playing around with proximal operators
"""
import numpy as np

class Operator:

    def __call__(v, rho):
        raise NotImplementedError


class nucnorm(Operator):

    def __init__(self, penalty):
        self.penalty = penalty

    def __call__(self, v, rho):
        u, s, v = np.linalg.svd(v, full_matrices=False)
        sthr = np.maximum(s - (self.penalty / float(rho)), 0)
        return np.linalg.multi_dot((u, np.diag(sthr), v))


class sparse(Operator):

    def __init__(self, penalty):
        """
        Soft thresholding
        """
        self.penalty = penalty

    def __call__(self, v, rho):
        lmbda = float(self.penalty) / rho
        return (v - lmbda) * (v >= lmbda) + (v + lmbda) * (v <= -lmbda)


class nonneg(Operator):
    def __init__(self):
        pass

    def __call__(self, v, rho):
        return np.maximum(v, 0)


class linsys(Operator):

    def __init__(self, A, b):
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
