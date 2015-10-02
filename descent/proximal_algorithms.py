"""
Proximal algorithms
"""

from .main import Optimizer
from .utils import destruct, wrap
from .proxops import getprox, ProximalOperator
import numpy as np
from collections import deque, namedtuple
from builtins import super

__all__ = ['ProximalGradientDescent', 'ADMM', 'pgd', 'admm']


class ProximalGradientDescent(Optimizer):

    def __init__(self, f_df, proxop, rho, theta_init, learning_rate=1e-3):

        # assert isinstance(proxop, ProximalOperator), "Must be a proximal operator!"

        self.proxop = proxop
        self.rho = rho
        self.lr = learning_rate

        # initializes objective and gradient
        self.obj, self.gradient = wrap(f_df, theta_init)
        super().__init__(theta_init)

    def __iter__(self):

        xk = self.theta.copy().astype('float')

        for k in range(self.maxiter):
            with self as state:
                grad = state.gradient(xk)
                xk = state.proxop(xk - state.lr * grad, state.rho)
                yield xk


class ADMM(Optimizer):

    def __init__(self, theta_init, tau=(10., 2., 2.), tol=1e-6):
        """
        Consensus ADMM
        """

        # assert isinstance(objective, ProximalOperator), "Must be a proximal operator!"

        self.operators = []
        self.tau = namedtuple('tau', ('init', 'inc', 'dec'))(*tau)
        self.gradient = None
        super().__init__(theta_init)

    def obj(self, x):
        # return np.sum([f.obj(x) for f in self.operators])
        return 0.

    def add(self, name, *args, **kwargs):
        self.operators.append(getprox(name, *args, **kwargs))

    def __iter__(self):

        num_obj = len(self.operators)
        assert num_obj >= 1, "Must be at least one objective"

        # initialize
        primals = [self.theta.flatten() for _ in range(num_obj)]
        duals = [np.zeros(self.theta.size) for _ in range(num_obj)]
        theta_avg = np.mean(primals, axis=0).ravel()
        rho = self.tau.init

        for k in range(self.maxiter):
            with self as state:

                # store the parameters from the previous iteration
                theta_prev = theta_avg

                # update each primal variable copy by taking a proximal step via each objective
                for varidx, dual in enumerate(duals):
                    primals[varidx] = self.operators[varidx](self.restruct(theta_prev-dual), rho).ravel()

                # average primal copies
                theta_avg = np.mean(primals, axis=0)

                # update the dual variables (after primal update has finished)
                for varidx, primal in enumerate(primals):
                    duals[varidx] += primal - theta_avg

                # compute primal and dual residuals
                primal_resid = float(np.sum([np.linalg.norm(primal - theta_avg) for primal in primals]))
                dual_resid = num_obj * rho ** 2 * np.linalg.norm(theta_avg - theta_prev)

                # update penalty parameter according to primal and dual residuals
                # (see sect. 3.4.1 of the Boyd and Parikh ADMM paper)
                if primal_resid > self.tau.init * dual_resid:
                    rho *= float(self.tau.inc)
                elif dual_resid > self.tau.init * primal_resid:
                    rho /= float(self.tau.dec)

                # check for convergence
                tol = 0
                if (primal_resid <= tol) & (dual_resid <= tol):
                    self.converged = True
                    break

                yield theta_avg


# aliases
pgd = ProximalGradientDescent
admm = ADMM
