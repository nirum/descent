"""
Proximal algorithms
"""

from __future__ import division
from .main import Optimizer
from .utils import destruct, wrap
from .proximal_operators import _getproxop, ProximalOperator
import numpy as np
from collections import deque, namedtuple, defaultdict
from builtins import super

__all__ = ['ProximalGradientDescent', 'AcceleratedProximalGradient', 'ProximalConsensus']


class ProximalGradientDescent(Optimizer):
    """
    Proximal gradient descent

    Parameters
    ----------
    f_df : callable
        Function that returns the objective and gradient

    theta_init : array_like
        Initial parameters

    proxop : ProximalOperator
        (e.g. from the proximal_operators module)

    learning_rate : float, optional
        (default: 0.001)

    """

    def __init__(self, f_df, theta_init, proxop, learning_rate=1e-3):
        self.proxop = proxop
        self.lr = learning_rate

        # initializes objective and gradient
        self.obj, self.gradient = wrap(f_df, theta_init)
        super().__init__(theta_init)

    def __iter__(self):

        xk = self.theta.copy().astype('float')

        for k in range(self.maxiter):
            with self as state:
                grad = self.restruct(state.gradient(xk))
                xk = state.proxop(xk - state.lr * grad, 1. / state.lr)
                yield xk


class AcceleratedProximalGradient(Optimizer):
    """
    Accelerated proximal gradient descent

    Parameters
    ----------
    f_df : callable
        Function that returns the objective and gradient

    theta_init : array_like
        Initial parameters

    proxop : ProximalOperator
        (e.g. from the proximal_operators module)

    learning_rate : float, optional
        (default: 0.001)

    """

    def __init__(self, f_df, theta_init, proxop, learning_rate=1e-3):
        self.proxop = proxop
        self.lr = learning_rate

        self.obj, self.gradient = wrap(f_df, theta_init)
        super().__init__(theta_init)

    def __iter__(self):

        xk = self.theta.copy().astype('float')
        xprev = xk.copy()
        yk = xk.copy()

        for k in range(self.maxiter):
            with self as state:

                omega = k / (k + 3)

                # update y's
                yk = xk + omega * (xk - xprev)

                # compute the gradient
                grad = self.restruct(state.gradient(yk))

                # update previous
                xprev = xk

                # compute the new iterate
                xk = state.proxop(yk - state.lr * grad, 1. / state.lr)

                yield xk


class ProximalConsensus(Optimizer):
    """
    Proximal Consensus (ADMM)

    Parameters
    ----------
    theta_init : array_like
        Initial parameters

    tau : (float, float, float)
        ADMM scheduling. The augmented Lagrangian quadratic penalty parameter,
        rho, is initialized to tau[0]. Depending on the primal and dual residuals,
        the parameter is increased by a factor of tau[1] or decreased by a factor
        of tau[2] at every iteration. (See Boyd et. al. 2011 for details)

    """

    def __init__(self, theta_init, tau=(10., 2., 2.), tol=(1e-6, 1e-3)):

        self.operators = []
        self.tau = namedtuple('tau', ('init', 'inc', 'dec'))(*tau)
        self.tol = namedtuple('tol', ('primal', 'dual'))(*tol)
        self.gradient = None
        super().__init__(theta_init)

    def obj(self, x):
        return np.nansum([f.objective(self.restruct(x)) for f in self.operators])

    def add(self, name, *args, **kwargs):
        self.operators.append(_getproxop(name, *args, **kwargs))

    def __iter__(self):

        num_obj = len(self.operators)
        assert num_obj >= 1, "Must be at least one objective"

        # initialize
        primals = [self.theta.flatten() for _ in range(num_obj)]
        duals = [np.zeros(self.theta.size) for _ in range(num_obj)]
        theta_avg = np.mean(primals, axis=0).ravel()
        rho = self.tau.init

        self.resid = defaultdict(list)

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

                self.resid['primal'].append(primal_resid)
                self.resid['dual'].append(dual_resid)
                self.resid['rho'].append(rho)

                # check for convergence
                if (primal_resid <= self.tol.primal) & (dual_resid <= self.tol.dual):
                    self.converged = True
                    raise StopIteration("Converged")

                # store primals
                self.primals = primals

                yield theta_avg
