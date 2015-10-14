"""
Parallel algorithms
"""

from .main import Optimizer
from .utils import destruct, wrap
from .proximal_operators import _getproxop, ProximalOperator
import numpy as np
from collections import deque, namedtuple, defaultdict
from builtins import super

__all__ = ['DistributedProximalConsensus']


def farm(funcs, args, *additional_args):
    """
    Maps a bunch of arguments to a bunch of functions, and collects the results

    Parameters
    ----------
    funcs : iterable
        List of functions

    args : iterable
        List of arguments (must have the same size as the functions)

    *additional_args : optional
        Optional arguments passed along to each function call

    Returns
    -------
    results : list
        List containing the result of each function

    """

    try:
        from concurrent.futures import ProcessPoolExecutor

        futures = list()
        with ProcessPoolExecutor() as pool:
            for func, arg in zip(funcs, args):
                futures.append(pool.submit(func, arg, *additional_args))

        return [f.result() for f in futures]

    except ImportError:
        # print('Error: concurrent.futures module not found.')
        # revert to sequential processing
        return [func(arg, *additional_args) for func, arg in zip(funcs, args)]


class DistributedProximalConsensus(Optimizer):

    def __init__(self, theta_init, tau=(10., 2., 2.), tol=(1e-6, 1e-3)):
        """
        Consensus ADMM

        Parameters
        ----------
        theta_init : array_like
            Initial parameters

        """

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
                # funcs, args = [self.operators[varidx] for varidx, dual in enumerate(duals)]
                    # primals[varidx] = self.operators[varidx](self.restruct(theta_prev-dual), rho).ravel()
                primals = [x.ravel() for x in farm(self.operators, (theta_prev - dual for dual in duals), rho)]

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

                yield theta_avg
