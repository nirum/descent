from . import algorithms
from . import proxops
from copy import deepcopy
from itertools import count
from collections import namedtuple, defaultdict
from .utils import wrap, restruct, destruct
import numpy as np


class Optimizer:

    def __init__(self, theta_init):
        self.iteration = 0
        self.theta = theta_init

    def __next__(self):
        raise NotImplementedError

    def run(self, maxiter=None):

        maxiter = np.inf if maxiter is None else (maxiter + self.iteration)

        try:
            for k in count(start=self.iteration):

                self.iteration = k

                # TODO: time each iteration
                # get the next iteration
                self.theta = next(self)

                # TODO: run callbacks

                # TODO: check for convergence
                if k >= maxiter:
                    break

        except KeyboardInterrupt:
            pass

    def restruct(self, x):
        return restruct(x, self.theta)


class GradientDescent(Optimizer):

    def __init__(self, theta_init, f_df, algorithm, options, proxop=None, rho=None):

        super().__init__(theta_init)
        self.objective, self.gradient = wrap(f_df, theta_init)

        if type(algorithm) is str:
            self.algorithm = getattr(algorithms, algorithm)(destruct(theta_init), **options)
        elif issubclass(algorithm, algorithms.Algorithm):
            self.algorithm = algorithm(destruct(theta_init), **options)
        else:
            raise ValueError('Algorithm not valid')

        if proxop is not None:

            assert isinstance(proxop, proxops.ProximalOperatorBaseClass), \
                "proxop must subclass the proximal operator base class"

            assert rho is not None, \
                "Must give a value for rho"

            self.proxop = proxop
            self.rho = rho

    def __next__(self):
        """
        Runs one step of the optimization algorithm

        """

        grad = self.gradient(destruct(self.theta))
        xk = self.algorithm(grad)

        if 'proxop' in self.__dict__:
            xk = destruct(self.proxop(self.restruct(xk), self.rho))

        return self.restruct(xk)


class Consensus(Optimizer):

    def __init__(self, theta_init, proxops, tau=(10., 2., 2.), tol=(1e-6, 1e-3)):
        """
        Proximal Consensus (ADMM)

        Parameters
        ----------
        theta_init : array_like
            Initial parameters

        proxops : list
            Proximal operators

        tau : (float, float, float)
            ADMM scheduling. The augmented Lagrangian quadratic penalty parameter,
            rho, is initialized to tau[0]. Depending on the primal and dual residuals,
            the parameter is increased by a factor of tau[1] or decreased by a factor
            of tau[2] at every iteration. (See Boyd et. al. 2011 for details)
        """

        assert len(proxops) >= 1, "Must be at least one objective"

        super().__init__(theta_init)
        self.operators = proxops
        self.tau = namedtuple('tau', ('init', 'inc', 'dec'))(*tau)
        self.tol = namedtuple('tol', ('primal', 'dual'))(*tol)
        self.gradient = None

        # initialize
        self.primals = [destruct(theta_init) for _ in proxops]
        self.duals = [np.zeros_like(p) for p in self.primals]
        self.rho = self.tau.init
        self.resid = defaultdict(list)

    def __next__(self):

        # store the parameters from the previous iteration
        theta_prev = destruct(self.theta)

        # update each primal variable
        self.primals = [op(self.restruct(theta_prev - dual), self.rho).ravel()
                        for op, dual in zip(self.operators, self.duals)]

        # average primal copies
        theta_avg = np.mean(self.primals, axis=0)

        # update the dual variables (after primal update has finished)
        self.duals = [dual + primal - theta_avg
                      for dual, primal in zip(self.duals, self.primals)]

        # compute primal and dual residuals
        primal_resid = float(np.sum([np.linalg.norm(primal - theta_avg)
                                     for primal in self.primals]))
        dual_resid = len(self.operators) * self.rho ** 2 * \
            np.linalg.norm(theta_avg - theta_prev)

        # update penalty parameter according to primal and dual residuals
        # (see sect. 3.4.1 of the Boyd and Parikh ADMM paper)
        if primal_resid > self.tau.init * dual_resid:
            self.rho *= float(self.tau.inc)
        elif dual_resid > self.tau.init * primal_resid:
            self.rho /= float(self.tau.dec)

        # self.resid['primal'].append(primal_resid)
        # self.resid['dual'].append(dual_resid)
        # self.resid['rho'].append(rho)

        # check for convergence
        # if (primal_resid <= self.tol.primal) & (dual_resid <= self.tol.dual):
            # self.converged = True
            # raise StopIteration("Converged")

        return self.restruct(theta_avg)
