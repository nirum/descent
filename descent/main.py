from .io import printer, store
from . import algorithms
from . import proxops
from copy import deepcopy
from itertools import count
from .utils import wrap, restruct, destruct
import numpy as np


class Optimizer:

    def __init__(self, theta_init, f_df, algorithm, options, proxop=None, rho=None):

        self.theta = theta_init
        self.objective, self.gradient = wrap(f_df, theta_init)
        self.iteration = 0

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

    def restruct(self, x):
        return restruct(x, self.theta)

    def run(self, maxiter=None):

        maxiter = np.inf if maxiter is None else (maxiter + self.iteration)
        xk = destruct(self.theta)

        try:
            for k in count(start=self.iteration):

                # increment the iteration
                self.iteration = k

                grad = self.gradient(xk)
                xk = self.algorithm(grad)

                if 'proxop' in self.__dict__:
                    xk = destruct(self.proxop(self.restruct(xk), self.rho))

                # check for convergence
                if k >= maxiter:
                    break

        except KeyboardInterrupt:
            pass

        self.theta = self.restruct(xk)
