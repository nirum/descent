"""
Main routines for the descent package
"""

import numpy as np
import time
from .utils import wrap, destruct, restruct
from collections import defaultdict, namedtuple
from toolz.curried import juxt
from .display import Ascii
from .storage import List
from builtins import super
from copy import deepcopy

Datum = namedtuple('Datum', ['iteration', 'obj', 'grad', 'params', 'runtime'])

__all__ = ['Optimizer']

# this is the awesome master Optimizer superclass, used to house properties
# for all optimization algorithms
class Optimizer(object):

    def __init__(self, theta_init):
        """
        Optimization base class
        """

        # initialize storage of runtimes
        self.runtimes = []

        # display and storage
        self.display = Ascii()
        self.storage = List()

        # custom callbacks
        self.callbacks = []

        # default maxiter
        self.maxiter = 1000

        # machine epsilon (currently unused)
        self.eps = np.finfo(float).eps

        # exit message (for display)
        self.exit_message = None

        self.theta = deepcopy(theta_init)

    def run(self, maxiter=1e3, tol=(1e-18, 1e-18, 1e-16)):

        # reset exit message (for display)
        self.exit_message = None

        # tolerance
        tol = namedtuple('tolerance', ['obj', 'param', 'grad'])(*tol)

        self.maxiter = int(maxiter)
        starting_iteration = len(self)
        callback_func = juxt(*self.callbacks)

        # store previous iterates
        obj_prev = np.Inf
        theta_prev = np.Inf

        # init display
        if self.display:
            self.display.start()

        try:
            for ix, theta in enumerate(self):

                k = starting_iteration + ix
                obj = self.obj(theta)

                if self.gradient:
                    grad = self.restruct(self.gradient(theta))
                else:
                    grad = None

                # hack to get around the first iteration runtime
                if ix >= 1:

                    # collect a bunch of information for the current iterate
                    d = Datum(k, obj, grad, self.restruct(theta), self.runtimes[-1])

                    # send out to callbacks
                    callback_func(d)

                    # display/storage
                    if self.display is not None:
                        self.display(d)

                    if self.storage is not None:
                        self.storage(d)


                # tolerance
                if grad is not None:
                    if np.linalg.norm(destruct(grad), 2) <= (tol.grad * np.sqrt(theta.size)):
                        self.exit_message = 'Stopped on interation {}. Scaled gradient norm: {}'.format(ix, np.sqrt(theta.size) * np.linalg.norm(destruct(grad), 2))
                        break

                elif np.abs(obj - obj_prev) <= tol.obj:
                    self.exit_message = 'Stopped on interation {}. Objective value not changing, |f(x^k) - f(x^{k+1})|: {}'.format(ix, np.abs(obj - obj_prev))
                    break

                elif np.linalg.norm(theta - theta_prev, 2) <= (tol.param * np.sqrt(theta.size)):
                    self.exit_message = 'Stopped on interation {}. Parameters not changing, \sqrt(dim) * ||x^k - x^{k+1}||_2: {}'.format(ix, np.sqrt(theta.size) * np.linalg.norm(theta - theta_prev, 2))
                    break

                theta_prev = theta.copy()
                obj_prev = obj

        except KeyboardInterrupt:
            pass

        self.display.cleanup(d, self.runtimes, self.exit_message) if self.display else None
        self.theta = self.restruct(theta)

    def __len__(self):
        return len(self.runtimes)

    def restruct(self, x):
        return restruct(x, self.theta)

    def reset(self):
        self.runtimes = [0]

    # because why not make each Optimizer a ContextManager
    # (used to wrap the per-iteration computation)
    def __enter__(self):
        """
        Enter
        """

        # time the running time of the inner loop computation
        self.iteration_time = time.time()

        return self

    def __exit__(self, *args):
        """
        exit(self, type, value, traceback)
        """

        runtime = time.time() - self.iteration_time
        self.runtimes.append(runtime)

    def __str__(self): # pragma no cover
        return '{}\n{} iterations\nObjective: {}'.format(
            self.__class__.__name__, len(self), self.obj(destruct(self.theta)))

    def __repr__(self): # pragma no cover
        return str(self)

    def _repr_html_(self): # pragma no cover
        return '''
               <h2>{}</h2>
               <p>{} iterations, objective: {}</p>
               '''.format(
                   self.__class__.__name__,
                   len(self),
                   self.obj(destruct(self.theta)))
