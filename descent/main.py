"""
Main routines for the descent package
"""

import numpy as np
import time
from . import algorithms
from .utils import wrap, destruct, restruct
from collections import defaultdict, namedtuple
from toolz.curried import juxt
from .display import Ascii
from .storage import List
from builtins import super
from copy import deepcopy
from types import GeneratorType

Datum = namedtuple('Datum', ['iteration', 'obj', 'grad', 'params', 'runtime'])

__all__ = ['Optimizer']


# this is the awesome master Optimizer superclass, used to house properties
# for all optimization algorithms
class Optimizer(object):

    def __init__(self, f_df, theta_init, algorithm, **kwargs):
        """
        Optimization base class

        Parameters
        ----------
        f_df : function

        theta_init : array, dict, or list

        algorithm : string

        **kwargs : dict

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

        # get objective and gradient
        self.obj, self.gradient = wrap(f_df, theta_init)

        self.theta_init = theta_init

        # initialize algorithm
        try:
            self.algorithm = getattr(algorithms, algorithm)(deepcopy(destruct(theta_init)), **kwargs)
        except:
            raise ValueError("Algorithm '" + str(algorithm) + "' not valid.")

        self.theta = self.restruct(self.algorithm.send(None))

    def run(self, maxiter=1e3, tol=(1e-18, 1e-18, 1e-16)):

        # reset exit message (for display)
        self.exit_message = None
        theta_prev = np.Inf
        obj_prev = np.Inf

        # tolerance
        tol = namedtuple('tolerance', ['obj', 'param', 'grad'])(*tol)

        self.maxiter = int(maxiter)
        callback_func = juxt(*self.callbacks)

        # init display
        if self.display is not None:
            self.display.start()
            display_batch_size = self.display.every

        else:
            display_batch_size = 1

        try:
            for k in range(len(self), len(self) + self.maxiter):

                # store objective and gradient computation time
                tstart = time.time()
                obj = self.obj(self.theta)
                grad = self.gradient(self.theta)
                obj_runtime = time.time() - tstart

                tstart = time.time()
                self.theta = self.restruct(self.algorithm.send(grad))
                alg_runtime = time.time() - tstart

                self.runtimes.append(obj_runtime + alg_runtime)

                # collect a bunch of information for the current iterate
                d = Datum(k, obj, grad, self.theta, np.sum(self.runtimes[-display_batch_size:]))

                # send out to callbacks
                callback_func(d)

                # display/storage
                if self.display is not None:
                    self.display(d)

                if self.storage is not None:
                    self.storage(d)

                # tolerance
                grad_vec = destruct(grad)
                theta_vec = destruct(self.theta)
                if np.linalg.norm(grad_vec, 2) <= (tol.grad * np.sqrt(grad_vec.size)):
                    self.exit_message = 'Stopped on interation {}. Scaled gradient norm: {}'.format(k, np.sqrt(grad_vec.size) * np.linalg.norm(grad_vec, 2))
                    break

                elif np.abs(obj - obj_prev) <= tol.obj:
                    self.exit_message = 'Stopped on interation {}. Objective value not changing, |f_current - f_prev|: {}'.format(k, np.abs(obj - obj_prev))
                    break

                elif np.linalg.norm(theta_vec - theta_prev, 2) <= (tol.param * np.sqrt(theta_vec.size)):
                    self.exit_message = 'Stopped on interation {}. Parameters not changing, \sqrt(dim) * ||x_current - x_prev||_2: {}'.format(k, np.sqrt(theta_vec.size) * np.linalg.norm(theta_vec - theta_prev, 2))
                    break

                theta_prev = theta_vec.copy()
                obj_prev = obj

        except KeyboardInterrupt:
            pass

        self.display.cleanup(d, self.runtimes, self.exit_message) if self.display else None

    def __len__(self):
        return len(self.runtimes)

    def restruct(self, x):
        return restruct(x, self.theta_init)

    def reset(self):
        self.runtimes = []
        self.exit_message = None

    def __str__(self): # pragma no cover
        return '{}\n{} iterations\nObjective: {}'.format(
            self.__class__.__name__, len(self), self.obj(destruct(self.theta)))

    def __repr__(self): # pragma no cover
        return str(self)

    def _repr_html_(self): # pragma no cover
        return '''
               <h2>{}</h2>
               <p>{} iterations, objective: {}</p>
               '''.format(self.__class__.__name__,
                          len(self),
                          self.obj(destruct(self.theta)))
