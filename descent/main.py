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

        self.theta = deepcopy(theta_init)

    def run(self, maxiter=1e3):

        self.maxiter = int(maxiter)
        starting_iteration = len(self)
        callback_func = juxt(*self.callbacks)

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

                if len(self.runtimes) == 0:
                    rt = 0.
                else:
                    rt = self.runtimes[-1]

                # build the datum
                d = Datum(k, obj, grad, self.restruct(theta), rt)

                # farm out to callbacks
                callback_func(d)

                # display/storage
                if self.display is not None:
                    self.display(d)

                if self.storage is not None:
                    self.storage(d)

        except KeyboardInterrupt:
            pass

        self.display.cleanup(d, self.runtimes) if self.display else None
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

    def __str__(self):
        return '{}\n{} iterations\nObjective: {}'.format(
            self.__class__.__name__, len(self), self.obj(self.theta))

    def __repr__(self):
        return str(self)
