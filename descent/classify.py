import numpy as np
import time
from .utils import wrap, destruct, restruct, datum
from collections import defaultdict
from toolz.curried import juxt
from .display import Ascii
from .storage import List
from builtins import super

# this is the awesome master Optimizer superclass, used to house properties
# for all optimization algorithms
class Optimizer(object):

    def __init__(self):
        """
        Optimization base class
        """

        # clear
        self.reset()

        # display and storage
        self.display = Ascii()
        self.storage = List()

        # custom callbacks
        self.callbacks = []

        # default maxiter
        self.maxiter = 100

    def run(self, maxiter=1e2):

        if self.completed:
            print('Already finished!')
            return None

        if self.started:
            print('Already started!')
            return None

        if self.suspended:
            print('Already started!')
            return None

        self.started = True
        self.maxiter = int(maxiter)

        self.callbacks.append(self.display) if self.display else None
        self.callbacks.append(self.storage) if self.storage else None
        callback_func = juxt(*self.callbacks)

        try:

            for k, val in enumerate(self):

                # pull values out
                obj, params, grad = val

                # build the datum
                d = datum(k, obj, restruct(params, self.theta_init), restruct(grad, self.theta_init), self.elapsed)

                # farm out to callbacks
                callback_func(d)

        except KeyboardInterrupt:
            print('Shutting Down!')

            self.display.cleanup(d, self.runtimes) if self.display else None
            self.suspended = True
            return params

        self.completed = True
        self.display.cleanup(d, self.runtimes) if self.display else None
        return restruct(params, self.theta_init)

    def reset(self, *args, **kwargs):

        self.started = False
        self.completed = False
        self.suspended = False
        self.elapsed = 0.
        self.runtimes = []

    # because why not make each Optimizer a ContextManager
    # (used to wrap the per-iteration computation)
    def __enter__(self):

        # time the running time of the inner loop computation
        self.iteration_time = time.time()

        return self

    def __exit__(self, *args):
        """
        exit(self, type, value, traceback)
        """

        runtime = time.time() - self.iteration_time
        self.runtimes.append(runtime)
        self.elapsed += runtime

    def __str__(self):
        return self.__class__.__name__


# Useful superclass for gradient based optimizers (those that utilize an f_df
# function that returns an objective and gradient)
class GradientOptimizer(Optimizer):

    def __init__(self, f_df, theta_init):
        """
        Gradient based optimization base class
        """

        # memoize the given objective function
        self.obj, self.gradient = wrap(f_df, theta_init)

        # store the initial parameters, in the original format
        self.theta_init = theta_init

        super().__init__()


class ProximalOptimizer(Optimizer):
    pass
