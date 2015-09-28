import numpy as np
import time
from .utils import wrap, destruct, restruct
from collections import defaultdict
from toolz.curried import juxt


# this is the awesome master Optimizer superclass, used to house properties
# for all optimization algorithms
class Optimizer(object):

    def __init__(self):
        """
        Optimization base class
        """

        # flags to keep track of the state of the optimizer
        self.started = False
        self.suspended = False
        self.completed = False

        # time stuff
        self.timers = defaultdict(list)

        self.callbacks = []

    def run(self, maxiter=1e2):

        if self.started:
            print('Already started!')
            return None

        if self.suspended:
            print('Already started!')
            return None

        if self.completed:
            print('Already finished!')
            return None

        self.started = True
        self.maxiter = int(maxiter)
        self.runtimes = list()

        callback_func = juxt(*self.callbacks)

        for obj, params in self:

            try:

                # run callbacks
                callback_func(obj, params)

                # display
                print('Error = {}'.format(obj), flush=True)

                # storage
                # self.results.append(...)

            except KeyboardInterrupt:
                print('Shutting Down!')

                # TODO: deal with shut down

                self.suspended = True

                break

        self.completed = True
        return params

    def reset(self, *args, **kwargs):

        self.started = False
        self.completed = False
        self.suspended = False

    # because why not make each Optimizer a ContextManager
    # (used to wrap the per-iteration computation)
    def __enter__(self):

        # time the running time of the inner loop computation
        self.inner_timer = time.time()

        return self

    def __exit__(self, *args):
        """
        exit(self, type, value, traceback)
        """

        self.timers['inner'].append(time.time() - self.inner_timer)

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

        # store the initial parameters, unraveled into a 1-D numpy array
        self.theta_init = destruct(theta_init)

        super().__init__()


class ProximalOptimizer(Optimizer):
    pass
