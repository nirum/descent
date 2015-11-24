from .connectors import run, select, concat, pipe, broadcast, saveall, save, join
from .utils import wrap, make_coroutine, destruct_coro, restruct_coro, destruct
from .io import printer, store
from . import algorithms
from . import proxops
from copy import deepcopy
from itertools import count
import numpy as np


class Optimizer:

    def __init__(self, theta_init):

        self.theta = deepcopy(theta_init)
        self.iteration = 0

        # default callbacks
        self.callbacks = [printer('iteration', 'objective'),
                            store('iteration', 'objective')]

    def run(self, maxiter=None):

        maxiter = np.inf if maxiter is None else (maxiter + self.iteration)
        self.cleanup = False

        try:
            for k in count(start=self.iteration):

                # increment the iteration
                self.iteration = k

                # get the new iterate of the parameters by passing them through
                # the pipeline
                self.theta = pipe(self.theta, self.pipeline, log=False)

                # broadcast metadata to any callbacks
                self._callback()

                # check for convergence
                if k >= maxiter:
                    break

        except KeyboardInterrupt:
            pass

        # cleanup
        self.cleanup = True
        self._callback()

    def _callback(self):
        results = broadcast(self.__dict__, self.callbacks)

        if self.cleanup:
            tmp = {cb.__name__: res for cb, res in zip(self.callbacks, results) if res is not None}
            self.__dict__.update(tmp)


class GradientDescent(Optimizer):

    def __init__(self, theta_init, f_df, optimizer, projection=None):

        optimizer.send(destruct(theta_init))
        f_df_coro = make_coroutine(f_df)()

        if projection is None:
            projection = join(concat(0.), proxops.identity())

        self.pipeline = [f_df_coro,
                         saveall(('objective', 'gradient'), self.__dict__),
                         select(1),
                         destruct_coro(),
                         optimizer,
                         restruct_coro(theta_init),
                         projection,
                         ]

        super().__init__(theta_init)

class Consensus(Optimizer):

    def __init__(self, theta_init, operators=None):

        self.operators = operators
        super().__init__(theta_init)
