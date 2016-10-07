from . import proxops
from .utils import destruct, restruct, wrap

import sys
from time import perf_counter
from collections import namedtuple, defaultdict
from itertools import count
from functools import wraps

import numpy as np
from scipy.optimize import OptimizeResult
from toolz import compose
import tableprint as tp

__all__ = ['Consensus', 'gradient_optimizer']


class Optimizer:
    def restruct(self, x):
        if 'theta' not in self.__dict__:
            raise KeyError('theta not defined')
        return restruct(x, self.theta)

    def optional_print(self, message):
        if self.display:
            self.display.write(message + "\n")
            self.display.flush()


class Consensus(Optimizer):
    def __init__(self, tau=(10., 2., 2.), tol=(1e-6, 1e-3)):
        """
        Proximal Consensus (ADMM)

        Parameters
        ----------
        tau : (float, float, float)
            ADMM scheduling. The augmented Lagrangian quadratic penalty parameter,
            rho, is initialized to tau[0]. Depending on the primal and dual residuals,
            the parameter is increased by a factor of tau[1] or decreased by a factor
            of tau[2] at every iteration. (See Boyd et. al. 2011 for details)

        tol : (float, float)
            Primal and Dual residual tolerances
        """
        self.operators = []
        self.tau = namedtuple('tau', ('init', 'inc', 'dec'))(*tau)
        self.tol = namedtuple('tol', ('primal', 'dual'))(*tol)

    def add(self, operator, *args):
        """Adds a proximal operator to the list of operators"""

        if isinstance(operator, str):
            op = getattr(proxops, operator)(*args)
        elif issubclass(operator, proxops.ProximalOperatorBaseClass):
            op = operator

        self.operators.append(op)
        return self

    def minimize(self, x0, display=None, maxiter=np.Inf):

        self.theta = x0
        primals = [destruct(x0) for _ in self.operators]
        duals = [np.zeros_like(p) for p in primals]
        rho = self.tau.init
        resid = defaultdict(list)

        try:
            for k in count():

                # store the parameters from the previous iteration
                theta_prev = destruct(self.theta)

                # update each primal variable
                primals = [op(self.restruct(theta_prev - dual), rho).ravel()
                           for op, dual in zip(self.operators, duals)]

                # average primal copies
                theta_avg = np.mean(primals, axis=0)

                # update the dual variables (after primal update has finished)
                duals = [dual + primal - theta_avg
                         for dual, primal in zip(duals, primals)]

                # compute primal and dual residuals
                primal_resid = float(np.sum([np.linalg.norm(primal - theta_avg)
                                             for primal in primals]))
                dual_resid = len(self.operators) * rho ** 2 * \
                    np.linalg.norm(theta_avg - theta_prev)

                # update penalty parameter according to primal and dual residuals
                # (see sect. 3.4.1 of the Boyd and Parikh ADMM paper)
                if primal_resid > self.tau.init * dual_resid:
                    rho *= float(self.tau.inc)
                elif dual_resid > self.tau.init * primal_resid:
                    rho /= float(self.tau.dec)

                resid['primal'].append(primal_resid)
                resid['dual'].append(dual_resid)
                resid['rho'].append(rho)

                # check for convergence
                if (primal_resid <= self.tol.primal) & (dual_resid <= self.tol.dual):
                    break

                if k >= maxiter:
                    break

        except KeyboardInterrupt:
            pass

        return OptimizeResult({
            'x': self.restruct(theta_avg),
            'k': k,
        })


def gradient_optimizer(coro):
    """Turns a coroutine into a gradient based optimizer."""

    class GradientOptimizer(Optimizer):

        @wraps(coro)
        def __init__(self, *args, **kwargs):
            self.algorithm = coro(*args, **kwargs)
            self.algorithm.send(None)
            self.operators = []

        def set_transform(self, func):
            self.transform = compose(destruct, func, self.restruct)

        def minimize(self, f_df, x0, display=sys.stdout, maxiter=1e3):

            self.display = display
            self.theta = x0

            # setup
            xk = self.algorithm.send(destruct(x0).copy())
            store = defaultdict(list)
            runtimes = []
            if len(self.operators) == 0:
                self.operators = [proxops.identity()]

            # setup
            obj, grad = wrap(f_df, x0)
            transform = compose(destruct, *reversed(self.operators), self.restruct)

            self.optional_print(tp.header(['Iteration', 'Objective', '||Grad||', 'Runtime']))
            try:
                for k in count():

                    # setup
                    tstart = perf_counter()
                    f = obj(xk)
                    df = grad(xk)
                    xk = transform(self.algorithm.send(df))
                    runtimes.append(perf_counter() - tstart)
                    store['f'].append(f)

                    # Update display
                    self.optional_print(tp.row([k,
                                                f,
                                                np.linalg.norm(destruct(df)),
                                                tp.humantime(runtimes[-1])]))

                    if k >= maxiter:
                        break

            except KeyboardInterrupt:
                pass

            self.optional_print(tp.bottom(4))

            # cleanup
            self.optional_print(u'\u279b Final objective: {}'.format(store['f'][-1]))
            self.optional_print(u'\u279b Total runtime: {}'.format(tp.humantime(sum(runtimes))))
            self.optional_print(u'\u279b Per iteration runtime: {} +/- {}'.format(
                tp.humantime(np.mean(runtimes)),
                tp.humantime(np.std(runtimes)),
            ))

            # result
            return OptimizeResult({
                'x': self.restruct(xk),
                'f': f,
                'df': self.restruct(df),
                'k': k,
                'obj': np.array(store['f']),
            })

    return GradientOptimizer
