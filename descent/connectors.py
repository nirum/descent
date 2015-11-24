from copy import deepcopy
from itertools import count
from functools import wraps
from .utils import coroutine

__all__ = ['run', 'select', 'concat']


def run(x0, pipeline, maxiter=None):

    # init
    x = deepcopy(x0)

    if maxiter is None:
        maxiter = np.inf

    try:
        for k in count():

            for coro in pipeline:
                x = coro.send(x)

            # break?
            if k >= maxiter:
                break

    except KeyboardInterrupt:
        pass

    return x


def pipe(x, pipeline, log=False):

    if log:
        print('Piping {} through {} pipes.'.format(x, len(pipeline)))

    for coro in pipeline:
        x = coro.send(x)

        if log:
            print('-> ({}) {}'.format(coro.__name__, x))

    return x


@coroutine
def join(*coros):
    x = yield
    while True:
        for coro in coros:
            x = coro.send(x)
        x = yield x


def broadcast(x, branches):
    return [branch.send(x) for branch in branches]


@coroutine
def saveall(keys, cache):
    values = yield
    while True:
        for key, value in zip(keys, values):
            cache[key] = value

        values = yield values

@coroutine
def save(key, cache):
    value = yield
    while True:
        cache[key] = value
        value = yield value


@coroutine
def select(index):
    """
    Selects one element from a sent tuple or list of values

    """

    x = yield
    while True:
        x = yield x[index]


@coroutine
def concat(rho):
    """
    Concatenates a value

    """

    x = yield
    while True:
        x = yield (x, rho)
