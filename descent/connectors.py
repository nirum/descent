from copy import deepcopy
from itertools import count
from functools import wraps
from .utils import coroutine

__all__ = ['run', 'select']


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



@coroutine
def select(index):
    """
    Selects one element from a sent tuple or list of values

    """

    x = yield
    while True:
        x = yield x[index]
