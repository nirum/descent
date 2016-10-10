from __future__ import (absolute_import, division, print_function, unicode_literals)
import sys
import numpy as np
from toolz.curried import concat, map, pipe
from toolz.functoolz import is_arity
from toolz import first, second, compose
from collections import OrderedDict
from multipledispatch import dispatch
from functools import wraps
import tableprint as tp

DESTRUCT_DOCSTR = """Deconstructs the input into a 1-D numpy array"""
RESTRUCT_DOCSTR = """Reshapes the input into the type of the second argument"""

__all__ = ['check_grad', 'destruct', 'restruct', 'wrap']


def wrap(f_df, xref, size=1):
    """
    Memoizes an objective + gradient function, and splits it into
    two functions that return just the objective and gradient, respectively.

    Parameters
    ----------
    f_df : function
        Must be unary (takes a single argument)

    xref : list, dict, or array_like
        The form of the parameters

    size : int, optional
        Size of the cache (Default=1)
    """
    memoized_f_df = lrucache(lambda x: f_df(restruct(x, xref)), size)
    objective = compose(first, memoized_f_df)
    gradient = compose(destruct, second, memoized_f_df)
    return objective, gradient


def docstring(docstr):
    """
    Decorates a function with the given docstring

    Parameters
    ----------
    docstr : string
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.__doc__ = docstr
        return wrapper
    return decorator


def lrucache(func, size):
    """
    A simple implementation of a least recently used (LRU) cache.
    Memoizes the recent calls of a computationally intensive function.

    Parameters
    ----------
    func : function
        Must be unary (takes a single argument)

    size : int
        The size of the cache (number of previous calls to store)
    """

    if size == 0:
        return func
    elif size < 0:
        raise ValueError("size argument must be a positive integer")

    # this only works for unary functions
    if not is_arity(1, func):
        raise ValueError("The function must be unary (take a single argument)")

    # initialize the cache
    cache = OrderedDict()

    def wrapper(x):
        if not(type(x) is np.ndarray):
            raise ValueError("Input must be an ndarray")

        # hash the input, using tostring for small and repr for large arrays
        if x.size <= 1e4:
            key = hash(x.tostring())
        else:
            key = hash(repr(x))

        # if the key is not in the cache, evalute the function
        if key not in cache:

            # clear space if necessary (keeps the most recent keys)
            if len(cache) >= size:
                cache.popitem(last=False)

            # store the new value in the cache
            cache[key] = func(x)

        return cache[key]

    return wrapper


def check_grad(f_df, xref, stepsize=1e-6, tol=1e-6, width=15, style='round', out=sys.stdout):
    """
    Compares the numerical gradient to the analytic gradient

    Parameters
    ----------
    f_df : function
        The analytic objective and gradient function to check

    x0 : array_like
        Parameter values to check the gradient at

    stepsize : float, optional
        Stepsize for the numerical gradient. Too big and this will poorly estimate the gradient.
        Too small and you will run into precision issues (default: 1e-6)

    tol : float, optional
        Tolerance to use when coloring correct/incorrect gradients (default: 1e-5)

    width : int, optional
        Width of the table columns (default: 15)

    style : string, optional
        Style of the printed table, see tableprint for a list of styles (default: 'round')
    """
    CORRECT = u'\x1b[32m\N{CHECK MARK}\x1b[0m'
    INCORRECT = u'\x1b[31m\N{BALLOT X}\x1b[0m'

    obj, grad = wrap(f_df, xref, size=0)
    x0 = destruct(xref)
    df = grad(x0)

    # header
    out.write(tp.header(["Numerical", "Analytic", "Error"], width=width, style=style) + "\n")
    out.flush()

    # helper function to parse a number
    def parse_error(number):

        # colors
        failure = "\033[91m"
        passing = "\033[92m"
        warning = "\033[93m"
        end = "\033[0m"
        base = "{}{:0.3e}{}"

        # correct
        if error < 0.1 * tol:
            return base.format(passing, error, end)

        # warning
        elif error < tol:
            return base.format(warning, error, end)

        # failure
        else:
            return base.format(failure, error, end)

    # check each dimension
    num_errors = 0
    for j in range(x0.size):

        # take a small step in one dimension
        dx = np.zeros(x0.size)
        dx[j] = stepsize

        # compute the centered difference formula
        df_approx = (obj(x0 + dx) - obj(x0 - dx)) / (2 * stepsize)
        df_analytic = df[j]

        # relative error
        normsum = np.linalg.norm(df_approx) + np.linalg.norm(df_analytic)
        error = np.linalg.norm(df_approx - df_analytic) / normsum \
            if normsum > 0 else 0

        num_errors += error >= tol
        errstr = CORRECT if error < tol else INCORRECT
        out.write(tp.row([df_approx, df_analytic, parse_error(error) + ' ' + errstr],
                         width=width, style=style) + "\n")
        out.flush()

    out.write(tp.bottom(3, width=width, style=style) + "\n")
    return num_errors


@docstring(DESTRUCT_DOCSTR)
@dispatch(int)
def destruct(x):
    return destruct(float(x))


@docstring(DESTRUCT_DOCSTR)
@dispatch(float)
def destruct(x):
    return np.array([x])


@docstring(DESTRUCT_DOCSTR)
@dispatch(dict)
def destruct(x):
    # take values by sorted keys
    return destruct([x[k] for k in sorted(x)])


@docstring(DESTRUCT_DOCSTR)
@dispatch(tuple)
def destruct(x):
    return destruct(list(x))


@docstring(DESTRUCT_DOCSTR)
@dispatch(list)
def destruct(x):
    # unravel each array, c
    return pipe(x, map(destruct), concat, list, np.array)


@docstring(DESTRUCT_DOCSTR)
@dispatch(np.ndarray)
def destruct(x):
    return x.ravel()


@docstring(RESTRUCT_DOCSTR)
@dispatch(np.ndarray, int)
def restruct(x, ref):
    return float(x)


@docstring(RESTRUCT_DOCSTR)
@dispatch(np.ndarray, float)
def restruct(x, ref):
    return float(x)


@docstring(RESTRUCT_DOCSTR)
@dispatch(np.ndarray, dict)
def restruct(x, ref):
    idx = 0
    newdict = ref.copy()
    for key in sorted(ref):
        elem_size = destruct(ref[key]).size
        newdict[key] = restruct(x[idx:(idx + elem_size)], ref[key])
        idx += elem_size

    return newdict


@docstring(RESTRUCT_DOCSTR)
@dispatch(np.ndarray, np.ndarray)
def restruct(x, ref):
    return x.reshape(ref.shape)


@docstring(RESTRUCT_DOCSTR)
@dispatch(np.ndarray, tuple)
def restruct(x, ref):
    return tuple(restruct(x, list(ref)))


@docstring(RESTRUCT_DOCSTR)
@dispatch(np.ndarray, list)
def restruct(x, ref):
    idx = 0
    newlist = []
    for elem in ref:
        elem_size = destruct(elem).size
        newlist.append(restruct(x[idx:(idx + elem_size)], elem))
        idx += elem_size

    return newlist
