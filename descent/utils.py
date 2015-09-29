import sys
import numpy as np
from toolz.curried import concat, map, pipe, curry
from toolz.functoolz import isunary
from toolz import first, second, compose
from collections import OrderedDict, namedtuple
from multipledispatch import dispatch
from functools import wraps

__all__ = ['check_grad', 'destruct', 'restruct', 'minibatchify']

DESTRUCT_DOCSTR = """Deconstructs the input into a 1-D numpy array"""
RESTRUCT_DOCSTR = """Reshapes the input into the type of the second argument"""

# data type that is being passed around per iteration
datum = namedtuple('Datum', ['iteration', 'obj', 'grad', 'params', 'runtime'])


def wrap(f_df, xref, size=1):
    """
    Memoizes an objective + gradient function, and splits it into
    two functions that return just the objective and gradient, respectively.

    Parameters
    ----------
    f_df : function
        Must be unary (takes a single argument)

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

    # this only works for unary functions
    assert isunary(func), "The function must be unary (take a single argument)"

    # initialize the cache
    cache = OrderedDict()

    def wrapper(x):

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


def minibatchify(gen):
    """
    Takes an objective fun. and a data generator and returns an noisy oracle

    Parameters
    ----------
    gen : generator
        Generates data that is passed to an objective function

    Returns
    -------
    decorator : function
        A function decorator that wraps an objective function & gradient

    """

    def decorate(f_df):

        @wraps(f_df)
        def wrapper(theta):
            return f_df(theta, next(gen))

        return wrapper

    return decorate


def check_grad(f_df, xref, stepsize=1e-6, n=50, tol=1e-6, out=sys.stdout):
    """
    Compares the numerical gradient to the analytic gradient

    Parameters
    ----------
    f_df : function
    x0 : array_like
    stepsize : float, optional
    n : int, optional
    tol : float, optional
    """

    if sys.version_info >= (3,0):
        CORRECT = u'\x1b[32m\N{HEAVY CHECK MARK}\x1b[0m'
        INCORRECT = u'\x1b[31m\N{BALLOT X}\x1b[0m'
    else:
        CORRECT = u'\x1b[32mPass\x1b[0m'
        INCORRECT = u'\x1b[31mFail\x1b[0m'

    obj, grad = wrap(f_df, xref)
    x0 = destruct(xref)
    df = grad(x0)

    # header
    out.write(("{}".format('', "Checking the analytical gradient:\n")))
    out.write(("{}".format("------------------------------------\n")))
    out.write(("{:<10} | {:<10} | {:<15}\n"
               .format("Numerical", "Analytic", "Error")))
    out.write(("{}".format("------------------------------------\n")))
    out.flush()

    # check each dimension
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

        errstr =  CORRECT if error < tol else INCORRECT
        out.write(("{:<10.4f} | {:<10.4f} | {:<5.6f} | {:^2}\n"
                   .format(df_approx, df_analytic, error, errstr)))
        out.flush()


@docstring(DESTRUCT_DOCSTR)
@dispatch(int)
def destruct(x):
    """Convert an int to a numpy array"""
    return destruct(float(x))


@docstring(DESTRUCT_DOCSTR)
@dispatch(float)
def destruct(x):
    """Convert a float to a numpy array"""
    return np.array([x])


@docstring(DESTRUCT_DOCSTR)
@dispatch(dict)
def destruct(x):
    """
    Deconstructs a data structure into a 1-D np.ndarray (via multiple dispatch)
    Converts a dictionary whose values are numpy arrays to a single array
    """

    # take values by sorted keys
    return destruct([x[k] for k in sorted(x)])


@docstring(DESTRUCT_DOCSTR)
@dispatch(tuple)
def destruct(x):
    """
    Deconstructs a data structure into a 1-D np.ndarray (via multiple dispatch)
    Converts a tuple of numpy arrays to a single array
    """
    return destruct(list(x))


@docstring(DESTRUCT_DOCSTR)
@dispatch(list)
def destruct(x):
    """
    Deconstructs a data structure into a 1-D np.ndarray (via multiple dispatch)
    Converts a list of numpy arrays to a single array
    """

    # make sure the values are all numpy arrays
    list(map(enforce(np.ndarray), x))

    # unravel each array, c
    return pipe(x, map(np.ravel), concat, list, np.array)


@docstring(DESTRUCT_DOCSTR)
@dispatch(np.ndarray)
def destruct(x):
    """
    Deconstructs a data structure into a 1-D np.ndarray (via multiple dispatch)
    Converts an N-D numpy array to a 1-D array
    """

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
    """
    Reconstructs a data structure from a 1-D np.ndarray (via multiple dispatch)
    Converts an unraveled array to a dictionary
    """

    idx = 0
    d = ref.copy()
    for k in sorted(ref):
        d[k] = x[idx:(idx+ref[k].size)].reshape(ref[k].shape)
        idx += ref[k].size

    return d


@docstring(RESTRUCT_DOCSTR)
@dispatch(np.ndarray, np.ndarray)
def restruct(x, ref):
    """
    Reconstructs a data structure from a 1-D np.ndarray (via multiple dispatch)
    Converts an unraveled array to an N-D array
    """
    return x.reshape(ref.shape)


@docstring(RESTRUCT_DOCSTR)
@dispatch(np.ndarray, tuple)
def restruct(x, ref):
    """
    Reconstructs a data structure from a 1-D np.ndarray (via multiple dispatch)
    Converts an unraveled array to an tuple
    """
    return tuple(restruct(x, list(ref)))


@docstring(RESTRUCT_DOCSTR)
@dispatch(np.ndarray, list)
def restruct(x, ref):
    """
    Reconstructs a data structure from a 1-D np.ndarray (via multiple dispatch)
    Converts an unraveled array to a list of numpy arrays
    """

    idx = 0
    s = []
    for si in ref:
        s.append(x[idx:(idx+si.size)].reshape(si.shape))
        idx += si.size

    return s


@curry
def enforce(typeclass, arg):
    """Asserts that the input is of a given typeclass"""
    assert type(arg) == typeclass, "Input must be of " + str(typeclass)
