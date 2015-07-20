import numpy as np
from toolz.curried import concat, map, pipe, curry
from toolz.functoolz import isunary
from toolz import first, second, compose
from collections import OrderedDict
from multipledispatch import dispatch


def wrap(f_df, size=1):
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

    memoized_f_df = lrucache(f_df, size)
    objective = compose(first, memoized_f_df, destruct)
    gradient = compose(second, memoized_f_df, destruct)
    return objective, gradient


def lrucache(fun, size):
    """
    A simple implementation of a least recently used (LRU) cache.
    Memoizes the recent calls of a computationally intensive function.

    Parameters
    ----------
    fun : function
        Must be unary (takes a single argument)

    size : int
        The size of the cache (number of previous calls to store)
    """

    # this only works for unary functions
    assert isunary(fun), "The function must be unary (take a single argument)"

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
            cache[key] = fun(x)

        return cache[key]

    return wrapper


def check_grad(f_df, x0, eps=1e-6, n=50, tol=1e-4):
    """
    Compares the numerical gradient to the analytic gradient

    Parameters
    ----------
    f_df : function
    x0 : array_like
    eps : float, optional
    n : int, optional
    tol : float, optional
    """

    xarray = destruct(x0).copy

    obj, grad = wrap(f_df)
    df = grad(x0)
    f0 = obj(x0)

    # header
    print(("{:^10} {}".format('', "Checking the analytical gradient:")))
    print(("{:^10} {}".format('', "---------------------------------")))
    print(("{:^10} {:<10} | {:<10} | {:<15}"
           .format('', "Numerical", "Analytic", "Error")))

    # check each dimension
    for j in range(x0.size):

        dx = np.zeros(x0.size)
        dx[j] = eps

        df_approx = (obj(x0 + dx) - f0) / eps
        df_analytic = df[j]
        err = (df_approx-df_analytic)**2

        errstr = '********' if err > tol else ''
        print(("{:^10} {:<10.4f} | {:<10.4f} | {:<15.6f}"
               .format(errstr, df_approx, df_analytic, err)))


@dispatch(dict)
def destruct(x):
    """
    Deconstructs a data structure into a 1-D np.ndarray (via multiple dispatch)
    Converts a dictionary whose values are numpy arrays to a single array
    """

    # take values by sorted keys
    return destruct([x[k] for k in sorted(x)])


@dispatch(tuple)
def destruct(x):
    """
    Deconstructs a data structure into a 1-D np.ndarray (via multiple dispatch)
    Converts a tuple of numpy arrays to a single array
    """
    return destruct(list(x))


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


@dispatch(np.ndarray)
def destruct(x):
    """
    Deconstructs a data structure into a 1-D np.ndarray (via multiple dispatch)
    Converts an N-D numpy array to a 1-D array
    """

    return x.ravel()


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


@dispatch(np.ndarray, np.ndarray)
def restruct(x, ref):
    """
    Reconstructs a data structure from a 1-D np.ndarray (via multiple dispatch)
    Converts an unraveled array to an N-D array
    """
    return x.reshape(ref.shape)


@dispatch(np.ndarray, tuple)
def restruct(x, ref):
    """
    Reconstructs a data structure from a 1-D np.ndarray (via multiple dispatch)
    Converts an unraveled array to an tuple
    """
    return tuple(restruct(x, list(ref)))


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
