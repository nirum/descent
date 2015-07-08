import numpy as np
from toolz.curried import concat, map, pipe, curry
from toolz.functoolz import isunary
from toolz import first, second, compose, take
from collections import OrderedDict


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
    objective = compose(first, memoized_f_df)
    gradient = compose(second, memoized_f_df)
    return objective, gradient


def dict_to_array(d):
    """Converts a dictionary whose values are numpy arrays to a single array"""
    return list_to_array(d.values())


def array_to_dict(a, dref):
    """Converts an unraveled array to a dictionary"""

    idx = 0
    d = dref.copy()
    for k in dref.keys():
        d[k] = a[idx:(idx+dref[k].size)].reshape(dref[k].shape)
        idx += dref[k].size

    return d


def list_to_array(s):
    """Converts a sequence of numpy arrays to a single array"""

    # make sure the values are all numpy arrays
    list(map(enforce(np.ndarray), s))

    # unravel each array, c
    return pipe(s, map(np.ravel), concat, list, np.array)


def array_to_list(a, sref):
    """Converts an unraveled array to a list of numpy arrays"""

    idx = 0
    s = []
    for si in sref:
        s.append(a[idx:(idx+si.size)].reshape(si.shape))
        idx += si.size

    return s


@curry
def enforce(typeclass, arg):
    """Asserts that the input is of a given typeclass"""

    assert type(arg) == typeclass, "Input must be of " + str(typeclass)


def lrucache(fun, size):
    """
    A simple implementation of a least recently used (LRU) cache.
    Memoizes the recent calls of a computationally intensive function.

    Parameters
    ----------
    fun : function
        Must be unary (take a single argument), and that argument must be hashable

    size : int
        The size of the cache (number of previous calls to store)
    """

    # this only works for unary functions
    assert isunary(fun), "The function must be unary (take a single argument)"

    # the cache (storage) and hash function
    cache = OrderedDict()
    hashfun = lambda x: hash(x.tostring()) if isinstance(x, object) else hash(x)

    def wrapper(x):

        # hash the argument
        try:
            key = hashfun(x)
        except (AttributeError, TypeError):
            print('Input must be hashable')

        # if the key is not in the cache, evalute the function
        if key not in cache:

            # clear space if necessary (keeps the most recent keys)
            if len(cache) >= size:
                cache.pop(take(1, cache.iterkeys()).next())

            # store the new value in the cache
            cache[key] = fun(x)

        return cache[key]

    return wrapper


def check_grad(f_df, x0, eps=1e-6, n=50, tol=1e-4):

    obj, grad = wrap(f_df)
    df = grad(x0)
    f0 = obj(x0)

    # header
    print("{:^10} {}".format('', "Checking the analytical gradient:"))
    print("{:^10} {}".format('', "---------------------------------"))
    print("{:^10} {:<10} | {:<10} | {:<15}".format('', "Numerical", "Analytic", "Error"))

    # check each dimension
    for j in range(x0.size):

        dx = np.zeros(x0.size)
        dx[j] = eps

        df_approx = (obj(x0 + dx) - f0) / eps
        df_analytic = df[j]
        err = (df_approx-df_analytic)**2

        errstr = '********' if err > tol else ''
        print("{:^10} {:<10.4f} | {:<10.4f} | {:<15.6f}".format(errstr, df_approx, df_analytic, err))
