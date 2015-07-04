import numpy as np
from toolz.curried import concat, map, pipe, curry


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
