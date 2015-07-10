"""
Test conversion utilities
"""

import numpy as np
from descent.utils import destruct, restruct, lrucache
from nose.tools import timed
from time import sleep


def test_destruct():
    """Tests the destruct utility function"""

    # numpy ndarray
    aref = np.linspace(0, 1, 30).reshape(2, 5, 3).T
    assert np.allclose(aref.ravel(), destruct(aref)), "Numpy ndarray destruct"

    # Dictionary
    dref = {
        'a': np.arange(5).reshape(1, 5),
        'c': np.linspace(0, 1, 6).reshape(3, 2),
        'b': np.ones(2)
    }
    darray = np.array([0, 1, 2, 3, 4, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1])
    assert np.allclose(darray, destruct(dref)), "Dictionary destruct"

    # List
    lref = [np.eye(2), np.array([-1, 1]), np.array([[10, 20], [30, 40]])]
    larray = np.array([1, 0, 0, 1, -1, 1, 10, 20, 30, 40])
    assert np.allclose(larray, destruct(lref)), "List destruct"

    # Tuple
    assert np.allclose(larray, destruct(tuple(lref))), "Tuple destruct"


def test_restruct():
    """Tests the destruct utility function"""

    # numpy ndarray
    aref = np.linspace(0, 1, 30).reshape(2, 5, 3).T
    assert np.allclose(restruct(aref.ravel(), np.zeros((3, 5, 2))), aref), \
        "Numpy ndarray restruct"

    # Dictionary
    dref = {
        'a': np.arange(5).reshape(1, 5),
        'b': np.ones(2),
        'c': np.linspace(0, 1, 6).reshape(3, 2)
    }
    dzeros = {
        'a': np.zeros((1, 5)),
        'b': np.zeros(2),
        'c': np.zeros((3, 2))
    }
    darray = np.array([0, 1, 2, 3, 4, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1])
    for key, val in list(restruct(darray, dzeros).items()):
        assert np.allclose(dref[key], val), "Dict restruct"

    # List
    lref = [np.eye(2), np.array([-1, 1]), np.array([[10, 20], [30, 40]])]
    larray = np.array([1, 0, 0, 1, -1, 1, 10, 20, 30, 40])
    lzeros = [np.zeros((2, 2)), np.zeros(2), np.zeros((2, 2))]
    for idx, val in enumerate(restruct(larray, lzeros)):
        assert np.allclose(lref[idx], val), "List restruct"

    # Tuple
    for idx, val in enumerate(restruct(larray, tuple(lzeros))):
        assert np.allclose(lref[idx], val), "Tuple restruct"


@timed(1.1)
def test_lrucache():
    """Tests the lrucache decorator"""

    def slowfunc(x):
        """A fake slow function (for testing lrucache)"""
        sleep(1)
        return x**2

    # cached function memoizes the last call
    cachedfunc = lrucache(slowfunc, 1)

    # first call will be slow (1 second)
    y1 = cachedfunc(2)

    # second call should be fast (dictionary lookup)
    y2 = cachedfunc(2)

    # and both results should be the same
    assert y1 == y2, "Cached return values must match"
