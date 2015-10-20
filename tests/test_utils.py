"""
Test conversion utilities
"""

import numpy as np
from descent.utils import destruct, restruct, lrucache, check_grad
from six import StringIO
from time import sleep, time


def test_lrucache():
    """Tests the lrucache decorator"""

    def slowfunc(x):
        """A fake slow function (for testing lrucache)"""
        sleep(1)
        return x**2

    # cached function memoizes the last call
    cachedfunc = lrucache(slowfunc, 1)

    # first call will be slow (1 second)
    start = time()
    y1 = cachedfunc(np.array(2))
    call1 = time() - start

    # second call should be fast (dictionary lookup)
    start = time()
    y2 = cachedfunc(np.array(2))
    call2 = time() - start

    # assert timing results
    assert call1 >= 1.0, "First call takes at least a second"
    assert call2 <= 1e-3, "Second call is just a dictionary lookup"

    # and both results should be the same
    assert y1 == y2, "Cached return values must match"


def test_check_grad():
    """Tests the check_grad() function"""

    def f_df_correct(x):
        return x**2, 2*x

    def f_df_incorrect(x):
        return x**3, 0.5*x**2

    output = StringIO()
    check_grad(f_df_correct, 5, out=output)

    # helper functions
    getvalues = lambda o: [float(s.strip()) for s in \
                           o.getvalue().split('\n')[3].split('|')[:-1]]

    # get the first row of data
    values = getvalues(output)
    assert values[0] == values[1] == 10.0, "Correct gradient computation"
    assert values[2] == 0.0, "Correct error computation"

    output = StringIO()
    check_grad(f_df_incorrect, 5, out=output)
    values = getvalues(output)
    printed_error = values[2]
    correct_error = np.abs(values[0] - values[1]) \
                    / (np.abs(values[0]) + np.abs(values[1]))
    assert np.isclose(printed_error, correct_error), "Correct relative error"


def test_destruct():
    """Tests the destruct utility function"""

    # numpy ndarray
    aref = np.linspace(0, 1, 30).reshape(2, 5, 3).T
    assert np.allclose(aref.ravel(), destruct(aref)), "Numpy ndarray destruct"

    # Numeric
    assert np.array(2.0) == destruct(2), "Integer destruct"
    assert np.array(-5.0) == destruct(-5.0), "Float destruct"

    # Dictionary
    dref = {
        'a': np.arange(5).reshape(1, 5),
        'c': np.linspace(0, 1, 6).reshape(3, 2),
        'b': np.ones(2)
    }
    darray = np.array([0, 1, 2, 3, 4, 1, 1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    assert np.allclose(darray, destruct(dref)), "Dictionary destruct"

    # List
    lref = [np.eye(2), np.array([-1, 1]), np.array([[10, 20], [30, 40]])]
    larray = np.array([1, 0, 0, 1, -1, 1, 10, 20, 30, 40])
    assert np.allclose(larray, destruct(lref)), "List destruct"

    # Tuple
    assert np.allclose(larray, destruct(tuple(lref))), "Tuple destruct"

    # Composed / mixed types
    lref = [np.eye(2), {'a': np.array([-1,1]), 'b': np.arange(3)}, 7]
    larray = np.array([1, 0, 0, 1, -1, 1, 0, 1, 2, 7])
    assert np.allclose(larray, destruct(lref)), "List of mixed types"

    dref = {
        'a': [3.5, 1.2, -33],
        'b': np.arange(6).reshape(2,3),
        'c': 7
    }
    darray = np.array([3.5, 1.2, -33, 0, 1, 2, 3, 4, 5, 7])
    assert np.allclose(darray, destruct(dref)), "Dictionary of mixed types"

def test_restruct():
    """Tests the destruct utility function"""

    # numpy ndarray
    aref = np.linspace(0, 1, 30).reshape(2, 5, 3).T
    assert np.allclose(restruct(aref.ravel(), np.zeros((3, 5, 2))), aref), \
        "Numpy ndarray restruct"

    # Numeric (preserves floating points)
    assert restruct(np.array(2.2), 0) == 2.2, "Integer restruct"
    assert restruct(np.array(-5.1), 0.0) == -5.1, "Float restruct"

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
    darray = np.array([0, 1, 2, 3, 4, 1, 1, 0, 0.2, 0.4, 0.6, 0.8, 1])
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

    # Composed / mixed types
    lref = [np.eye(2), {'a': np.array([-1,1]), 'b': np.arange(3)}, 7]
    lzeros = [np.zeros((2,2)), {'a': np.zeros(2), 'b': np.zeros(3)}, 0]
    larray = np.array([1, 0, 0, 1, -1, 1, 0, 1, 2, 7])
    assert np.allclose(destruct(restruct(larray, lzeros)), destruct(lref)), "List of mixed types"

    dref = {
        'a': [3.5, 1.2, -33],
        'b': np.arange(6).reshape(2,3),
        'c': 7
    }
    dzeros = {
        'a': [0, 0, 0],
        'b': np.zeros((2,3)),
        'c': 0
    }
    darray = np.array([3.5, 1.2, -33, 0, 1, 2, 3, 4, 5, 7])
    assert np.allclose(destruct(restruct(darray, dzeros)), destruct(dref)), "Dictionary of mixed types"
