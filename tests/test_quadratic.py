"""
Test optimization of a quadratic function
"""

import numpy as np
from descent.algorithms import gd
from descent.utils import wrap


def f_df(x):
    return 0.5 * x.T.dot(x), x


def test_fixedpoint():
    xstar = np.array([0, 0])
    obj, grad = wrap(f_df)
    opt = gd(grad, xstar)
    assert np.allclose(next(opt), xstar)
