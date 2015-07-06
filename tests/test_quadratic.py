"""
Test optimization of a quadratic function
"""

import numpy as np
from descent.algorithms import gd, agd, loop
from descent.utils import splitf


def f_df(x):
    return 0.5 * x.T.dot(x), x


def test_fixedpoint():
    xstar = np.array([0, 0])
    obj, grad = splitf(f_df)
    opt = gd(grad, xstar)
    assert np.allclose(next(opt), xstar)
