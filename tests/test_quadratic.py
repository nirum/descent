"""
Test optimization of a quadratic function
"""

import numpy as np
from descent.algorithms import gdm
from descent.utils import wrap


def f_df(x):
    return 0.5 * x.T.dot(x), x


def test_fixedpoint():
    """Test that the minimum of a quadratic is a fixed point"""

    xstar = np.array([0, 0])
    obj, grad = wrap(f_df)
    opt = gdm(grad, xstar)
    assert np.allclose(next(opt), xstar)
