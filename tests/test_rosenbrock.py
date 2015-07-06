"""
Test optimization of the rosenbrock function
"""

import numpy as np
from rosenbrock import f_df
from descent.algorithms import gd, loop


def test_rosen():
    """Test rosenbrock"""

    xstar = np.ones(2)
    tol = 5e-2

    xhat = loop(gd(eta=1e-3), f_df, np.zeros(2), maxiter=10000)

    assert np.allclose(xhat, xstar, atol=tol)
