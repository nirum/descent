"""
Test optimization of the rosenbrock function
"""

import numpy as np
from descent.algorithms import gd
from descent.main import loop
from descent.utils import wrap


def rosenbrock(theta):

    x = theta[0]
    y = theta[1]

    # Rosenbrock's banana function
    obj = (1-x)**2 + 100*(y-x**2)**2

    # gradient for the Rosenbrock function
    grad = np.zeros(2)
    grad[0] = 2*x - 400*(x*y - x**3) - 2
    grad[1] = 200*(y-x**2)

    return obj, grad


def test_rosen(tol=5e-2):
    """Test rosenbrock"""

    obj, grad = wrap(rosenbrock)

    xstar = np.array([1, 1])
    assert np.all(grad(xstar) == 0)

    xhat = loop(gd(eta=1e-3), grad, np.zeros(2), maxiter=10000)
    assert np.linalg.norm(xhat-xstar) <= tol
