"""
Test optimization of the rosenbrock function
"""

import numpy as np
from descent.algorithms import gdm
from descent.main import loop


def rosenbrock(theta):
    """Objective and gradient for the rosenbrock function"""

    x = theta[0]
    y = theta[1]

    # Rosenbrock's banana function
    obj = (1-x)**2 + 100*(y-x**2)**2

    # gradient for the Rosenbrock function
    grad = np.zeros(2)
    grad[0] = 2*x - 400*(x*y - x**3) - 2
    grad[1] = 200*(y-x**2)

    return obj, grad


def test_rosen(tol=5e-3):
    """Test minimization of the rosenbrock function"""

    # check that the gradient is zeros at the optimal point
    xstar = np.array([1, 1])
    assert np.all(rosenbrock(xstar)[1] == 0)

    # run gradient descent with momentum
    xhat = loop(gdm(eta=1e-3, mu=0.2), rosenbrock, np.zeros(2), maxiter=100000)
    assert np.linalg.norm(xhat-xstar) <= tol
