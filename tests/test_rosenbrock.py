"""
Test optimization of the rosenbrock function
"""

from __future__ import (absolute_import, division, print_function)
import numpy as np
from descent import GradientDescent
from nose_parameterized import parameterized


def rosenbrock(theta):
    """Objective and gradient for the rosenbrock function"""

    x = theta[0]
    y = theta[1]

    # Rosenbrock's banana function
    obj = (1 - x)**2 + 100 * (y - x**2)**2

    # gradient for the Rosenbrock function
    grad = np.zeros(2)
    grad[0] = 2 * x - 400 * (x * y - x**3) - 2
    grad[1] = 200 * (y - x**2)

    return obj, grad


@parameterized([
    ('sgd', {'lr': 1e-3, 'momentum': 0.1}),
    ('nag', {'lr': 1e-3}),
    ('rmsprop', {'lr': 1e-3}),
    ('adam', {'lr': 1e-3}),
    ('smorms', {'lr': 1e-3}),
    ('sag', {'nterms': 2, 'lr': 2e-3}),
])
def test_rosen(algorithm, options, tol=1e-2):
    """Test minimization of the rosenbrock function"""

    # check that the gradient is zeros at the optimal point
    xstar = np.array([1, 1])
    assert np.all(rosenbrock(xstar)[1] == 0)

    # initialize
    opt = GradientDescent(np.zeros(2), rosenbrock, algorithm, options)

    # run the optimization algorithm
    opt.run(maxiter=1e4)
    assert np.linalg.norm(opt.theta - xstar) <= tol
