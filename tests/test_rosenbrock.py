"""
Test optimization of the rosenbrock function
"""

import numpy as np
from descent import sgd, rmsprop, adam


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


def test_rosen(tol=1e-2):
    """Test minimization of the rosenbrock function"""

    # check that the gradient is zeros at the optimal point
    xstar = np.array([1, 1])
    assert np.all(rosenbrock(xstar)[1] == 0)

    # list of algorithms to test (and their parameters)
    algorithms = [(sgd, {'learning_rate': 1e-3, 'momentum': 0.1}),
                  (rmsprop, {'learning_rate': 1e-3}),
                  (adam, {'learning_rate': 1e-3})]

    # loop over algorithms
    for algorithm, kwargs in algorithms:

        # initialize
        optimizer = algorithm(rosenbrock, np.zeros(2), **kwargs)
        optimizer.display = None
        optimizer.storage = None

        # run the optimization algorithm
        optimizer.run(maxiter=1e4)
        assert np.linalg.norm(optimizer.theta - xstar) <= tol
