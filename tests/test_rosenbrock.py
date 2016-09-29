"""
Test optimization of the rosenbrock function
"""
import numpy as np
import pytest
from descent import algorithms
from descent.objectives import rosenbrock


@pytest.mark.parametrize("algorithm,options", [
  ('sgd', {'lr': 1e-3, 'mom': 0.1}),
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
    opt = getattr(algorithms, algorithm)(**options)
    
    res = opt.minimize(rosenbrock, np.zeros(2,), display=None, maxiter=1e4)
    # run the optimization algorithm
    #opt.run(maxiter=1e4)
    assert np.linalg.norm(res.x - xstar) <= tol
