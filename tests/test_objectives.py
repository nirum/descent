"""
Test objective functions
"""
import numpy as np
import pytest
from descent import algorithms
from descent import objectives 
from descent.utils import check_grad


@pytest.mark.parametrize("function", (getattr(objectives, f) for f in objectives.__all__))
def test_rosen(function, tol=1e-2):
    """Test minimization of the rosenbrock function"""
    x0 = np.random.randn(2,)
    assert check_grad(function, x0) == 0
