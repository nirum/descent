"""
Test objective functions
"""
import numpy as np
import pytest
from descent import algorithms
from descent import objectives 
from descent.utils import check_grad


@pytest.mark.parametrize("function", (getattr(objectives, f) for f in objectives.__all__))
def test_gradient(function, tol=1e-2):
    x0 = function.param_init()
    assert check_grad(function, x0) == 0


@pytest.mark.parametrize("function", (getattr(objectives, f) for f in objectives.__all__))
def test_minima(function, tol=1e-2):
    if function.xstar is not None:
        assert np.allclose(np.zeros(function.ndim,), function(function.xstar)[1])
