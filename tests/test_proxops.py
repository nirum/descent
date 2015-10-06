"""
Test proximal operators
"""

from descent import proxops
import numpy as np


def test_nucnorm():

    pen = 1.
    rho = 0.1
    tol = 1.

    op = proxops.nucnorm(pen)

    X = 2*np.outer(np.random.randn(50), np.random.randn(25))
    V = X + 0.5 * np.random.randn(50,25)

    nn = lambda A: np.linalg.svd(A, compute_uv=False).sum()

    assert np.abs(nn(X) - nn(op(X, rho)) - pen / rho) <= tol


def test_sparse():

    pen = 0.1
    rho = 0.1
    v = np.linspace(-5, 5, 1e3)

    gamma = pen / rho
    x = (v - gamma * np.sign(v)) * (np.abs(v) > gamma)

    op = proxops.sparse(pen)

    assert np.allclose(op(v, rho), x)


def test_nonneg():

    op = proxops.nonneg()

    v = np.array([-2., 0.54, -0.2, 24.])
    x = np.array([0., 0.54, 0., 24.])

    assert np.allclose(op(v, 1.), x)


def test_linsys():

    A = np.random.randn(50,20)
    x = np.random.randn(20,)
    y = A.dot(x)

    op = proxops.linsys(A, y)

    assert np.allclose(op(x, 1.), x)
    assert np.allclose(op(np.zeros(x.size), 0.), x)


def test_squared_error():

    xobs = np.array([-2., -1., 0., 1., 2.])
    tol = 1e-5

    op = proxops.squared_error(xobs)

    assert np.allclose(op(np.zeros(5), 1.), np.array([-1., -0.5, 0., 0.5, 1.]))
    assert np.allclose(op(xobs, 10.), xobs)
    assert np.linalg.norm(op(np.zeros(5), 1e-6) - xobs) <= tol
