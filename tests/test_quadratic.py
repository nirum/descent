"""
Test optimization of a quadratic function
"""

import numpy as np
from descent.algorithms import gdm
from descent.utils import wrap
from descent.main import optimize


def test_fixedpoint():
    """Test that the minimum of a quadratic is a fixed point"""

    def f_df(x):
        return 0.5 * x.T.dot(x), x

    xstar = np.array([0., 0.])
    obj, grad = wrap(f_df, xstar)
    opt = gdm(grad, xstar, 10)
    assert np.allclose(next(opt), xstar)


def test_destruct():
    """Test that the destruct/restruct wrapping works"""

    t = np.linspace(0, 2*np.pi, 100)
    tol = 1e-3

    theta_true = [np.sin(t), np.cos(t)]
    theta_init = [np.cos(t), np.sin(t)]

    def f_df(theta):
        obj = 0.5*(theta[0]-theta_true[0])**2 + 0.5*(theta[1]-theta_true[1])**2
        grad = [theta[0]-theta_true[0], theta[1]-theta_true[1]]
        return np.sum(obj), grad

    theta_hat = optimize(gdm, f_df, theta_init, maxiter=1e3)

    for theta in zip(theta_hat, theta_true):
        assert np.linalg.norm(theta[0] - theta[1]) <= tol
