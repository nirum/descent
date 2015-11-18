"""
Test optimization of a quadratic function
"""

import numpy as np
from descent import Optimizer
from descent.utils import wrap
from descent.algorithms import sgd


def test_fixedpoint():
    """
    Test that the minimum of a quadratic is a fixed point
    """

    def grad(x):
        return x

    xstar = np.array([0., 0.])
    coro = sgd(xstar)
    xinit = coro.send(None)
    xnext = coro.send(grad(xinit))

    assert np.allclose(xnext, xstar)


def test_quadratic_bowl():
    """
    Test optimization in a quadratic bowl
    """

    t = np.linspace(0, 2*np.pi, 100)
    tol = 1e-3

    theta_true = [np.sin(t), np.cos(t)]
    theta_init = [np.cos(t), np.sin(t)]

    def f_df(theta):
        obj = 0.5*(theta[0]-theta_true[0])**2 + 0.5*(theta[1]-theta_true[1])**2
        grad = [theta[0]-theta_true[0], theta[1]-theta_true[1]]
        return np.sum(obj), grad

    opt = Optimizer(f_df, theta_init, 'sgd', learning_rate=1e-2)
    opt.display = None
    opt.run(maxiter=1e3)

    for theta in zip(opt.theta, theta_true):
        assert np.linalg.norm(theta[0] - theta[1]) <= tol
