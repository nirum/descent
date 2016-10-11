"""
Example objectives
"""
import numpy as np
from functools import wraps

__all__ = ['rosenbrock', 'sphere', 'matyas', 'beale', 'booth', 'mccormick']


def objective(param_scales=(1, 1), xstar=None, seed=None):
    """Gives objective functions a number of dimensions and parameter range

    Parameters
    ----------
    param_scales : (int, int)
        Scale (std. dev.) for choosing each parameter

    xstar : array_like
        Optimal parameters
    """
    ndim = len(param_scales)

    def decorator(func):

        @wraps(func)
        def wrapper(theta):
            return func(theta)

        def param_init():
            np.random.seed(seed)
            return np.random.randn(ndim,) * np.array(param_scales)

        wrapper.ndim = ndim
        wrapper.param_init = param_init
        wrapper.xstar = xstar

        return wrapper

    return decorator


@objective(xstar=(1., 1.))
def rosenbrock(theta):
    """Objective and gradient for the rosenbrock function"""

    x, y = theta
    obj = (1 - x)**2 + 100 * (y - x**2)**2

    grad = np.zeros(2)
    grad[0] = 2 * x - 400 * (x * y - x**3) - 2
    grad[1] = 200 * (y - x**2)
    return obj, grad


@objective(xstar=(0., 0.))
def sphere(theta):
    """l2-norm of the parameters"""
    return 0.5 * np.linalg.norm(theta)**2, theta


@objective(xstar=(0., 0.))
def matyas(theta):
    """Matyas function"""
    x, y = theta
    obj = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    grad = np.array([0.52 * x - 0.48 * y, 0.52 * y - 0.48 * x])
    return obj, grad


@objective(xstar=(3., 0.5))
def beale(theta):
    """Beale's function"""
    x, y = theta
    A = 1.5 - x + x * y
    B = 2.25 - x + x * y**2
    C = 2.625 - x + x * y**3
    obj = A ** 2 + B ** 2 + C ** 2
    grad = np.array([
        2 * A * (y - 1) + 2 * B * (y ** 2 - 1) + 2 * C * (y ** 3 - 1),
        2 * A * x + 4 * B * x * y + 6 * C * x * y ** 2
    ])
    return obj, grad


@objective(xstar=(1., 3.))
def booth(theta):
    """Booth's function"""
    x, y = theta

    A = x + 2 * y - 7
    B = 2 * x + y - 5
    obj = A**2 + B**2
    grad = np.array([2 * A + 4 * B, 4 * A + 2 * B])
    return obj, grad


@objective(xstar=(-0.5471975511965975, -1.5471975511965975))
def mccormick(theta):
    """McCormick function"""
    x, y = theta
    obj = np.sin(x + y) + (x - y)**2 - 1.5 * x + 2.5 * y + 1
    grad = np.array([np.cos(x + y) + 2 * (x - y) - 1.5,
                     np.cos(x + y) - 2 * (x - y) + 2.5])
    return obj, grad


@objective(xstar=(0., 0.))
def camel(theta):
    """Three-hump camel function"""
    x, y = theta
    obj = 2 * x ** 2 - 1.05 * x ** 4 + x ** 6 / 6 + x * y + y ** 2
    grad = np.array([
        4 * x - 4.2 * x ** 3 + x ** 5 + y,
        x + 2 * y
    ])
    return obj, grad
