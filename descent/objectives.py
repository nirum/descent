"""
Example objectives
"""
import numpy as np
from functools import wraps

__all__ = ['rosenbrock', 'sphere', 'matyas', 'beale', 'booth', 'mccormick',
           'camel', 'michalewicz', 'bohachevsky1', 'zakharov', 'dixon_price']


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


@objective(xstar=(1, 1))
def rosenbrock(theta):
    """Objective and gradient for the rosenbrock function"""

    x, y = theta
    obj = (1 - x)**2 + 100 * (y - x**2)**2

    grad = np.zeros(2)
    grad[0] = 2 * x - 400 * (x * y - x**3) - 2
    grad[1] = 200 * (y - x**2)
    return obj, grad


@objective(xstar=(0, 0))
def sphere(theta):
    """l2-norm of the parameters"""
    return 0.5 * np.linalg.norm(theta)**2, theta


@objective(xstar=(0, 0))
def matyas(theta):
    """Matyas function"""
    x, y = theta
    obj = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    grad = np.array([0.52 * x - 0.48 * y, 0.52 * y - 0.48 * x])
    return obj, grad


@objective(xstar=(3, 0.5))
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


@objective(xstar=(1, 3))
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


@objective(xstar=(0, 0))
def camel(theta):
    """Three-hump camel function"""
    x, y = theta
    obj = 2 * x ** 2 - 1.05 * x ** 4 + x ** 6 / 6 + x * y + y ** 2
    grad = np.array([
        4 * x - 4.2 * x ** 3 + x ** 5 + y,
        x + 2 * y
    ])
    return obj, grad


@objective(xstar=(2.2029055201726027, 1.5707963267948954))
def michalewicz(theta):
    x, y = theta
    obj = - np.sin(x) * np.sin(x ** 2 / np.pi) ** 20 - \
        np.sin(y) * np.sin(2 * y ** 2 / np.pi) ** 20

    grad = np.array([
        - np.cos(x) * np.sin(x ** 2 / np.pi) ** 20 - (40 / np.pi) * x * np.sin(x) * np.sin(x ** 2 / np.pi) ** 19 * np.cos(x ** 2 / np.pi),
        - np.cos(y) * np.sin(2 * y ** 2 / np.pi) ** 20 - (80 / np.pi) * y * np.sin(y) * np.sin(2 * y ** 2 / np.pi) ** 19 * np.cos(2 * y ** 2 / np.pi),
    ])

    return obj, grad


@objective(xstar=(0, 0))
def bohachevsky1(theta):
    """One of the Bohachevsky functions"""
    x, y = theta
    obj = x ** 2 + 2 * y ** 2 - 0.3 * np.cos(3 * np.pi * x) - 0.4 * np.cos(4 * np.pi * y) + 0.7
    grad = np.array([
        2 * x + 0.3 * np.sin(3 * np.pi * x) * 3 * np.pi,
        4 * y + 0.4 * np.sin(4 * np.pi * y) * 4 * np.pi,
    ])
    return obj, grad


@objective(xstar=(0, 0))
def zakharov(theta):
    """Zakharov function"""
    x, y = theta
    obj = x ** 2 + y ** 2 + (0.5 * x + y) ** 2 + (0.5 * x + y) ** 4
    grad = np.array([
        2.5 * x + y + 2 * (0.5 * x + y) ** 3,
        4 * y + x + 4 * (0.5 * x + y) ** 3,
    ])
    return obj, grad


@objective(xstar=(1, 1/np.sqrt(2)))
def dixon_price(theta):
    """Dixon-Price function"""
    x, y = theta
    obj = (x - 1) ** 2 + 2 * (2 * y ** 2 - x) ** 2
    grad = np.array([
        2 * x - 2 - 4 * (2 * y ** 2 - x),
        16 * (2 * y ** 2 - x) * y,
    ])
    return obj, grad


@objective(xstar=(0, -1))
def goldstein_price(theta):
    """Goldstein-Price function"""
    x, y = theta
    obj = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
          (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * x ** 2))
    grad = np.array([
        ((2*x - 3*y)**2*(78*x - 36*y - 32) + (8*x - 12*y)*(39*x**2 - 36*x*y - 32*x + 48*y + 18))*((x + y + 1)**2*(3*x**2 + 6*x*y - 14*x + 3*y**2 - 14*y + 19) + 1) + ((2*x - 3*y)**2*(39*x**2 - 36*x*y - 32*x + 48*y + 18) + 30)*((x + y + 1)**2*(6*x + 6*y - 14) + (2*x + 2*y + 2)*(3*x**2 + 6*x*y - 14*x + 3*y**2 - 14*y + 19)),
        ((-36*x + 48)*(2*x - 3*y)**2 + (-12*x + 18*y)*(39*x**2 - 36*x*y - 32*x + 48*y + 18))*((x + y + 1)**2*(3*x**2 + 6*x*y - 14*x + 3*y**2 - 14*y + 19) + 1) + ((2*x - 3*y)**2*(39*x**2 - 36*x*y - 32*x + 48*y + 18) + 30)*((x + y + 1)**2*(6*x + 6*y - 14) + (2*x + 2*y + 2)*(3*x**2 + 6*x*y - 14*x + 3*y**2 - 14*y + 19)),
    ])
    return obj, grad


@objective(xstar=(-2.903534, -2.903534))
def styblinski_tang(theta):
    """Styblinski-Tang function"""
    x, y = theta
    obj = 0.5 * (x ** 4 - 16 * x ** 2 + 5 * x + y ** 4 - 16 * y ** 2 + 5 * y)
    grad = np.array([
        2 * x ** 3 - 16 * x + 2.5,
        2 * y ** 3 - 16 * y + 2.5,
    ])
    return obj, grad
