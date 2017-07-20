"""
Example objectives
"""
import autograd.numpy as np
from autograd.convenience_wrappers import value_and_grad

__all__ = ['doublewell', 'rosenbrock', 'sphere', 'matyas', 'beale', 'booth', 'mccormick',
           'camel', 'michalewicz', 'bohachevsky1', 'zakharov', 'dixon_price', 'monkey_saddle',
           'hyperbolic_paraboloid']


@value_and_grad
def doublewell(theta):
    """Pointwise minimum of two quadratic bowls"""
    k0, k1, depth = 0.01, 100, 0.5
    shallow = 0.5 * k0 * theta ** 2 + depth
    deep = 0.5 * k1 * theta ** 2
    return float(np.minimum(shallow, deep))


@value_and_grad
def monkey_saddle(theta):
    """Monkey saddle"""
    x, y = theta
    return x ** 3 - 3 * x * y ** 2


@value_and_grad
def hyperbolic_paraboloid(theta):
    """Hyperbolic paraboloid (pringle chip)"""
    x, y = theta
    return 0.5 * (x ** 2 - y ** 2)


@value_and_grad
def rosenbrock(theta):
    """Objective and gradient for the rosenbrock function"""
    x, y = theta
    return (1 - x)**2 + 100 * (y - x**2)**2


def sphere(theta):
    """l2-norm of the parameters"""
    return 0.5 * np.linalg.norm(theta)**2, theta


@value_and_grad
def matyas(theta):
    """Matyas function"""
    x, y = theta
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


@value_and_grad
def beale(theta):
    """Beale's function"""
    x, y = theta
    A = 1.5 - x + x * y
    B = 2.25 - x + x * y**2
    C = 2.625 - x + x * y**3
    return A ** 2 + B ** 2 + C ** 2


@value_and_grad
def booth(theta):
    """Booth's function"""
    x, y = theta

    A = x + 2 * y - 7
    B = 2 * x + y - 5
    return A**2 + B**2


@value_and_grad
def mccormick(theta):
    """McCormick function"""
    x, y = theta
    obj = np.sin(x + y) + (x - y)**2 - 1.5 * x + 2.5 * y + 1
    grad = np.array([np.cos(x + y) + 2 * (x - y) - 1.5,
                     np.cos(x + y) - 2 * (x - y) + 2.5])
    return obj, grad


@value_and_grad
def camel(theta):
    """Three-hump camel function"""
    x, y = theta
    return 2 * x ** 2 - 1.05 * x ** 4 + x ** 6 / 6 + x * y + y ** 2


@value_and_grad
def michalewicz(theta):
    """Michalewicz function"""
    x, y = theta
    return -np.sin(x) * np.sin(x ** 2 / np.pi) ** 20 - np.sin(y) * np.sin(2 * y ** 2 / np.pi) ** 20


@value_and_grad
def bohachevsky1(theta):
    """One of the Bohachevsky functions"""
    x, y = theta
    return x ** 2 + 2 * y ** 2 - 0.3 * np.cos(3 * np.pi * x) - 0.4 * np.cos(4 * np.pi * y) + 0.7


@value_and_grad
def zakharov(theta):
    """Zakharov function"""
    x, y = theta
    return x ** 2 + y ** 2 + (0.5 * x + y) ** 2 + (0.5 * x + y) ** 4


@value_and_grad
def dixon_price(theta):
    """Dixon-Price function"""
    x, y = theta
    return (x - 1) ** 2 + 2 * (2 * y ** 2 - x) ** 2


@value_and_grad
def goldstein_price(theta):
    """Goldstein-Price function"""
    x, y = theta
    return (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
           (30 + (2 * x - 3 * y) ** 2 *
            (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * x ** 2))


@value_and_grad
def styblinski_tang(theta):
    """Styblinski-Tang function"""
    x, y = theta
    return 0.5 * (x ** 4 - 16 * x ** 2 + 5 * x + y ** 4 - 16 * y ** 2 + 5 * y)
