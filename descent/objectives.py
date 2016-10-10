"""
Example objectives
"""
import numpy as np

__all__ = ['rosenbrock', 'sphere', 'matyas', 'ackley', 'beale', 'booth', 'mccormick']


def rosenbrock(theta):
    """Objective and gradient for the rosenbrock function"""

    x, y = theta
    obj = (1 - x)**2 + 100 * (y - x**2)**2

    grad = np.zeros(2)
    grad[0] = 2 * x - 400 * (x * y - x**3) - 2
    grad[1] = 200 * (y - x**2)
    return obj, grad


def sphere(theta):
    """l2-norm of the parameters"""
    return 0.5 * np.linalg.norm(theta)**2, theta


def matyas(theta):
    """Matyas function"""
    x, y = theta
    obj = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    grad = np.array([0.52 * x - 0.48 * y, 0.52 * y - 0.48 * x])
    return obj, grad


def ackley(theta):
    """Ackley's function"""
    x, y = theta
    A = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    B = np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    obj = A - B + np.exp(1.) + 20.

    # grad = np.array([A * -0.2 * 0.5 * 0.5 * (x ** 2 + y ** 2) * 2 * x - 
    grad = np.array([
        A * (-0.2 * np.sqrt(0.5) * (x ** 2 + y ** 2)**(-0.5) * x) + B * np.pi * np.sin(2 * np.pi * x),
        A * (-0.2 * np.sqrt(0.5) * (x ** 2 + y ** 2)**(-0.5) * y) + B * np.pi * np.sin(2 * np.pi * y),
    ])
    return obj, grad


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


def booth(theta):
    """Booth's function"""
    x, y = theta

    A = x + 2 * y - 7
    B = 2 * x + y - 5
    obj = A**2 + B**2
    grad = np.array([2 * A + 4 * B, 4 * A + 2 * B])
    return obj, grad


def mccormick(theta):
    """McCormick function"""
    x, y = theta
    obj = np.sin(x + y) + (x - y)**2 - 1.5 * x + 2.5 * y + 1
    grad = np.array([np.cos(x + y) + 2 * (x - y) - 1.5,
                     np.cos(x + y) - 2 * (x - y) + 2.5])
    return obj, grad
