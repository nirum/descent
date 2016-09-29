"""
Example objectives
"""
import numpy as np

__all__ = ['rosenbrock', 'sphere', 'matyas']


def rosenbrock(theta):
    """Objective and gradient for the rosenbrock function"""

    x, y = theta
    obj = (1 - x)**2 + 100 * (y - x**2)**2

    grad = np.zeros(2)
    grad[0] = 2 * x - 400 * (x * y - x**3) - 2
    grad[1] = 200 * (y - x**2)
    return obj, grad


def sphere(theta):
    return 0.5 * np.linalg.norm(theta)**2, theta


def matyas(theta):
    x, y = theta
    obj = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    grad = np.array([0.52 * x - 0.48 * y, 0.52 * y - 0.48 * x])
    return obj, grad
