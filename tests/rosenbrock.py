"""
Rosenbrock function and gradient
"""

import numpy as np


def f_df(theta):

    x = theta[0]
    y = theta[1]

    # Rosenbrock's banana function
    obj = (1-x)**2 + 100*(y-x**2)**2

    # gradient for the Rosenbrock function
    grad = np.zeros(2)
    grad[0] = 2*x - 400*(x*y - x**3) -2
    grad[1] = 200*(y-x**2)

    return obj, grad
