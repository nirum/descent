"""
Test suite for sparse regression
"""
import numpy as np
from descent import algorithms
from descent.proxops import sparse


def generate_sparse_system(n=100, m=50, p=0.1, eta=0.05, seed=1234):
    """
    Generate a sparse, noisy system

    """
    print("Generating data for sparse regression")
    global x_true, x_obs, A

    # define the seed
    np.random.seed(seed)

    # the true sparse signal
    x_true = 10 * np.random.randn(n) * (np.random.rand(n) < p)

    # the noisy, observed signal
    A = np.random.randn(m, n)
    y = A.dot(x_true)

    return A, y, x_true


def test_sparse_regression():
    """
    Test sparse regression

    """

    A, y, x_true = generate_sparse_system()

    # least squares solution
    xls = np.linalg.lstsq(A, y)[0]

    # helper function to test relative error
    def test_error(xhat):
        test_err = np.linalg.norm(xhat - x_true, 2)
        naive_err = np.linalg.norm(xls - x_true, 2)
        err_ratio = test_err / naive_err
        assert err_ratio <= 0.01

    # Proximal gradient descent and Accelerated proximal gradient descent
    for algorithm in ['sgd', 'nag']:

        # objective
        def f_df(x):
            err = A.dot(x) - y
            obj = 0.5 * np.linalg.norm(err) ** 2
            grad = A.T.dot(err)
            return obj, grad

        # optimizer
        opt = getattr(algorithms, algorithm)(lr=5e-3)
        opt.operators.append(sparse(10.0))

        # run it
        res = opt.minimize(f_df, xls, display=None, maxiter=5000)

        # test
        test_error(res.x)
