"""
Test suite for sparse regression
"""
import numpy as np
from descent import algorithms, Consensus
from descent.proxops import sparse, linsys


def generate_sparse_system(n=100, m=50, p=0.1, eta=0.05, seed=1234):
    """Generate a sparse, noisy system (Ax = y)

    Parameters
    ----------
    n : int
        Number of variables

    m : int
        Number of observations

    p : float
        Probability of non-zero variable

    eta : float
        Noise level (standard deviation of additive Gaussian noise)

    seed : int
        Random seed

    Returns
    -------
    A : array_like
        Sensing matrix (m x n)

    y : array_like
        Observations (m,)

    x_true : array_like
        True sparse signal (n,)

    xls : array_like
        Least squares solution (n,)

    ls_error : float
        Error (2-norm) of the least squares solution
    """

    global x_true, x_obs, A

    # define the seed
    np.random.seed(seed)

    # the true sparse signal
    x_true = 10 * np.random.randn(n) * (np.random.rand(n) < p)

    # the noisy, observed signal
    A = np.random.randn(m, n)
    y = A.dot(x_true)

    # least squares solution
    xls = np.linalg.lstsq(A, y)[0]

    # least squares error
    ls_error = np.linalg.norm(xls - x_true, 2)

    return A, y, x_true, xls, ls_error


def relative_error(xhat, x_true, baseline):
    test_err = np.linalg.norm(xhat - x_true, 2)
    return test_err / baseline


def test_projected_gradient_descent():
    """Test sparse regression"""

    A, y, x_true, xls, ls_error = generate_sparse_system()

    # helper function to test relative error
    def test_error(xhat):
        assert relative_error(xhat, x_true, ls_error) <= 0.01

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
        opt.add(sparse(10.0))

        # run it
        res = opt.minimize(f_df, xls, display=None, maxiter=5000)

        # test
        test_error(res.x)


def test_consensus():
    """Test the consensus optimizer (ADMM)"""

    A, y, x_true, xls, ls_error = generate_sparse_system()

    # optimizer
    opt = Consensus()
    opt.add(linsys(A, y))
    opt.add(sparse(10.0))

    # run it
    res = opt.minimize(xls, display=None, maxiter=5000)

    # test
    assert relative_error(res.x, x_true, ls_error) <= 0.05
