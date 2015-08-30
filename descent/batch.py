"""
Playing around with minibatch generation (WIP)
"""

from functools import wraps
import numpy as np


def batchgen(indices, data, batch_size):
    """
    Generates random batches of data of a given size
    """
    assert batch_size <= len(indices)
    while True:
        yield data[np.random.choice(indices, size=batch_size, replace=False)]


def minibatchify(gen):
    """
    Takes an objective fun. and a data generator and returns an noisy oracle
    """

    def decorate(f_df):

        @wraps(f_df)
        def wrapper(theta):
            return f_df(theta, next(gen))

        return wrapper

    return decorate


# Example use
N = 1000
m = 50
train_data = np.random.randn(N)*0.5 + np.pi
g = batchgen(np.arange(N), train_data, m)


@minibatchify(g)
def f_df(theta, data):
    return np.mean(0.5*(theta-data)**2), np.mean(theta-data)
