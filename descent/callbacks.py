"""
Callback functions

callback(k, f, df, xk)
"""


def merge(callbacks):
    def wrapper(*args, **kwargs):
        for cb in callbacks:
            cb(*args, **kwargs)
    return wrapper


class Store:
    def __init__(self):
        self.obj = []
        self.gradients = []
        self.params = []

    def __call__(self, k, f, df, xk):
        self.obj.append(f)
        self.gradients.append(df.copy())
        self.params.append(xk.copy())
