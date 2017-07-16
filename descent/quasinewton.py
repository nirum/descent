"""
Quasi-Newton methods
"""
from collections import defaultdict
import numpy as np


def bfgs(f_df, x0, alpha=1e-3, niter=100):
    """BFGS Algorithm"""

    # initial inverse Hessian
    H = np.eye(x0.size)

    # iterate
    x = x0.copy()

    # initial objective and gradient
    fk, dfk = f_df(x)

    # storate
    store = defaultdict(list)
    store['obj'].append(fk)
    store['gradnorm'].append(np.linalg.norm(dfk))

    for k in range(niter):

        # compute objective and gradient
        f, df = f_df(x)

        # obtain next iterate
        pk = - H @ df
        sk = alpha * pk
        x += sk

        # update Hessian
        yk = df - dfk
        dfk = df
        syi = np.inner(sk, yk)
        syo = np.outer(sk, yk)
        qf = yk.T @ H @ yk
        H += ((syi + qf) * np.outer(sk, sk)) / (syi ** 2) - \
             (H @ syo.T + syo @ H) / syi

        store['obj'].append(f)
        store['gradnorm'].append(np.linalg.norm(df))

    return x, {k: np.array(v) for k, v in store.items()}
