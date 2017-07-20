"""
Quasi-Newton methods
"""
from collections import defaultdict
from .line_search import backtracking
import numpy as np


def sdls(f_df, x0, niter=100, alpha0=1., rho=0.5, c=1e-4):
    """Steepest descent with line search"""

    # iterate
    x = x0.copy()

    # initial objective and gradient
    fk, dfk = f_df(x)

    # storate
    store = defaultdict(list)
    store['obj'].append(fk)
    store['x'].append(x.copy())
    store['gradnorm'].append(np.linalg.norm(dfk))

    for k in range(niter):

        # compute objective and gradient
        f, df = f_df(x)

        # line search
        alpha = backtracking(f_df, x, -df, alpha0, rho=rho, c=c)[0]

        # compute next iterate
        sk = alpha * df
        x -= sk

        store['obj'].append(f)
        store['x'].append(x.copy())
        store['gradnorm'].append(np.linalg.norm(df))

    return x, {k: np.stack(v) for k, v in store.items()}


def bfgs(f_df, x0, niter=100, alpha0=1., rho=0.5, c=1e-4):
    """BFGS Algorithm"""

    # initial inverse Hessian
    I = np.eye(x0.size)
    H = I.copy()

    # iterate
    x = x0.copy()

    # initial objective and gradient
    fk, dfk = f_df(x)

    # storate
    store = defaultdict(list)
    store['obj'].append(fk)
    store['x'].append(x.copy())
    store['gradnorm'].append(np.linalg.norm(dfk))

    for k in range(niter):

        # compute objective and gradient
        f, df = f_df(x)

        # search direction
        pk = - H @ df

        # line search
        alpha = backtracking(f_df, x, pk, alpha0, rho=rho, c=c)[0]

        # compute next iterate
        sk = alpha * pk
        x += sk

        # update Hessian
        yk = df - dfk
        dfk = df
        sy = 1 / np.inner(sk, yk)
        if np.isfinite(sy):
            H = (I - sy * np.outer(sk, yk)) @ H @ (I - sy * np.outer(yk, sk)) + sy * np.outer(sk, sk)

        store['obj'].append(f)
        store['x'].append(x.copy())
        store['gradnorm'].append(np.linalg.norm(df))

    return x, {k: np.stack(v) for k, v in store.items()}
