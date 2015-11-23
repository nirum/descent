from .connectors import run, select
from .utils import wrap, make_coroutine, destruct_coro, restruct_coro, destruct
from . import algorithms
from . import proxops


def gradient_descent(f_df, x0, alg='sgd', **kwargs):

    optimizer = getattr(algorithms, alg)(**kwargs)
    optimizer.send(destruct(x0))

    f_df_coro = make_coroutine(f_df)()

    pipeline = [f_df_coro, select(1), destruct_coro(), optimizer, restruct_coro(x0)]

    return run(x0, pipeline, maxiter=1000)
