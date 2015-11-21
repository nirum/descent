

@coroutine
def sgd(lr=1e-3):
    xk = yield
    while True:
        gradient = yield xk
        xk -= learning_rate * gradient

@coroutine
def printer():
    k = 0
    while True:
        k += 1
        data = yield
        print('[{}] Got:'.format(k), data)

@coroutine
def nonneg():
    while True:
        d

def gradient_loop(f_df, x0, algorithm, targets, maxiter=1000):

    # initialize
    xk = x0.copy()
    algorithm.send(xk)

    for k in range(maxiter):

        objective, gradient = f_df(xk)

        # next iterate
        xk = algorithm.send(gradient)

        # broadcast
        broadcast(xk, targets)

def consensus_loop():
    pass

def coroutine(func):
    def start(*args,**kwargs):
        cr = func(*args,**kwargs)
        cr.send(None)
        return cr
    return start

# def loop(x0, pipeline, maxiter=1000):
    # xk = deepcopy(x0)
    # for k in range(maxiter):
        # xk = pipe(xk, pipeline)
        # yield xk

def loop(x0, pipeline):

    # init
    x = deepcopy(x0)

    while True:
        for coro in pipeline:
            x = coro.send(x)
        yield x

def pipe(x, targets):
    for tgt in targets:
        x = tgt.send(x)
    return x

def broadcast(source, targets):
    for x in source:
        for tgt in targets:
            tgt.send(x)
